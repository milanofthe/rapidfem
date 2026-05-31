/**
 * Lightweight raw-WebGL2 3D renderer for FEM tetrahedral / triangulated meshes.
 *
 * Adapted from rapidpassives/web/src/lib/render/canvas3d.ts — same shader,
 * camera math, and lighting model (single camera-following directional light
 * + 0.8 ambient, flat lit). Stripped of layout-specific code (LayerMap,
 * ProcessStack, instanced extrusion, GDS visibility, grid/axes) and replaced
 * with a triangle-mesh ingestion path that takes our parsed .msh data.
 */

import { canvas as canvasTheme } from '$lib/theme';
import {
	createVolumeProgram,
	createVolumeBuffers,
	uploadVolumeData,
	clearVolumeData,
	disposeVolume,
	renderVolume,
	type VolumeProgram,
	type VolumeBuffers,
} from './volume_render';

// ─── Types ───────────────────────────────────────────────────────────

export interface Camera {
	theta: number;     // azimuth (rad)
	phi: number;       // elevation (rad)
	distance: number;
	target: [number, number, number];
}

export function createCamera(): Camera {
	return { theta: -Math.PI / 4, phi: Math.PI / 5, distance: 300, target: [0, 0, 0] };
}

interface Mesh {
	vao: WebGLVertexArrayObject;
	buffers: WebGLBuffer[];
	count: number;
	color: [number, number, number];
	tag: number;
	visible: boolean;
	depth_offset?: [number, number];
	/** Has a per-vertex scalar buffer attached (location=2). When true, the
	 *  shader uses Viridis colormap on it instead of the flat color. */
	has_scalar?: boolean;
}

interface LineMesh {
	vao: WebGLVertexArrayObject;
	buffers: WebGLBuffer[];
	count: number;
	color: [number, number, number];
	tag: number;
	visible: boolean;
}

export interface GLState {
	gl: WebGL2RenderingContext;
	program: WebGLProgram;
	uMVP: WebGLUniformLocation;
	uNormalMat: WebGLUniformLocation;
	uColor: WebGLUniformLocation;
	uLightDir: WebGLUniformLocation;
	uAmbient: WebGLUniformLocation;
	uZFlip: WebGLUniformLocation;
	uColormap: WebGLUniformLocation;
	uClipPlane: WebGLUniformLocation;
	uClipEnable: WebGLUniformLocation;
	clip_plane: [number, number, number, number];
	clip_enable: boolean;
	lineProgram: WebGLProgram;
	uLineMVP: WebGLUniformLocation;
	uLineColor: WebGLUniformLocation;
	volumeProgram: VolumeProgram;
	volumeBuffers: VolumeBuffers;
	volumePhase: number;
	volumeRangeFloor: number;
	volumeRangeSpan: number;
	volumeLogScale: number;
	volumeOpacity: number;
	volumeSmoothing: number;
	meshes: Mesh[];
	lineMeshes: LineMesh[];
	bbox: { min: [number, number, number]; max: [number, number, number] };
}

// ─── Shaders (verbatim from rapidpassives) ──────────────────────────

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aScalar;
uniform mat4 uMVP;
uniform mat3 uNormalMat;
uniform float uZFlip;
out vec3 vNormal;
out float vScalar;
out vec3 vWorld;
void main() {
	vec3 n = aNormal;
	n.z *= uZFlip;
	vNormal = normalize(uNormalMat * n);
	vec3 pos = aPos;
	pos.z *= uZFlip;
	vWorld = pos;
	gl_Position = uMVP * vec4(pos, 1.0);
	vScalar = aScalar;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vNormal;
in float vScalar;
in vec3 vWorld;
uniform vec3 uColor;
uniform vec3 uLightDir;
uniform float uAmbient;
uniform float uColormap;
uniform vec4 uClipPlane;          // (nx, ny, nz, d): discard fragments where dot(world, n) > d
uniform float uClipEnable;
out vec4 fragColor;

// Polynomial Inferno colormap — black → purple → red → orange → yellow → white.
// Matches our warm rapidpassives palette better than viridis.
vec3 inferno(float t) {
	t = clamp(t, 0.0, 1.0);
	const vec3 c0 = vec3(0.0002, 0.0016, -0.0194);
	const vec3 c1 = vec3(0.1065, 0.5639, 3.9327);
	const vec3 c2 = vec3(11.6024, -3.972, -15.9423);
	const vec3 c3 = vec3(-41.7039, 17.4363, 44.354);
	const vec3 c4 = vec3(77.1629, -33.4023, -81.8073);
	const vec3 c5 = vec3(-71.319, 32.6261, 73.2095);
	const vec3 c6 = vec3(25.1311, -12.2426, -23.0703);
	return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

void main() {
	if (uClipEnable > 0.5) {
		if (dot(vWorld, uClipPlane.xyz) > uClipPlane.w) discard;
	}
	float diff = max(dot(normalize(vNormal), uLightDir), 0.0);
	vec3 base = mix(uColor, inferno(vScalar), uColormap);
	vec3 lit = base * (uAmbient + (1.0 - uAmbient) * diff);
	fragColor = vec4(lit, 1.0);
}`;

const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`;

const LINE_FS = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }`;

// ─── GL helpers ─────────────────────────────────────────────────────

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
	const s = gl.createShader(type)!;
	gl.shaderSource(s, src);
	gl.compileShader(s);
	if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
		const info = gl.getShaderInfoLog(s);
		gl.deleteShader(s);
		throw new Error('Shader compile: ' + info);
	}
	return s;
}

function linkProgram(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string): WebGLProgram {
	const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
	const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
	const p = gl.createProgram()!;
	gl.attachShader(p, vs);
	gl.attachShader(p, fs);
	gl.linkProgram(p);
	if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
		throw new Error('Program link: ' + gl.getProgramInfoLog(p));
	}
	gl.deleteShader(vs);
	gl.deleteShader(fs);
	return p;
}

function hexToRgb(hex: string): [number, number, number] {
	const r = parseInt(hex.slice(1, 3), 16) / 255;
	const g = parseInt(hex.slice(3, 5), 16) / 255;
	const b = parseInt(hex.slice(5, 7), 16) / 255;
	return [r, g, b];
}

// ─── Matrix math ────────────────────────────────────────────────────

type Mat4 = Float32Array;

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
	const f = 1 / Math.tan(fovY / 2);
	const nf = 1 / (near - far);
	const m = new Float32Array(16);
	m[0] = f / aspect; m[5] = f;
	m[10] = (far + near) * nf; m[11] = -1;
	m[14] = 2 * far * near * nf;
	return m;
}

function mat4LookAt(eye: number[], center: number[], up: number[]): Mat4 {
	const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
	let len = Math.sqrt(zx * zx + zy * zy + zz * zz);
	const z0 = zx / len, z1 = zy / len, z2 = zz / len;
	const xx = up[1] * z2 - up[2] * z1, xy = up[2] * z0 - up[0] * z2, xz = up[0] * z1 - up[1] * z0;
	len = Math.sqrt(xx * xx + xy * xy + xz * xz);
	const x0 = xx / len, x1 = xy / len, x2 = xz / len;
	const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
	const m = new Float32Array(16);
	m[0] = x0; m[1] = y0; m[2] = z0;
	m[4] = x1; m[5] = y1; m[6] = z1;
	m[8] = x2; m[9] = y2; m[10] = z2;
	m[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
	m[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
	m[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
	m[15] = 1;
	return m;
}

function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
	const o = new Float32Array(16);
	for (let i = 0; i < 4; i++) {
		for (let j = 0; j < 4; j++) {
			o[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
		}
	}
	return o;
}

function mat3NormalFromMat4(m: Mat4): Float32Array {
	const n = new Float32Array(9);
	n[0] = m[0]; n[1] = m[1]; n[2] = m[2];
	n[3] = m[4]; n[4] = m[5]; n[5] = m[6];
	n[6] = m[8]; n[7] = m[9]; n[8] = m[10];
	return n;
}

// ─── Camera → eye ───────────────────────────────────────────────────

export function cameraEye(cam: Camera): [number, number, number] {
	const cp = Math.cos(cam.phi);
	return [
		cam.target[0] + cam.distance * cp * Math.sin(cam.theta),
		cam.target[1] + cam.distance * cp * Math.cos(cam.theta),
		cam.target[2] + cam.distance * Math.sin(cam.phi)
	];
}

// ─── Public API ─────────────────────────────────────────────────────

export function initGL(canvas: HTMLCanvasElement): GLState | null {
	// No preserveDrawingBuffer: it forces an extra backbuffer copy every
	// frame. save_png() renders and reads the buffer synchronously within
	// one task, so the buffer is still intact when toBlob() snapshots it.
	const gl = canvas.getContext('webgl2', { antialias: true, alpha: true });
	if (!gl) return null;

	const program = linkProgram(gl, VS, FS);
	const lineProgram = linkProgram(gl, LINE_VS, LINE_FS);
	const volumeProgram = createVolumeProgram(gl);
	const volumeBuffers = createVolumeBuffers(gl);

	const bg = hexToRgb(canvasTheme.bg);
	gl.clearColor(bg[0], bg[1], bg[2], 1);
	gl.enable(gl.DEPTH_TEST);
	gl.disable(gl.CULL_FACE);

	return {
		gl,
		program,
		uMVP: gl.getUniformLocation(program, 'uMVP')!,
		uNormalMat: gl.getUniformLocation(program, 'uNormalMat')!,
		uColor: gl.getUniformLocation(program, 'uColor')!,
		uLightDir: gl.getUniformLocation(program, 'uLightDir')!,
		uAmbient: gl.getUniformLocation(program, 'uAmbient')!,
		uZFlip: gl.getUniformLocation(program, 'uZFlip')!,
		uColormap: gl.getUniformLocation(program, 'uColormap')!,
		uClipPlane: gl.getUniformLocation(program, 'uClipPlane')!,
		uClipEnable: gl.getUniformLocation(program, 'uClipEnable')!,
		clip_plane: [0, 0, 1, 0],
		clip_enable: false,
		lineProgram,
		uLineMVP: gl.getUniformLocation(lineProgram, 'uMVP')!,
		uLineColor: gl.getUniformLocation(lineProgram, 'uColor')!,
		volumeProgram,
		volumeBuffers,
		volumePhase: 0,
		volumeRangeFloor: -30,
		volumeRangeSpan: 6,
		volumeLogScale: 0,
		volumeOpacity: 1.0,
		volumeSmoothing: 0.0,
		meshes: [],
		lineMeshes: [],
		bbox: { min: [0, 0, 0], max: [0, 0, 0] }
	};
}

export function disposeGL(state: GLState): void {
	const { gl } = state;
	for (const m of state.meshes) {
		gl.deleteVertexArray(m.vao);
		for (const b of m.buffers) gl.deleteBuffer(b);
	}
	for (const m of state.lineMeshes) {
		gl.deleteVertexArray(m.vao);
		for (const b of m.buffers) gl.deleteBuffer(b);
	}
	disposeVolume(gl, state.volumeProgram, state.volumeBuffers);
	gl.deleteProgram(state.program);
	gl.deleteProgram(state.lineProgram);
}

/** Drop surface and line meshes. The volume cloud has its own lifecycle
 *  (setVolumeData / clearVolume manage its GL resources) and survives a
 *  rebuild so the field viz stays visible when the user toggles Geometry
 *  or Mesh layers. */
export function clearMeshes(state: GLState): void {
	const { gl } = state;
	for (const m of state.meshes) {
		gl.deleteVertexArray(m.vao);
		for (const b of m.buffers) gl.deleteBuffer(b);
	}
	state.meshes = [];
	for (const m of state.lineMeshes) {
		gl.deleteVertexArray(m.vao);
		for (const b of m.buffers) gl.deleteBuffer(b);
	}
	state.lineMeshes = [];
}

/** Upload (or replace) the volumetric field as a 3D RGBA32F texture.
 *
 *  `voxels` is the packed `(A, B, C, occ)` buffer produced by
 *  `volume_eval_phasor` (FD) or `volume_eval_scalar` (TD). Resolution is the
 *  cubic grid edge length (data.length must equal 4·N³). min/max are the
 *  world-space BBox the grid covers. */
export function setVolumeData(
	state: GLState,
	voxels: Float32Array,
	resolution: number,
	min: [number, number, number],
	max: [number, number, number],
): void {
	if (voxels.length === 0 || resolution === 0) {
		clearVolumeData(state.volumeBuffers);
		return;
	}
	uploadVolumeData(state.gl, state.volumeBuffers, voxels, resolution, min, max);
}

/** Clear the volume texture (e.g. when the user turns the field off). */
export function clearVolume(state: GLState): void {
	clearVolumeData(state.volumeBuffers);
}

/** Set the field colour-mapping range. floor/span are interpreted per mode:
 *  log → (log10(|F|min), log10(max/min)); lin → (|F|min, |F|max−|F|min). */
export function setVolumeRange(state: GLState, floor: number, span: number): void {
	state.volumeRangeFloor = floor;
	state.volumeRangeSpan = span;
}

/** Colour-mapping mode for the volume field. */
export function setVolumeScaleMode(state: GLState, mode: 'log' | 'lin'): void {
	state.volumeLogScale = mode === 'log' ? 1 : 0;
}

/** Update the time phase (call from requestAnimationFrame for the wave anim). */
export function setVolumePhase(state: GLState, phase: number): void {
	state.volumePhase = phase;
}

/** Global opacity multiplier for the volume (1.0 = default). */
export function setVolumeOpacity(state: GLState, opacity: number): void {
	state.volumeOpacity = opacity;
}

/** Mip-LOD bias for the volume sampler. 0 = sharp (mip 0 only); 1.0 ≈ one
 *  full mip level blurred. Use to hide P1 gradient discontinuities at tet
 *  faces in coarse-mesh regions. */
export function setVolumeSmoothing(state: GLState, smoothing: number): void {
	state.volumeSmoothing = smoothing;
}

/** Add a triangle group with a single color. positions and normals must
 *  contain 3 components per vertex, in matching order. `tag` is the physical
 *  group integer used for visibility toggling via setTagVisible. */
export function addMesh(
	state: GLState,
	positions: Float32Array,
	normals: Float32Array,
	color: [number, number, number],
	tag = 0,
	depth_offset?: [number, number],
	scalars?: Float32Array          // one scalar per vertex (range [0,1] for viridis)
): void {
	const { gl } = state;
	const vao = gl.createVertexArray()!;
	gl.bindVertexArray(vao);

	const posBuf = gl.createBuffer()!;
	gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
	gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
	gl.enableVertexAttribArray(0);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

	const normBuf = gl.createBuffer()!;
	gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
	gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
	gl.enableVertexAttribArray(1);
	gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

	const buffers = [posBuf, normBuf];
	let has_scalar = false;
	if (scalars) {
		const sBuf = gl.createBuffer()!;
		gl.bindBuffer(gl.ARRAY_BUFFER, sBuf);
		gl.bufferData(gl.ARRAY_BUFFER, scalars, gl.STATIC_DRAW);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
		buffers.push(sBuf);
		has_scalar = true;
	} else {
		// Provide a constant 0 for location=2 so the shader's vScalar is defined
		gl.disableVertexAttribArray(2);
		gl.vertexAttrib1f(2, 0);
	}

	gl.bindVertexArray(null);
	state.meshes.push({ vao, buffers, count: positions.length / 3, color, tag, visible: true, depth_offset, has_scalar });
}

/** Line segments. positions: 3 components per vertex, every two vertices = one segment. */
export function addLineMesh(state: GLState, positions: Float32Array, color: [number, number, number], tag = 0): void {
	const { gl } = state;
	const vao = gl.createVertexArray()!;
	gl.bindVertexArray(vao);

	const buf = gl.createBuffer()!;
	gl.bindBuffer(gl.ARRAY_BUFFER, buf);
	gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
	gl.enableVertexAttribArray(0);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

	gl.bindVertexArray(null);
	state.lineMeshes.push({ vao, buffers: [buf], count: positions.length / 3, color, tag, visible: true });
}

/** Toggle visibility of all meshes whose tag matches. */
export function setTagVisible(state: GLState, tag: number, visible: boolean): void {
	for (const m of state.meshes) if (m.tag === tag) m.visible = visible;
	for (const m of state.lineMeshes) if (m.tag === tag) m.visible = visible;
}

/** Set the clipping plane: discards fragments where dot(world, n) > d. */
export function setClipPlane(state: GLState, normal: [number, number, number], d: number, enable: boolean) {
	state.clip_plane = [normal[0], normal[1], normal[2], d];
	state.clip_enable = enable;
}

export function setBBox(state: GLState, min: [number, number, number], max: [number, number, number]): void {
	state.bbox.min = min;
	state.bbox.max = max;
}

/** Render one frame. */
export function render3D(
	state: GLState,
	camera: Camera,
	w: number,
	h: number,
	zFlip = 1
): void {
	const { gl } = state;
	gl.viewport(0, 0, w, h);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	const aspect = w / h || 1;
	// Near/far based on bbox extent so the substrate doesn't get clipped.
	const dx = state.bbox.max[0] - state.bbox.min[0];
	const dy = state.bbox.max[1] - state.bbox.min[1];
	const dz = state.bbox.max[2] - state.bbox.min[2];
	const sceneR = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
	const near = Math.max(camera.distance * 1e-3, sceneR * 1e-3, 1e-9);
	const far = (camera.distance + sceneR) * 8;
	const proj = mat4Perspective(Math.PI / 6, aspect, near, far);

	const eye = cameraEye(camera);
	const view = mat4LookAt(eye, camera.target as number[], [0, 0, 1]);
	const vp = mat4Multiply(proj, view);
	const normalMat = mat3NormalFromMat4(view);

	// Light follows camera, slightly offset upward — same recipe as rapidpassives
	const ldx = eye[0] - camera.target[0];
	const ldy = eye[1] - camera.target[1];
	const ldz = eye[2] - camera.target[2] + camera.distance * 0.3;
	const ldLen = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
	const lightDir: [number, number, number] = [ldx / ldLen, ldy / ldLen, ldz / ldLen];

	// Solid lit pass
	gl.useProgram(state.program);
	gl.uniformMatrix4fv(state.uMVP, false, vp);
	gl.uniformMatrix3fv(state.uNormalMat, false, normalMat);
	gl.uniform3f(state.uLightDir, lightDir[0], lightDir[1], lightDir[2]);
	gl.uniform1f(state.uAmbient, 0.8);
	gl.uniform1f(state.uZFlip, zFlip);
	gl.uniform4f(state.uClipPlane, state.clip_plane[0], state.clip_plane[1], state.clip_plane[2], state.clip_plane[3]);
	gl.uniform1f(state.uClipEnable, state.clip_enable ? 1.0 : 0.0);
	let offset_on = false;
	for (const m of state.meshes) {
		if (!m.visible) continue;
		if (m.depth_offset) {
			if (!offset_on) { gl.enable(gl.POLYGON_OFFSET_FILL); offset_on = true; }
			gl.polygonOffset(m.depth_offset[0], m.depth_offset[1]);
		} else if (offset_on) {
			gl.disable(gl.POLYGON_OFFSET_FILL);
			offset_on = false;
		}
		gl.uniform3f(state.uColor, m.color[0], m.color[1], m.color[2]);
		gl.uniform1f(state.uColormap, m.has_scalar ? 1.0 : 0.0);
		gl.bindVertexArray(m.vao);
		gl.drawArrays(gl.TRIANGLES, 0, m.count);
	}
	if (offset_on) gl.disable(gl.POLYGON_OFFSET_FILL);

	// Line pass (wireframe / axes / grid)
	if (state.lineMeshes.length > 0) {
		gl.useProgram(state.lineProgram);
		gl.uniformMatrix4fv(state.uLineMVP, false, vp);
		for (const lm of state.lineMeshes) {
			if (!lm.visible) continue;
			gl.uniform3f(state.uLineColor, lm.color[0], lm.color[1], lm.color[2]);
			gl.bindVertexArray(lm.vao);
			gl.drawArrays(gl.LINES, 0, lm.count);
		}
	}

	// Volumetric field as a 3D-texture raycaster. Draws the mesh-BBox cube
	// (back faces only) and integrates front-to-back through the volume in
	// the fragment shader. Depth test off, premultiplied-alpha blend over
	// the prior scene.
	if (state.volumeBuffers.uploaded) {
		renderVolume(
			gl,
			state.volumeProgram,
			state.volumeBuffers,
			vp,
			eye,
			zFlip,
			state.volumePhase,
			state.volumeRangeFloor,
			state.volumeRangeSpan,
			state.volumeLogScale,
			state.volumeOpacity,
			state.volumeSmoothing,
		);
	}
	gl.bindVertexArray(null);
}

/** Compute camera that fits the bbox in view, theta=phi=π/4 iso, FOV=30°. */
export function fitCamera(min: [number, number, number], max: [number, number, number]): Camera {
	const cx = (min[0] + max[0]) / 2;
	const cy = (min[1] + max[1]) / 2;
	const cz = (min[2] + max[2]) / 2;
	const dx = max[0] - min[0];
	const dy = max[1] - min[1];
	const dz = max[2] - min[2];
	// Use the bbox diagonal so all three extents fit, not just xy. The 0.6
	// factor approximates the silhouette extent at a 45° viewing angle (the
	// projected bbox onto the camera plane is < the full diagonal).
	const diag = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9);
	const halfFov = Math.PI / 12; // half of the 30° perspective FOV
	const distance = (diag * 0.6) / Math.tan(halfFov) * 1.05;
	return { theta: Math.PI / 4, phi: Math.PI / 4, distance, target: [cx, cy, cz] };
}
