/**
 * WebGL2 single-pass volume raycaster for a regular 3D-texture phasor field.
 *
 * Proxy geometry: the mesh-BBox cube, back-faces only so the fragment shader
 * runs once per silhouette pixel regardless of whether the camera is inside
 * or outside the volume. Ray-box intersection in the shader recovers the
 * true entry point; the camera-inside case is handled by clamping t_near to
 * zero.
 *
 * Compositing: standard front-to-back over-operator with premultiplied alpha
 * and early ray termination at α≥0.99. The host blends the result over the
 * prior scene with (ONE, ONE_MINUS_SRC_ALPHA) — i.e. the volume sits "on top
 * of" the opaque mesh, depth test disabled (passes through walls). Matches
 * the additive vibe of the point cloud while gaining proper inside-outside
 * masking via the occupancy channel.
 *
 * Texture format: RGBA32F where (R, G, B) = phasor (A, B, C) and A =
 * occupancy ∈ {0, 1}. The shader evaluates
 *      |F(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2C·cos(ωt)sin(ωt)
 * exactly like the point-cloud shader, then maps to colour via the inline
 * inferno LUT. TD mode reuses the same shader because its (s², s², 0)
 * encoding makes |F(t)|² collapse to s² independently of phase.
 */
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aCube;
uniform mat4 uMVP;
uniform vec3 uVolumeMin;
uniform vec3 uVolumeMax;
uniform float uZFlip;
out vec3 vWorld;
void main() {
	vWorld = mix(uVolumeMin, uVolumeMax, aCube);
	gl_Position = uMVP * vec4(vWorld.x, vWorld.y, uZFlip * vWorld.z, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vWorld;
uniform sampler3D uVolume;
uniform vec3 uVolumeMin;
uniform vec3 uVolumeMax;
uniform vec3 uEye;
uniform float uPhase;
uniform float uRangeFloor;
uniform float uRangeSpan;
uniform float uLogScale;
uniform float uOpacity;
uniform float uStepSize;
out vec4 fragColor;

// Inferno LUT, same one used by the point-cloud shader.
vec3 inferno(float t) {
	t = clamp(t, 0.0, 1.0);
	const vec3 c0 = vec3(0.0002, 0.0017, 0.0144);
	const vec3 c1 = vec3(0.1338, 0.0727, 0.3074);
	const vec3 c2 = vec3(0.7227, 0.2150, 0.3304);
	const vec3 c3 = vec3(0.9882, 0.6453, 0.0392);
	const vec3 c4 = vec3(0.9882, 1.0000, 0.6446);
	if (t < 0.25) return mix(c0, c1, t * 4.0);
	if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
	if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
	return mix(c3, c4, (t - 0.75) * 4.0);
}

// Ray-AABB slab test in world space. Returns vec2(t_near, t_far). Empty
// intersection when t_near > t_far.
vec2 ray_box(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
	vec3 inv = 1.0 / rd;
	vec3 t0 = (bmin - ro) * inv;
	vec3 t1 = (bmax - ro) * inv;
	vec3 lo = min(t0, t1);
	vec3 hi = max(t0, t1);
	return vec2(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

void main() {
	vec3 dir = normalize(vWorld - uEye);
	vec2 t = ray_box(uEye, dir, uVolumeMin, uVolumeMax);
	t.x = max(t.x, 0.0);
	if (t.x >= t.y) discard;

	vec3 box_size = uVolumeMax - uVolumeMin;
	vec3 entry_world = uEye + t.x * dir;
	vec3 exit_world  = uEye + t.y * dir;
	vec3 entry_tex = (entry_world - uVolumeMin) / box_size;
	vec3 exit_tex  = (exit_world  - uVolumeMin) / box_size;
	vec3 path = exit_tex - entry_tex;
	float path_len = length(path);
	if (path_len <= 0.0) discard;

	int n_steps = int(ceil(path_len / uStepSize));
	n_steps = min(n_steps, 512);
	float dt = path_len / float(n_steps);
	vec3 step_vec = path / float(n_steps);
	vec3 p = entry_tex + 0.5 * step_vec;

	float c = cos(uPhase);
	float s = sin(uPhase);
	float c2 = c * c;
	float s2 = s * s;
	float cs2 = 2.0 * c * s;

	vec4 dst = vec4(0.0);
	for (int i = 0; i < 512; i++) {
		if (i >= n_steps) break;
		vec4 vox = texture(uVolume, p);
		if (vox.a > 0.0) {
			float mag2 = max(vox.r * c2 + vox.g * s2 - vox.b * cs2, 0.0);
			float mag = sqrt(mag2);
			float u_lin = (mag - uRangeFloor) / max(uRangeSpan, 1e-9);
			float u_log = (log(max(mag, 1e-30)) / 2.302585093 - uRangeFloor) / max(uRangeSpan, 1e-9);
			float u = mix(u_lin, u_log, uLogScale);
			u = clamp(u, 0.0, 1.0);
			vec3 col = inferno(u);
			// Per-step opacity scales with field strength and integration
			// length; sample.a masks air to zero. The 4x boost compensates
			// for typical step counts (~128) so the volume "reads" at
			// reasonable opacity scales.
			float a = clamp(u * uOpacity * 4.0 * dt * vox.a, 0.0, 1.0);
			dst.rgb += (1.0 - dst.a) * a * col;
			dst.a   += (1.0 - dst.a) * a;
			if (dst.a > 0.99) break;
		}
		p += step_vec;
	}
	fragColor = dst;
}`;

export interface VolumeProgram {
	program: WebGLProgram;
	uMVP: WebGLUniformLocation;
	uVolumeMin: WebGLUniformLocation;
	uVolumeMax: WebGLUniformLocation;
	uEye: WebGLUniformLocation;
	uZFlip: WebGLUniformLocation;
	uVolume: WebGLUniformLocation;
	uPhase: WebGLUniformLocation;
	uRangeFloor: WebGLUniformLocation;
	uRangeSpan: WebGLUniformLocation;
	uLogScale: WebGLUniformLocation;
	uOpacity: WebGLUniformLocation;
	uStepSize: WebGLUniformLocation;
}

export interface VolumeBuffers {
	vao: WebGLVertexArrayObject;
	vbo: WebGLBuffer;
	ibo: WebGLBuffer;
	texture: WebGLTexture | null;
	resolution: number;
	min: [number, number, number];
	max: [number, number, number];
	uploaded: boolean;
	linear_float: boolean;
}

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
	const s = gl.createShader(type)!;
	gl.shaderSource(s, src);
	gl.compileShader(s);
	if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
		const info = gl.getShaderInfoLog(s);
		gl.deleteShader(s);
		throw new Error('Volume shader compile: ' + info);
	}
	return s;
}

export function createVolumeProgram(gl: WebGL2RenderingContext): VolumeProgram {
	const vs = compileShader(gl, gl.VERTEX_SHADER, VS);
	const fs = compileShader(gl, gl.FRAGMENT_SHADER, FS);
	const program = gl.createProgram()!;
	gl.attachShader(program, vs);
	gl.attachShader(program, fs);
	gl.linkProgram(program);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		throw new Error('Volume program link: ' + gl.getProgramInfoLog(program));
	}
	gl.deleteShader(vs);
	gl.deleteShader(fs);
	return {
		program,
		uMVP: gl.getUniformLocation(program, 'uMVP')!,
		uVolumeMin: gl.getUniformLocation(program, 'uVolumeMin')!,
		uVolumeMax: gl.getUniformLocation(program, 'uVolumeMax')!,
		uEye: gl.getUniformLocation(program, 'uEye')!,
		uZFlip: gl.getUniformLocation(program, 'uZFlip')!,
		uVolume: gl.getUniformLocation(program, 'uVolume')!,
		uPhase: gl.getUniformLocation(program, 'uPhase')!,
		uRangeFloor: gl.getUniformLocation(program, 'uRangeFloor')!,
		uRangeSpan: gl.getUniformLocation(program, 'uRangeSpan')!,
		uLogScale: gl.getUniformLocation(program, 'uLogScale')!,
		uOpacity: gl.getUniformLocation(program, 'uOpacity')!,
		uStepSize: gl.getUniformLocation(program, 'uStepSize')!,
	};
}

/** Unit-cube vertices and 12-triangle index list. CCW winding for outward
 *  faces — back-face culling (CULL_FRONT) keeps only the far silhouette,
 *  which the raycaster needs for correct entry/exit handling. */
const CUBE_VERTICES = new Float32Array([
	0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
	0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1,
]);
const CUBE_INDICES = new Uint16Array([
	0, 2, 1,  0, 3, 2,   // -Z
	4, 5, 6,  4, 6, 7,   // +Z
	0, 1, 5,  0, 5, 4,   // -Y
	2, 3, 7,  2, 7, 6,   // +Y
	1, 2, 6,  1, 6, 5,   // +X
	0, 4, 7,  0, 7, 3,   // -X
]);

export function createVolumeBuffers(gl: WebGL2RenderingContext): VolumeBuffers {
	const vao = gl.createVertexArray()!;
	gl.bindVertexArray(vao);
	const vbo = gl.createBuffer()!;
	gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
	gl.bufferData(gl.ARRAY_BUFFER, CUBE_VERTICES, gl.STATIC_DRAW);
	gl.enableVertexAttribArray(0);
	gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
	const ibo = gl.createBuffer()!;
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, CUBE_INDICES, gl.STATIC_DRAW);
	gl.bindVertexArray(null);
	gl.bindBuffer(gl.ARRAY_BUFFER, null);
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

	const linear_float =
		gl.getExtension('OES_texture_float_linear') !== null &&
		gl.getExtension('EXT_color_buffer_float') !== null;

	return {
		vao,
		vbo,
		ibo,
		texture: null,
		resolution: 0,
		min: [0, 0, 0],
		max: [0, 0, 0],
		uploaded: false,
		linear_float,
	};
}

/** Upload (or replace) the 3D texture from packed RGBA32F voxels. */
export function uploadVolumeData(
	gl: WebGL2RenderingContext,
	buf: VolumeBuffers,
	data: Float32Array,
	resolution: number,
	min: [number, number, number],
	max: [number, number, number],
): void {
	const t_start = typeof performance !== 'undefined' ? performance.now() : 0;
	if (!buf.texture) {
		buf.texture = gl.createTexture();
	}
	gl.bindTexture(gl.TEXTURE_3D, buf.texture);
	const filter = buf.linear_float ? gl.LINEAR : gl.NEAREST;
	gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
	gl.texImage3D(
		gl.TEXTURE_3D, 0, gl.RGBA32F,
		resolution, resolution, resolution, 0,
		gl.RGBA, gl.FLOAT, data,
	);
	gl.bindTexture(gl.TEXTURE_3D, null);
	buf.resolution = resolution;
	buf.min = [...min];
	buf.max = [...max];
	buf.uploaded = true;
	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t_start;
		console.log(`[volume] gl.texImage3D N=${resolution} ${(data.byteLength / 1048576).toFixed(1)} MB ${dt.toFixed(1)} ms`);
	}
}

export function clearVolumeData(buf: VolumeBuffers): void {
	buf.uploaded = false;
}

export function disposeVolume(
	gl: WebGL2RenderingContext,
	program: VolumeProgram,
	buf: VolumeBuffers,
): void {
	gl.deleteVertexArray(buf.vao);
	gl.deleteBuffer(buf.vbo);
	gl.deleteBuffer(buf.ibo);
	if (buf.texture) gl.deleteTexture(buf.texture);
	gl.deleteProgram(program.program);
}

/** Render the volume cube as the final pass. Caller has already drawn opaque
 *  meshes and lines, set up MVP, and disabled depth test as needed. */
export function renderVolume(
	gl: WebGL2RenderingContext,
	program: VolumeProgram,
	buf: VolumeBuffers,
	mvp: Float32Array,
	eye: [number, number, number],
	zFlip: number,
	phase: number,
	range_floor: number,
	range_span: number,
	log_scale: number,
	opacity: number,
): void {
	if (!buf.uploaded || !buf.texture) return;
	gl.useProgram(program.program);
	gl.uniformMatrix4fv(program.uMVP, false, mvp);
	gl.uniform3f(program.uVolumeMin, buf.min[0], buf.min[1], buf.min[2]);
	gl.uniform3f(program.uVolumeMax, buf.max[0], buf.max[1], buf.max[2]);
	gl.uniform3f(program.uEye, eye[0], eye[1], eye[2]);
	gl.uniform1f(program.uZFlip, zFlip);
	gl.uniform1f(program.uPhase, phase);
	gl.uniform1f(program.uRangeFloor, range_floor);
	gl.uniform1f(program.uRangeSpan, range_span);
	gl.uniform1f(program.uLogScale, log_scale);
	gl.uniform1f(program.uOpacity, opacity);
	gl.uniform1f(program.uStepSize, 1.0 / Math.max(buf.resolution, 1));
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_3D, buf.texture);
	gl.uniform1i(program.uVolume, 0);
	gl.disable(gl.DEPTH_TEST);
	gl.depthMask(false);
	gl.enable(gl.CULL_FACE);
	gl.cullFace(gl.FRONT);
	gl.enable(gl.BLEND);
	gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
	gl.bindVertexArray(buf.vao);
	gl.drawElements(gl.TRIANGLES, CUBE_INDICES.length, gl.UNSIGNED_SHORT, 0);
	gl.bindVertexArray(null);
	gl.disable(gl.BLEND);
	gl.cullFace(gl.BACK);
	gl.disable(gl.CULL_FACE);
	gl.depthMask(true);
	gl.enable(gl.DEPTH_TEST);
}
