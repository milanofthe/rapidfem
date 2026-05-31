/**
 * <fem-viewer> — Embeddable RapidFEM 3D viewer web component.
 *
 * Loads a baked example bundle (the `<name>.json` + `.bin` artefacts
 * produced by `scripts/bake_demo.py`) and renders it in any host page.
 *
 * Usage:
 *   <script src="https://fem.rapidpassives.org/embed/fem-viewer.js"></script>
 *   <fem-viewer src="/demo/fd_wr90.json" rotate cycle></fem-viewer>
 *
 * Attributes:
 *   src             URL to a baked `.json` (the matching `.bin` is auto-loaded)
 *   width / height  CSS dimensions (default 100% / 400px)
 *   rotate          continuous camera orbit
 *   cycle           cycle the display through Geometry → Mesh → Field
 *   mode            static display mode: `geometry` | `mesh` | `field`
 *                   (overridden by `cycle` if both are set; default `geometry`)
 *   interactive     enable mouse orbit/pan/zoom (default off)
 *   transparent     transparent background
 *   speed           animation speed multiplier (default 1)
 *   theta / phi     initial camera angles in degrees (default 45 / 45)
 *   field-mode      'lin' or 'log' (default 'lin')
 *   field-freq      frequency index for field display (default last)
 *   field-port      port index for field display (default 0)
 *   field-resolution  Cubic grid edge length for the volume raycaster
 *                     (default 96; 128 for full-page embeds; min 32)
 */

import {
	initGL, disposeGL, createCamera, clearMeshes, setTagVisible,
	setVolumeData, setVolumeRange, setVolumeScaleMode, clearVolume,
	render3D, fitCamera,
	type Camera, type GLState,
} from '../lib/render/canvas3d';
import {
	buildScene, clearFieldCloud, buildVolumeCloud, WIRE_TAG,
} from '../lib/render/scene_builder';
import {
	checkBinHeader, isBinRef, resolveFields, resolveGeoRefs, type BinRef,
} from '../lib/binpack';

// Cycle phases in display order; each holds for CYCLE_HOLD_S seconds.
type Phase = 'geometry' | 'mesh' | 'field';
const CYCLE_ORDER: Phase[] = ['geometry', 'mesh', 'field'];
const CYCLE_HOLD_S = 4.8;

interface MeshPayload {
	bbox: { min: [number, number, number]; max: [number, number, number] };
	nodes: number[];
	tris: number[];
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Record<string, string>;
	stats: { n_nodes: number; n_tets: number };
}
interface ResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_freq: number;
	fields?: BinRef | (number[] | null)[][];
}
interface BakedExample {
	cells: Array<{
		display_events: Array<{
			kind: 'geometry' | 'mesh' | 'result' | 'error';
			payload?: MeshPayload | ResultPayload | Record<string, unknown>;
		}>;
	}>;
}

function extractLatestMesh(baked: BakedExample): MeshPayload | null {
	for (let i = baked.cells.length - 1; i >= 0; i--) {
		for (const ev of baked.cells[i].display_events) {
			if (ev.kind === 'mesh' && ev.payload) return ev.payload as MeshPayload;
		}
	}
	return null;
}
function extractLatestResult(baked: BakedExample): ResultPayload | null {
	for (let i = baked.cells.length - 1; i >= 0; i--) {
		for (const ev of baked.cells[i].display_events) {
			if (ev.kind === 'result' && ev.payload) return ev.payload as ResultPayload;
		}
	}
	return null;
}

/** Fetch a `<name>.geo.bin` / `<name>.field.bin` sidecar next to the
 *  example JSON; `null` when the example carries no such buffer. */
async function fetchBinBuffer(jsonUrl: string, suffix: 'geo' | 'field'): Promise<ArrayBuffer | null> {
	const url = jsonUrl.replace(/\.json$/, `.${suffix}.bin`);
	const resp = await fetch(url);
	if (!resp.ok) return null;
	const buf = await resp.arrayBuffer();
	checkBinHeader(buf, url);
	return buf;
}

// (per-volume hull + edge extraction live in lib/render/mesh_scene.ts,
// shared with the in-app MeshViewer to keep the two pipelines bit-identical)

// (field sampling lives in scene_builder.ts — same algorithm the in-app
// viewer's viz.ts worker uses; we just inline the call here)

class FemViewerElement extends HTMLElement {
	private canvas: HTMLCanvasElement | null = null;
	private wrapper: HTMLDivElement | null = null;
	private loadingEl: HTMLDivElement | null = null;
	private labelEl: HTMLDivElement | null = null;
	private glState: GLState | null = null;
	private camera: Camera = createCamera();
	private animId = 0;
	private mounted = false;
	private mesh: MeshPayload | null = null;
	private fields: (number[] | null)[][] | null = null;
	private hasField = false;
	private needsRender = false;
	private isDragging = false;
	private isRightDrag = false;
	private lastMouse = { x: 0, y: 0 };
	private currentPhase: Phase = 'geometry';
	// Phase cycling reads performance.now() directly (no per-instance anchor),
	// so multiple <fem-viewer> tags on the same page tick in lockstep instead
	// of each drifting from its own load moment.
	// Perf state ────────────────────────────────────────────────────────
	private loadStarted = false;            // load() kicked off (lazy via IO)
	private isOnscreen = false;             // updated by IntersectionObserver
	private inObs: IntersectionObserver | null = null;
	private resObs: ResizeObserver | null = null;
	private faceTags: number[] = [];        // populated by buildScene
	private fieldCloudCache = new Map<string, {
		voxels: Float32Array;
		resolution: number;
		min: [number, number, number];
		max: [number, number, number];
		maxE2: number; minE2: number;
	}>();

	static get observedAttributes() {
		return ['src', 'width', 'height', 'rotate', 'cycle', 'mode', 'interactive',
		        'transparent', 'speed', 'theta', 'phi',
		        'field-mode', 'field-freq', 'field-port', 'field-resolution'];
	}

	connectedCallback() {
		this.mounted = true;
		const shadow = this.attachShadow({ mode: 'open' });
		const isTransparent = this.hasAttribute('transparent');
		this.wrapper = document.createElement('div');
		this.wrapper.style.cssText = `position:relative;width:${this.getAttribute('width') || '100%'};height:${this.getAttribute('height') || '400px'};background:${isTransparent ? 'transparent' : '#131316'};overflow:hidden;border-radius:inherit;`;
		this.canvas = document.createElement('canvas');
		this.canvas.style.cssText = `display:block;width:100%;height:100%;cursor:${this.hasAttribute('interactive') ? 'grab' : 'default'};`;
		this.wrapper.appendChild(this.canvas);

		this.loadingEl = document.createElement('div');
		this.loadingEl.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:500 11px/1 monospace;color:#55535a;';
		this.wrapper.appendChild(this.loadingEl);

		// Phase label (visible only when cycling)
		this.labelEl = document.createElement('div');
		this.labelEl.style.cssText = 'position:absolute;top:8px;left:10px;font:500 10px/1 monospace;color:#9a96a0;text-transform:uppercase;letter-spacing:0.5px;pointer-events:none;opacity:0;transition:opacity 0.3s;';
		this.wrapper.appendChild(this.labelEl);

		const badge = document.createElement('a');
		badge.href = 'https://fem.rapidpassives.org';
		badge.target = '_blank'; badge.rel = 'noopener';
		badge.textContent = 'RapidFEM';
		badge.style.cssText = 'position:absolute;bottom:6px;right:8px;font:500 9px/1 monospace;color:#55535a;text-decoration:none;opacity:0.7;transition:opacity 0.15s;';
		badge.onmouseenter = () => badge.style.opacity = '1';
		badge.onmouseleave = () => badge.style.opacity = '0.7';
		this.wrapper.appendChild(badge);

		shadow.appendChild(this.wrapper);

		this.glState = initGL(this.canvas);
		if (!this.glState) return;

		if (this.hasAttribute('interactive')) {
			this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
			this.canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
			this.canvas.addEventListener('pointerup', () => this.onPointerUp());
			this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
			this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
			this.canvas.addEventListener('dblclick', () => this.fitView());
		}
		this.resObs = new ResizeObserver(() => { this.needsRender = true; });
		this.resObs.observe(this.wrapper);

		// Defer fetch + animation until the embed is anywhere near the
		// viewport. On a 6-card landing page this means only the 2-3
		// cards above the fold load their JSON+bin on first paint; the
		// rest stream in as the user scrolls.
		const src = this.getAttribute('src');
		this.inObs = new IntersectionObserver((entries) => {
			for (const e of entries) {
				this.isOnscreen = e.isIntersecting;
				if (this.isOnscreen) {
					if (src && !this.loadStarted) {
						this.loadStarted = true;
						void this.load(src);
					} else if (this.mesh) {
						this.startAnimation();
					}
				} else {
					this.stopAnimation();
				}
			}
		}, { rootMargin: '200px' });
		this.inObs.observe(this);
	}

	disconnectedCallback() {
		this.mounted = false; this.animId++;
		this.inObs?.disconnect(); this.inObs = null;
		this.resObs?.disconnect(); this.resObs = null;
		if (this.glState) disposeGL(this.glState);
	}

	private stopAnimation() { this.animId++; }

	attributeChangedCallback(name: string, _old: string | null, val: string | null) {
		if (!this.mounted) return;
		if (name === 'src' && val) void this.load(val);
		else if (name === 'field-resolution') {
			this.fieldCloudCache.clear();   // resolution changed → invalidate
			this.applyPhaseInstant(this.resolvePhase());
			this.needsRender = true;
		}
		else if (name === 'mode' || name === 'field-mode' || name === 'field-freq' || name === 'field-port') {
			this.applyPhaseInstant(this.resolvePhase());
			this.needsRender = true;
		}
	}

	private onPointerDown(e: PointerEvent) {
		this.isDragging = true; this.isRightDrag = e.button === 2;
		this.lastMouse = { x: e.clientX, y: e.clientY };
		this.canvas?.setPointerCapture(e.pointerId);
		if (this.canvas) this.canvas.style.cursor = 'grabbing';
	}
	private onPointerMove(e: PointerEvent) {
		if (!this.isDragging) return;
		const dx = e.clientX - this.lastMouse.x;
		const dy = e.clientY - this.lastMouse.y;
		this.lastMouse = { x: e.clientX, y: e.clientY };
		if (this.isRightDrag) {
			const s = this.camera.distance * 0.0007;
			const ct = Math.cos(this.camera.theta), st = Math.sin(this.camera.theta);
			this.camera = {
				...this.camera,
				target: [
					this.camera.target[0] + (dx * ct - dy * st * Math.sin(this.camera.phi)) * s,
					this.camera.target[1] - (dx * st + dy * ct * Math.sin(this.camera.phi)) * s,
					this.camera.target[2] + dy * Math.cos(this.camera.phi) * s,
				],
			};
		} else {
			this.camera = {
				...this.camera,
				theta: this.camera.theta + dx * 0.005,
				phi: Math.max(0.05, Math.min(Math.PI / 2 - 0.05, this.camera.phi + dy * 0.005)),
			};
		}
		this.needsRender = true;
	}
	private onPointerUp() {
		this.isDragging = false; this.isRightDrag = false;
		if (this.canvas) this.canvas.style.cursor = 'grab';
	}
	private onWheel(e: WheelEvent) {
		e.preventDefault();
		this.camera = { ...this.camera, distance: this.camera.distance * (e.deltaY > 0 ? 1.1 : 1 / 1.1) };
		this.needsRender = true;
	}

	private fitView() {
		if (!this.glState || !this.mesh) return;
		this.camera = fitCamera(this.mesh.bbox.min, this.mesh.bbox.max);
		const theta = parseFloat(this.getAttribute('theta') || '45') * Math.PI / 180;
		const phi = parseFloat(this.getAttribute('phi') || '45') * Math.PI / 180;
		this.camera = { ...this.camera, theta, phi };
		this.needsRender = true;
	}

	private async load(srcUrl: string) {
		if (this.loadingEl) { this.loadingEl.textContent = 'Loading…'; this.loadingEl.style.display = 'flex'; }
		try {
			const url = new URL(srcUrl, location.href).href;
			const resp = await fetch(url);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const baked: BakedExample = await resp.json();

			// Resolve the geo buffer (mesh / geometry) for the whole bake.
			const geoBuf = await fetchBinBuffer(url, 'geo');
			if (geoBuf) {
				for (const cell of baked.cells) {
					for (const ev of cell.display_events) {
						if (ev.payload) {
							resolveGeoRefs(ev.payload as Record<string, unknown>, geoBuf);
						}
					}
				}
			}

			this.mesh = extractLatestMesh(baked);
			if (!this.mesh) throw new Error('no mesh payload in bake');

			const result = extractLatestResult(baked);
			if (result && result.fields && isBinRef(result.fields)) {
				if (this.loadingEl) this.loadingEl.textContent = 'Decoding field…';
				const fieldBuf = await fetchBinBuffer(url, 'field');
				this.fields = fieldBuf
					? resolveFields(fieldBuf, result.fields as BinRef)
					: null;
				this.hasField = !!this.fields?.some((row) => row.some((x) => x !== null));
			} else if (result && Array.isArray(result.fields)) {
				this.fields = result.fields as (number[] | null)[][];
				this.hasField = this.fields.some((row) => row.some((x) => x !== null));
			} else {
				this.fields = null;
				this.hasField = false;
			}

			this.buildSceneOnce();
			this.fitView();
			if (this.loadingEl) this.loadingEl.style.display = 'none';
			this.applyPhaseInstant(this.resolvePhase());
			if (this.isOnscreen) this.startAnimation();
		} catch (e) {
			console.error('[fem-viewer] load failed', e);
			if (this.loadingEl) this.loadingEl.textContent = `Error: ${(e as Error).message}`;
		}
	}

	/** Build the GL scene ONCE per load — faces + wireframe both
	 *  uploaded at the same time. Phase changes flip visibility flags
	 *  rather than re-uploading buffers, which makes cycle transitions
	 *  effectively free (was ~10–50 ms per phase × 6 cards × cycle). */
	private buildSceneOnce() {
		if (!this.glState || !this.mesh) return;
		clearMeshes(this.glState);
		this.fieldCloudCache.clear();
		const { faceTags } = buildScene(this.glState, this.mesh, {
			showFaces: true,
			showWire: true,
		});
		this.faceTags = faceTags;
		// Default to whatever the resolvePhase() returns for the initial
		// frame; the visibility flags then get set per-phase below.
		this.applyPhaseInstant(this.resolvePhase());
	}

	/** Toggle the resident scene's tag visibility instead of rebuilding. */
	private applyPhaseInstant(phase: Phase) {
		if (!this.glState) return;
		this.currentPhase = phase;
		const showFaces = phase === 'geometry';
		const showWire  = phase === 'mesh';
		for (const t of this.faceTags) setTagVisible(this.glState, t, showFaces);
		setTagVisible(this.glState, WIRE_TAG, showWire);
		if (phase === 'field' && this.hasField) this.applyField();
		else clearFieldCloud(this.glState);
		if (this.labelEl) {
			this.labelEl.textContent = phase;
			this.labelEl.style.opacity = this.hasAttribute('cycle') ? '0.65' : '0';
		}
	}

	private applyField() {
		if (!this.glState || !this.mesh) return;
		const clear = () => { if (this.glState) clearVolume(this.glState); };
		if (!this.hasField || !this.fields) { clear(); return; }
		const fi = Math.min(parseInt(this.getAttribute('field-freq') || '-1', 10), this.fields.length - 1);
		const fIdx = fi >= 0 ? fi : this.fields.length - 1;
		const pIdx = parseInt(this.getAttribute('field-port') || '0', 10);
		const row = this.fields[fIdx];
		const arr = row && row[pIdx];
		if (!arr) { clear(); return; }
		// Default 96³ for embeds keeps GPU memory in check on multi-card
		// landing pages (96³ × 16 B ≈ 14 MB per viewer). Bump for full-page
		// embeds via `field-resolution`.
		const N = Math.max(32, parseInt(this.getAttribute('field-resolution') || '96', 10));

		const key = `${fIdx}|${pIdx}|${N}`;
		let cached = this.fieldCloudCache.get(key);
		if (!cached) {
			cached = buildVolumeCloud(this.mesh, arr, N);
			this.fieldCloudCache.set(key, cached);
		}
		setVolumeData(this.glState, cached.voxels, cached.resolution, cached.min, cached.max);
		const mode = (this.getAttribute('field-mode') || 'lin') === 'log' ? 'log' as const : 'lin' as const;
		setVolumeScaleMode(this.glState, mode);
		if (mode === 'log') {
			const log_max = Math.log10(Math.sqrt(Math.max(cached.maxE2, 1e-30)));
			setVolumeRange(this.glState, log_max - 4, 4);
		} else {
			setVolumeRange(this.glState, 0, Math.sqrt(Math.max(cached.maxE2, 1e-30)));
		}
	}

	/** What phase should be displayed right now. */
	private resolvePhase(): Phase {
		if (this.hasAttribute('cycle')) {
			// Wall-clock-driven phase: every viewer reads the same `t`, so
			// multiple cards on a page advance in lockstep regardless of when
			// each one finished loading.
			const t = performance.now() / 1000;
			const order: Phase[] = this.hasField ? CYCLE_ORDER : CYCLE_ORDER.filter(p => p !== 'field');
			const idx = Math.floor(t / CYCLE_HOLD_S) % order.length;
			return order[idx];
		}
		const mode = (this.getAttribute('mode') || 'geometry').toLowerCase();
		if (mode === 'mesh' || mode === 'field') return mode as Phase;
		return 'geometry';
	}

	private syncCanvas(): { w: number; h: number } {
		if (!this.canvas) return { w: 0, h: 0 };
		const rect = this.canvas.getBoundingClientRect();
		const cssW = Math.round(rect.width), cssH = Math.round(rect.height);
		if (cssW <= 0 || cssH <= 0) return { w: 0, h: 0 };
		// Clamp dpr: a 3× retina canvas at 320 × 240 CSS = 960 × 720
		// backbuffer = 4× the pixel work of dpr=1.5. The visual delta on
		// an embed thumbnail is negligible; the perf delta is significant
		// (6 cards × 240 px tall × 60 fps × 3× ≫ same at 1.5×).
		const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
		const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
		if (this.canvas.width !== bw || this.canvas.height !== bh) {
			this.canvas.width = bw; this.canvas.height = bh;
			this.canvas.style.width = cssW + 'px';
			this.canvas.style.height = cssH + 'px';
		}
		// Return BACKBUFFER pixel size — render3D feeds this straight into
		// gl.viewport(0, 0, w, h). Passing CSS dimensions on a 2× hidpi
		// display would render into the bottom-left quarter only, which is
		// what "not centered" looked like.
		return { w: bw, h: bh };
	}

	private renderFrame() {
		if (!this.glState || !this.canvas || !this.mounted) return;
		const { w, h } = this.syncCanvas();
		if (w <= 0 || h <= 0) return;
		const isTransparent = this.hasAttribute('transparent');
		const speed = parseFloat(this.getAttribute('speed') || '1');
		if (this.hasAttribute('rotate') && !this.isDragging) {
			this.camera = { ...this.camera, theta: this.camera.theta + 0.003 * speed };
		}
		if (this.hasAttribute('cycle')) {
			const next = this.resolvePhase();
			if (next !== this.currentPhase) this.applyPhaseInstant(next);
		}
		// 5th arg is zFlip — MUST be 1 in the rapidfem renderer, otherwise the
		// vertex shader multiplies every vertex's z by 0 and the geometry
		// collapses to a horizontal slice at z=0. (Lines stay correct because
		// the line shader doesn't reference uZFlip.)
		render3D(this.glState, this.camera, w, h, 1);
		this.needsRender = false;
	}

	private startAnimation() {
		const id = ++this.animId;
		const animated = this.hasAttribute('rotate') || this.hasAttribute('cycle');
		if (!animated && !this.hasAttribute('interactive')) {
			this.renderFrame();
			return;
		}
		const tick = () => {
			if (!this.mounted || id !== this.animId) return;
			if (animated || this.needsRender) this.renderFrame();
			requestAnimationFrame(tick);
		};
		requestAnimationFrame(tick);
	}
}

if (typeof customElements !== 'undefined' && !customElements.get('fem-viewer')) {
	customElements.define('fem-viewer', FemViewerElement);
}
