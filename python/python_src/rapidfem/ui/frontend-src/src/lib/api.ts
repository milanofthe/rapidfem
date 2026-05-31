/**
 * Backend client for the rapidfem UI.
 *
 * Talks to the Flask app started by `rapidfem serve`:
 *   GET /api/files, PUT/DELETE /api/files/<path>  (workdir file access)
 *   GET /api/examples, GET /api/examples/<name>   (shipped examples)
 *   GET /api/health
 *
 * Cell execution runs through the subprocess kernel in `kernel.ts`.
 */

import type { MeshData } from './msh';

// ── URL base ──────────────────────────────────────────────────────────────

// In production the SvelteKit static build is served by the same Flask
// process at /, so relative URLs work. In dev (vite at :5173) we point at
// the Flask port directly.
const DEV_BACKEND = 'http://127.0.0.1:5174';

export function api_base(): string {
	if (typeof window !== 'undefined' && window.location.port === '5173') {
		return DEV_BACKEND;
	}
	return '';
}

// ── Shared types ──────────────────────────────────────────────────────────

export type SParam = { re: number; im: number };
export type SMatrix = SParam[][]; // [obs][exc]

export interface PythonError {
	type: string;
	message: string;
	traceback: string;
}

export interface GeometryEntity {
	name: string;
	tag: number;
	dim: number;
	color: [number, number, number];
	// Triangle mode (after g.mesh()): flat xyz, 3 verts × 3 floats per tri.
	positions?: number[];
	normals?: number[];
	// Wireframe mode (before g.mesh()): flat xyz pairs, 2 verts per segment.
	lines?: number[];
	material: string | null;
}

export interface GeometryPayload {
	kind: 'geometry';
	wireframe?: boolean;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	entities: GeometryEntity[];
	stats: { n_entities: number; n_segments?: number; n_triangles?: number; maxh: number };
}

export interface MeshPayload {
	kind: 'mesh';
	bbox: { min: [number, number, number]; max: [number, number, number] };
	nodes: number[];           // flat xyz
	tris: number[];            // flat 3 × idx
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Record<string, string>;
	phys_dim: Record<string, number>;
	name_to_tag: Record<string, number>;
	stats: { n_nodes: number; n_tets: number; n_tris: number; mesh_time_s: number; msh_bytes: number };
}

export interface FileEntry {
	path: string;
	size: number;
	mtime: number;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────

async function post_json<T>(path: string, body: unknown): Promise<T> {
	const res = await fetch(api_base() + path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	return res.json();
}

async function get_json<T>(path: string): Promise<T> {
	const res = await fetch(api_base() + path);
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	return res.json();
}

// ── Public API ────────────────────────────────────────────────────────────

export async function listExamples(): Promise<{ examples: { name: string }[] }> {
	return get_json<{ examples: { name: string }[] }>('/api/examples');
}

export async function readExample(name: string): Promise<string> {
	// In static-demo mode the example source lives inside the baked JSON.
	const { IS_STATIC_MODE, DEMO_BASE } = await import('./static_mode');
	if (IS_STATIC_MODE) {
		const stem = name.replace(/\.py$/, '');
		const r = await fetch(`${DEMO_BASE}${stem}.json`);
		if (!r.ok) throw new Error(`baked example fetch failed: ${r.status}`);
		const baked = await r.json() as { source: string };
		return baked.source;
	}
	const r = await get_json<{ ok: boolean; content?: string; error?: string }>(`/api/examples/${encodeURIComponent(name)}`);
	if (!r.ok || r.content === undefined) throw new Error(r.error ?? 'example not found');
	return r.content;
}

export async function listFiles(): Promise<{ workdir: string; files: FileEntry[] }> {
	return get_json<{ workdir: string; files: FileEntry[] }>('/api/files');
}

export async function readFile(path: string): Promise<string> {
	const r = await get_json<{ ok: boolean; content?: string; error?: string }>(
		`/api/files/${encodeURI(path)}`,
	);
	if (!r.ok || r.content === undefined) throw new Error(r.error ?? 'read failed');
	return r.content;
}

export async function writeFile(path: string, content: string): Promise<void> {
	const res = await fetch(api_base() + `/api/files/${encodeURI(path)}`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ content }),
	});
	if (!res.ok) throw new Error(`writeFile ${path}: HTTP ${res.status}`);
}

export async function deleteFile(path: string): Promise<void> {
	const res = await fetch(api_base() + `/api/files/${encodeURI(path)}`, { method: 'DELETE' });
	if (!res.ok) throw new Error(`deleteFile ${path}: HTTP ${res.status}`);
}

export async function renameFile(from: string, to: string): Promise<void> {
	return post_json('/api/files/rename', { from, to });
}

export async function health(): Promise<{ ok: boolean; workdir: string; frontend_bundled: boolean }> {
	return get_json('/api/health');
}

// ── Adapters: server payload → frontend types ─────────────────────────────

export function meshPayloadToMeshData(p: MeshPayload): MeshData {
	const nodes = new Float64Array(p.nodes);
	const tris = new Uint32Array(p.tris);
	const tri_phys = new Int32Array(p.tri_phys);
	const tets = new Uint32Array(p.tets);
	const tet_phys = new Int32Array(p.tet_phys);
	const phys_names = new Map<number, string>();
	for (const [k, v] of Object.entries(p.phys_names)) phys_names.set(Number(k), v);
	const phys_dim = new Map<number, number>();
	for (const [k, v] of Object.entries(p.phys_dim)) phys_dim.set(Number(k), v);
	return {
		nodes, tris, tri_phys, tets, tet_phys,
		phys_names, phys_dim,
		bbox: { min: [...p.bbox.min] as [number, number, number], max: [...p.bbox.max] as [number, number, number] },
	};
}

export function sparamsToSMatrices(s: number[][][][]): SMatrix[] {
	return s.map((freq_mat) =>
		freq_mat.map((row) => row.map(([re, im]) => ({ re, im }))),
	);
}

// ── Time-domain display payloads ──────────────────────────────────────────
// Emitted by rapidfem.show() for the ProblemTD verb results — see
// api._td_*_payload on the Python side.

/** One trace of a time-series plot. A time-domain probe carries a real
 *  `y`; a frequency-domain transfer function carries a complex
 *  `y_re` / `y_im` pair. */
export interface TdSeries {
	label: string;
	y?: number[];
	y_re?: number[];
	y_im?: number[];
}

/** `td_timeseries` payload — driven_transient probe signals (`domain:'time'`)
 *  or a transfer function (`domain:'freq'`). */
export interface TdTimeSeriesPayload {
	domain: 'time' | 'freq';
	x_label: string;
	x: number[];
	series: TdSeries[];
	source_label: string;
}

/** `td_result` payload — a time-domain modal-port scattering matrix. Carries
 *  the same nested-list shape as the frequency-domain result, so it feeds the
 *  existing S-parameter panel unchanged. */
export interface TdResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_port: number;
	n_freq: number;
}

/** `td_trajectory` payload — a self-contained DG-corner mesh plus a
 *  per-node field magnitude per snapshot. The frontend samples a point
 *  cloud from `nodes` / `tets` at runtime (energy-weighted, like the FD
 *  field viz) at the user-picked density, then evaluates `frames_e` /
 *  `frames_h` at those fixed sample points. The per-node magnitudes are
 *  quantised to integers 0…1000 of `field_max` (the global per-channel
 *  maximum — rescale by `field_max/1000`, the colour scale stays fixed
 *  across the whole animation). */
export interface TdTrajectoryPayload {
	nodes: number[];           // flat xyz of the deduplicated DG corners
	tets: number[];            // flat 4 × node-idx
	n_node: number;
	n_elem: number;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	n_snapshots: number;
	times: number[];
	field_max: { E: number; H: number };
	frames_e: number[][];      // [n_snap][n_node], quantised 0…1000
	frames_h: number[][];
}

// Note: the old `viz` point-cloud sampler API is gone — the field viz now
// runs through `volume_resample` and the canvas3d volume raycaster.
export {
	volume_build_static, volume_eval_phasor, volume_eval_scalar,
	volume_energy_range,
} from './volume_resample';
