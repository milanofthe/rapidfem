/// <reference lib="webworker" />
/**
 * Web Worker host for the heavy `volume_build_static` pass.
 *
 * Build cost grows as O(N³) in the grid resolution. The async wrapper
 * spawns a pool of these workers and partitions the volume by Z-slice;
 * each worker handles `[iz_start, iz_end)` and returns the slab's
 * `(tet_indices, bary)` partial buffers. The main thread stitches them
 * into one full-volume grid.
 *
 * Eval (`volume_eval_phasor` / `volume_eval_scalar`) stays on the main
 * thread: at 128³ those are 4–20 ms per call and the worker round-trip
 * would cost more than the saved frame budget for TD animation.
 */
import { volume_build_static_partition } from './volume_resample';
import type { MeshData } from './msh';

interface BuildRequest {
	request_id: number;
	kind: 'build';
	nodes: Float64Array;
	tets: Uint32Array;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	resolution: number;
	iz_start: number;
	iz_end: number;
}

export interface BuildResult {
	request_id: number;
	kind: 'build_result';
	tet_indices: Uint32Array;        // partial: length N · N · (iz_end - iz_start)
	bary: Float32Array;              // partial: length N · N · (iz_end - iz_start) · 4
	resolution: number;
	iz_start: number;
	iz_end: number;
	min: [number, number, number];
	max: [number, number, number];
}

declare const self: DedicatedWorkerGlobalScope;

self.addEventListener('message', (e: MessageEvent<BuildRequest>) => {
	const req = e.data;
	if (req.kind !== 'build') return;
	const mesh: MeshData = {
		nodes: req.nodes,
		tris: new Uint32Array(0),
		tri_phys: new Int32Array(0),
		tets: req.tets,
		tet_phys: new Int32Array(0),
		phys_names: new Map(),
		phys_dim: new Map(),
		bbox: req.bbox,
	};
	const part = volume_build_static_partition(mesh, req.resolution, req.iz_start, req.iz_end);
	const result: BuildResult = {
		request_id: req.request_id,
		kind: 'build_result',
		tet_indices: part.tet_indices,
		bary: part.bary,
		resolution: part.resolution,
		iz_start: part.iz_start,
		iz_end: part.iz_end,
		min: part.min,
		max: part.max,
	};
	self.postMessage(result, [part.tet_indices.buffer, part.bary.buffer]);
});
