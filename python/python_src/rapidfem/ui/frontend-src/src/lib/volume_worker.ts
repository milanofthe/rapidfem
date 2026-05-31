/// <reference lib="webworker" />
/**
 * Web Worker host for the heavy `volume_build_static` pass.
 *
 * Build cost grows as O(N³) in the grid resolution: at 96³ a 25k-tet mesh
 * lands at ~40 ms, but 128³ pushes that to ~100 ms and 192³ to >300 ms —
 * enough to drop frames on the main thread. Moving the build into a worker
 * keeps the canvas smooth at any resolution the user picks.
 *
 * Eval (`volume_eval_phasor` / `volume_eval_scalar`) stays on the main
 * thread: it's already 4–20 ms per call and the worker round-trip would
 * cost more than the saved frame budget for TD animation.
 *
 * Protocol: one `build` message in, one `build_result` message out. Output
 * buffers are transferred (zero-copy); the input mesh is structured-cloned
 * so the caller keeps its references (the main thread still needs `tets` /
 * `nodes` for the eval passes).
 */
import { volume_build_static } from './volume_resample';
import type { MeshData } from './msh';

interface BuildRequest {
	request_id: number;
	kind: 'build';
	nodes: Float64Array;
	tets: Uint32Array;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	resolution: number;
}

export interface BuildResult {
	request_id: number;
	kind: 'build_result';
	tet_indices: Uint32Array;
	bary: Float32Array;
	resolution: number;
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
	const grid = volume_build_static(mesh, req.resolution);
	const result: BuildResult = {
		request_id: req.request_id,
		kind: 'build_result',
		tet_indices: grid.tet_indices,
		bary: grid.bary,
		resolution: grid.resolution,
		min: grid.min,
		max: grid.max,
	};
	self.postMessage(result, [grid.tet_indices.buffer, grid.bary.buffer]);
});
