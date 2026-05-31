/// <reference lib="webworker" />
/**
 * Stateful pool-worker for the volume resampler.
 *
 * Each worker holds its assigned z-slab partition + a copy of the mesh
 * topology after a successful build. Subsequent eval calls only need the
 * per-frame field array as input (≈250 KB for a 25k-node mesh), evaluate
 * the slab against the cached lookup, and return the slab's voxel buffer
 * (≈8 MB at 128³ per slab) via transferable for zero-copy upload.
 *
 * Build:       host -> { nodes, tets, bbox, resolution, iz_start, iz_end }
 *              worker stores partition + mesh, returns metadata only
 *              (tet_indices / bary stay in the worker)
 *
 * Eval phasor: host -> { field_abc }
 *              worker -> { iz_start, iz_end, voxels: Float32Array }
 *
 * Eval scalar: host -> { scalar }
 *              worker -> { iz_start, iz_end, voxels: Float32Array }
 */
import {
	volume_build_static_partition,
	volume_eval_phasor_partition,
	volume_eval_scalar_partition,
	type VolumeGridPartition,
} from './volume_resample';
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

interface EvalPhasorRequest {
	request_id: number;
	kind: 'eval_phasor';
	field_abc: Float32Array;
}

interface EvalScalarRequest {
	request_id: number;
	kind: 'eval_scalar';
	scalar: Float32Array;
}

type Request = BuildRequest | EvalPhasorRequest | EvalScalarRequest;

export interface BuildResult {
	request_id: number;
	kind: 'build_result';
	resolution: number;
	iz_start: number;
	iz_end: number;
	min: [number, number, number];
	max: [number, number, number];
}

export interface EvalResult {
	request_id: number;
	kind: 'eval_phasor_result' | 'eval_scalar_result';
	iz_start: number;
	iz_end: number;
	voxels: Float32Array;
}

declare const self: DedicatedWorkerGlobalScope;

let cached: { mesh: MeshData; partition: VolumeGridPartition } | null = null;

self.addEventListener('message', (e: MessageEvent<Request>) => {
	const req = e.data;
	if (req.kind === 'build') {
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
		const partition = volume_build_static_partition(mesh, req.resolution, req.iz_start, req.iz_end);
		cached = { mesh, partition };
		const result: BuildResult = {
			request_id: req.request_id,
			kind: 'build_result',
			resolution: partition.resolution,
			iz_start: partition.iz_start,
			iz_end: partition.iz_end,
			min: partition.min,
			max: partition.max,
		};
		self.postMessage(result);
		return;
	}
	if (req.kind === 'eval_phasor') {
		if (!cached) return;
		const voxels = volume_eval_phasor_partition(cached.partition, cached.mesh, req.field_abc);
		const result: EvalResult = {
			request_id: req.request_id,
			kind: 'eval_phasor_result',
			iz_start: cached.partition.iz_start,
			iz_end: cached.partition.iz_end,
			voxels,
		};
		self.postMessage(result, [voxels.buffer]);
		return;
	}
	if (req.kind === 'eval_scalar') {
		if (!cached) return;
		const voxels = volume_eval_scalar_partition(cached.partition, cached.mesh, req.scalar);
		const result: EvalResult = {
			request_id: req.request_id,
			kind: 'eval_scalar_result',
			iz_start: cached.partition.iz_start,
			iz_end: cached.partition.iz_end,
			voxels,
		};
		self.postMessage(result, [voxels.buffer]);
		return;
	}
});
