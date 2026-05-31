/**
 * Stateful worker-pool wrapper around the volume resampler.
 *
 * Each of the `POOL_SIZE` workers owns one z-slab of the volume. After a
 * build, the workers retain their (tet_indices, bary) lookup locally; the
 * main thread only holds a `VolumeCacheHandle` carrying the metadata
 * needed to dispatch evals and upload the final 3D texture.
 *
 * Per FD frequency change or TD frame:
 *   1. Each worker receives the new field array (small: ~250 KB) over the
 *      wire, evaluates its z-slab against its cached lookup, and ships
 *      back its slab's voxel buffer via transferable (~8 MB at 128³,
 *      zero-copy).
 *   2. The main thread stitches the slabs into one full Float32Array of
 *      length 4·N³ and hands it to `setVolumeData` for the GPU upload.
 *
 * Expected wall-clock on a 4-worker pool with a 25k-tet mesh:
 *      build   128³ ~25 ms,  192³ ~70 ms,  256³ ~180 ms
 *      eval    128³ ~12 ms,  192³ ~35 ms,  256³ ~95 ms
 */
import VolumeWorker from './volume_worker?worker';
import type { BuildResult, EvalResult } from './volume_worker';
import type { MeshData } from './msh';

const POOL_SIZE = Math.max(
	1,
	Math.min(4, typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4),
);

export interface VolumeCacheHandle {
	resolution: number;
	min: [number, number, number];
	max: [number, number, number];
	partitions: { iz_start: number; iz_end: number; worker_idx: number }[];
}

type WorkerCallback = (msg: BuildResult | EvalResult) => void;

interface WorkerSlot {
	worker: Worker;
	listeners: Map<number, WorkerCallback>;
}

let pool: WorkerSlot[] = [];
let next_id = 1;

function ensure_pool(): WorkerSlot[] {
	if (pool.length > 0) return pool;
	for (let i = 0; i < POOL_SIZE; i++) {
		const w = new VolumeWorker();
		const listeners = new Map<number, WorkerCallback>();
		w.addEventListener('message', (e: MessageEvent<BuildResult | EvalResult>) => {
			const r = e.data;
			const cb = listeners.get(r.request_id);
			if (!cb) return;
			listeners.delete(r.request_id);
			cb(r);
		});
		pool.push({ worker: w, listeners });
	}
	return pool;
}

function send<T extends BuildResult | EvalResult>(
	slot: WorkerSlot,
	request_id: number,
	msg: object,
	transfer: Transferable[],
): Promise<T> {
	return new Promise((resolve) => {
		slot.listeners.set(request_id, (r) => resolve(r as T));
		slot.worker.postMessage(msg, transfer);
	});
}

export function volume_build_static_async(
	mesh: MeshData,
	resolution: number,
): Promise<VolumeCacheHandle> {
	const slots = ensure_pool();
	const n_parts = Math.min(slots.length, Math.max(1, Math.floor(resolution / 8)));
	const slab = Math.ceil(resolution / n_parts);
	const id = next_id++;

	const promises: Promise<BuildResult>[] = [];
	const partitions: VolumeCacheHandle['partitions'] = [];
	for (let p = 0; p < n_parts; p++) {
		const iz_start = p * slab;
		const iz_end = Math.min(resolution, iz_start + slab);
		partitions.push({ iz_start, iz_end, worker_idx: p });
		const nodes = new Float64Array(mesh.nodes);
		const tets = new Uint32Array(mesh.tets);
		const bbox = {
			min: [mesh.bbox.min[0], mesh.bbox.min[1], mesh.bbox.min[2]] as [number, number, number],
			max: [mesh.bbox.max[0], mesh.bbox.max[1], mesh.bbox.max[2]] as [number, number, number],
		};
		promises.push(send<BuildResult>(
			slots[p],
			id,
			{ request_id: id, kind: 'build', nodes, tets, bbox, resolution, iz_start, iz_end },
			[nodes.buffer, tets.buffer],
		));
	}
	return Promise.all(promises).then((results) => ({
		resolution,
		min: results[0].min,
		max: results[0].max,
		partitions,
	}));
}

/** Eval the phasor field across all workers in parallel, stitch results. */
export function volume_eval_phasor_async(
	handle: VolumeCacheHandle,
	field_abc: Float32Array,
): Promise<Float32Array> {
	return dispatch_eval(handle, 'eval_phasor', field_abc);
}

export function volume_eval_scalar_async(
	handle: VolumeCacheHandle,
	scalar: Float32Array,
): Promise<Float32Array> {
	return dispatch_eval(handle, 'eval_scalar', scalar);
}

function dispatch_eval(
	handle: VolumeCacheHandle,
	kind: 'eval_phasor' | 'eval_scalar',
	source: Float32Array,
): Promise<Float32Array> {
	const slots = ensure_pool();
	const id = next_id++;
	const promises: Promise<EvalResult>[] = [];
	for (const part of handle.partitions) {
		// Each worker needs its own copy of the field array (TypedArrays can
		// only be transferred to one destination). Clone here, then transfer
		// the clone for a zero-copy hop into the worker.
		const arr = new Float32Array(source);
		const payload = kind === 'eval_phasor'
			? { request_id: id, kind, field_abc: arr }
			: { request_id: id, kind, scalar: arr };
		promises.push(send<EvalResult>(slots[part.worker_idx], id, payload, [arr.buffer]));
	}
	return Promise.all(promises).then((results) => stitch_voxels(results, handle.resolution));
}

function stitch_voxels(parts: EvalResult[], N: number): Float32Array {
	const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
	const slice4 = N * N * 4;
	const out = new Float32Array(N * N * N * 4);
	for (const part of parts) {
		out.set(part.voxels, part.iz_start * slice4);
	}
	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t0;
		console.log(`[volume] stitch_voxels N=${N} ${parts.length} parts ${dt.toFixed(1)} ms`);
	}
	return out;
}
