/**
 * Worker-pool wrapper around `volume_build_static_partition`.
 *
 * A pool of up to `POOL_SIZE` workers is spawned lazily on the first call.
 * Each build splits the volume by Z-slice (one slab per worker), dispatches
 * all slabs in parallel, and merges the partial buffers into one full
 * `VolumeGridStatic` on the main thread.
 *
 * Why partitioned: each worker still iterates every tet, but only fills
 * voxels inside its z-slab. Per-tet setup is duplicated (~3% of the work)
 * but the inner voxel scan (the dominant cost) parallelises cleanly.
 * Expected wall-clock: ~25-30 ms at 128³ and ~50-70 ms at 192³ on a
 * four-core box, versus ~100 ms / ~320 ms with a single worker.
 *
 * Worker re-use: pool is created once and reused for the lifetime of the
 * page. There is no per-build worker spawn cost after the first call.
 */
import VolumeWorker from './volume_worker?worker';
import type { BuildResult } from './volume_worker';
import type { MeshData } from './msh';
import type { VolumeGridStatic } from './volume_resample';

const TET_OUTSIDE = 0xffffffff;
const POOL_SIZE = Math.max(
	1,
	Math.min(4, typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4),
);

interface BuildJob {
	resolve: (g: VolumeGridStatic) => void;
	resolution: number;
	expected_partitions: number;
	received: BuildResult[];
}

const workers: Worker[] = [];
const jobs = new Map<number, BuildJob>();
let next_id = 1;

function ensure_pool(): Worker[] {
	if (workers.length > 0) return workers;
	for (let i = 0; i < POOL_SIZE; i++) {
		const w = new VolumeWorker();
		w.addEventListener('message', (e: MessageEvent<BuildResult>) => {
			const r = e.data;
			if (r.kind !== 'build_result') return;
			const job = jobs.get(r.request_id);
			if (!job) return;
			job.received.push(r);
			if (job.received.length === job.expected_partitions) {
				jobs.delete(r.request_id);
				job.resolve(stitch_partitions(job.received, job.resolution));
			}
		});
		workers.push(w);
	}
	return workers;
}

function stitch_partitions(parts: BuildResult[], N: number): VolumeGridStatic {
	const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
	const slice = N * N;
	const total_voxels = slice * N;
	const tet_indices = new Uint32Array(total_voxels);
	tet_indices.fill(TET_OUTSIDE);
	const bary = new Float32Array(total_voxels * 4);
	let min: [number, number, number] = [0, 0, 0];
	let max: [number, number, number] = [0, 0, 0];
	for (const part of parts) {
		const off_tet = part.iz_start * slice;
		const off_bary = off_tet * 4;
		tet_indices.set(part.tet_indices, off_tet);
		bary.set(part.bary, off_bary);
		min = part.min;
		max = part.max;
	}
	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t0;
		console.log(`[volume] stitch N=${N} ${parts.length} parts ${dt.toFixed(1)} ms`);
	}
	return { tet_indices, bary, resolution: N, min, max };
}

export function volume_build_static_async(
	mesh: MeshData,
	resolution: number,
): Promise<VolumeGridStatic> {
	const pool = ensure_pool();
	const id = next_id++;
	// Z-slab partition: split [0, N) across the pool, last slab takes the
	// remainder. n_parts <= POOL_SIZE; if N is tiny we shrink the pool.
	const n_parts = Math.min(pool.length, Math.max(1, Math.floor(resolution / 8)));
	const slab = Math.ceil(resolution / n_parts);

	return new Promise((resolve) => {
		jobs.set(id, {
			resolve,
			resolution,
			expected_partitions: n_parts,
			received: [],
		});
		for (let p = 0; p < n_parts; p++) {
			const iz_start = p * slab;
			const iz_end = Math.min(resolution, iz_start + slab);
			// Each worker gets its own structured-clone-safe copy of the
			// mesh buffers (caller's TypedArrays may be Svelte $state
			// proxies). Copies are then transferred (zero-copy on the
			// wire); the main thread keeps the originals for eval.
			const nodes = new Float64Array(mesh.nodes);
			const tets = new Uint32Array(mesh.tets);
			const bbox = {
				min: [mesh.bbox.min[0], mesh.bbox.min[1], mesh.bbox.min[2]] as [number, number, number],
				max: [mesh.bbox.max[0], mesh.bbox.max[1], mesh.bbox.max[2]] as [number, number, number],
			};
			pool[p].postMessage(
				{
					request_id: id,
					kind: 'build',
					nodes,
					tets,
					bbox,
					resolution,
					iz_start,
					iz_end,
				},
				[nodes.buffer, tets.buffer],
			);
		}
	});
}
