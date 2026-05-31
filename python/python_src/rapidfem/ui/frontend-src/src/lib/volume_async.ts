/**
 * Async wrapper around the volume-build worker.
 *
 * A single worker instance is spawned lazily on first call and reused for
 * the lifetime of the page. Each `volume_build_static_async` returns a
 * promise resolved when the matching `build_result` arrives. Out-of-order
 * dispatch is impossible (worker is FIFO), but we still key by request_id
 * for clarity and so multiple concurrent in-flight builds (e.g. trajectory
 * + mesh swap in quick succession) resolve to the right caller.
 */
import VolumeWorker from './volume_worker?worker';
import type { BuildResult } from './volume_worker';
import type { MeshData } from './msh';
import type { VolumeGridStatic } from './volume_resample';

let worker_instance: Worker | null = null;
const pending = new Map<number, (g: VolumeGridStatic) => void>();
let next_id = 1;

function ensure_worker(): Worker {
	if (worker_instance) return worker_instance;
	const w = new VolumeWorker();
	w.addEventListener('message', (e: MessageEvent<BuildResult>) => {
		const r = e.data;
		if (r.kind !== 'build_result') return;
		const cb = pending.get(r.request_id);
		if (!cb) return;
		pending.delete(r.request_id);
		cb({
			tet_indices: r.tet_indices,
			bary: r.bary,
			resolution: r.resolution,
			min: r.min,
			max: r.max,
		});
	});
	worker_instance = w;
	return w;
}

export function volume_build_static_async(
	mesh: MeshData,
	resolution: number,
): Promise<VolumeGridStatic> {
	const w = ensure_worker();
	const id = next_id++;
	// Detach from any Svelte $state proxy the caller might be holding by
	// allocating fresh typed arrays. structuredClone cannot serialize a
	// proxy-wrapped TypedArray and throws DataCloneError. The copies are
	// then transferred (zero-copy across the wire); the main thread keeps
	// its originals for the eval pass.
	const nodes = new Float64Array(mesh.nodes);
	const tets = new Uint32Array(mesh.tets);
	const bbox = {
		min: [mesh.bbox.min[0], mesh.bbox.min[1], mesh.bbox.min[2]] as [number, number, number],
		max: [mesh.bbox.max[0], mesh.bbox.max[1], mesh.bbox.max[2]] as [number, number, number],
	};
	return new Promise((resolve) => {
		pending.set(id, resolve);
		w.postMessage(
			{ request_id: id, kind: 'build', nodes, tets, bbox, resolution },
			[nodes.buffer, tets.buffer],
		);
	});
}
