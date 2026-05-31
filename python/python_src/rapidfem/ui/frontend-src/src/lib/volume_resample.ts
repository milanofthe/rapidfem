/**
 * Resample a tet-mesh field onto a regular 3D grid for volume rendering.
 *
 * Mirrors the FD + TD split of `viz_sample` / `viz_sample_static`. Both modes
 * produce the same GPU upload shape: an RGBA32F 3D texture with
 * `(A, B, C, occ)` per voxel where `occ ∈ {0, 1}` masks the mesh interior
 * and the shader evaluates the phasor magnitude
 *      |F(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2C·cos·sin
 * exactly as the point-cloud shader does today.
 *
 * Pipeline:
 *
 *   volume_build_static(mesh, N)  →  VolumeGridStatic
 *           (per-voxel tet index + 4 barycentric weights, field-independent,
 *            ~32 MB at N=128. Built once per mesh.)
 *
 *   volume_eval_phasor(grid, mesh, field_abc)  →  Float32Array  (FD)
 *           Per-voxel linear interpolation of A, B, C from the per-node
 *           phasor triple. One upload per frequency change.
 *
 *   volume_eval_scalar(grid, mesh, scalar)  →  Float32Array     (TD)
 *           Per-voxel linear interpolation of a per-node scalar, encoded as
 *           (s², s², 0, occ) so the phasor shader yields |F|² = s²
 *           independently of phase. One upload per animation frame.
 *
 * Resolution: 128³ is the default — ~32 MB GPU upload, ~2M inside-tests
 * during build, sub-second on the main thread for typical FEM meshes. 256³
 * is supported but costs 8× build + 8× memory (~256 MB upload), reserve for
 * archival rendering. Bumping the constant is one parameter, no API change.
 */
import type { MeshData } from './msh';

// 96³ keeps the build under 200 ms on a 25k-tet mesh and the GPU upload at
// ~13.5 MB. 128³ looked marginally crisper but cost 2.4× the build time and
// 2.4× the memory; not worth it for the live viewer. Embeds default to 96
// already; this aligns the in-app viewer with that.
export const DEFAULT_RESOLUTION = 96;
const TET_OUTSIDE = 0xffffffff;

/** Geometric cache: which tet contains each voxel and at what bary weights.
 *  Field-independent so it survives any field swap on the same mesh. */
export interface VolumeGridStatic {
	/** Per-voxel containing tet index. `TET_OUTSIDE` for voxels outside mesh. */
	tet_indices: Uint32Array;
	/** Per-voxel barycentric (λ0, λ1, λ2, λ3) wrt the containing tet's nodes. */
	bary: Float32Array;
	/** Cubic grid resolution: data length is N³. */
	resolution: number;
	/** World-space BBox the grid covers, padded half a voxel outside the mesh. */
	min: [number, number, number];
	max: [number, number, number];
}

/**
 * Build the per-voxel tet/bary cache. Scatter-based: for each tet, walk its
 * BBox in voxel coords and check barycentric inside-ness at the voxel
 * center. Shared faces produce identical bary on both sides, so write-order
 * is irrelevant.
 */
export function volume_build_static(
	mesh: MeshData,
	resolution: number = DEFAULT_RESOLUTION,
): VolumeGridStatic {
	const t_start = typeof performance !== 'undefined' ? performance.now() : 0;
	const N = resolution;
	const n_voxels = N * N * N;
	const tet_indices = new Uint32Array(n_voxels);
	tet_indices.fill(TET_OUTSIDE);
	const bary = new Float32Array(n_voxels * 4);

	const n_tets = mesh.tets.length / 4;
	const empty_min: [number, number, number] = [...mesh.bbox.min];
	const empty_max: [number, number, number] = [...mesh.bbox.max];
	if (n_tets === 0) {
		return { tet_indices, bary, resolution: N, min: empty_min, max: empty_max };
	}

	// Pad the grid BBox by half a voxel so boundary tets fit fully inside.
	const pad_frac = 0.5 / N;
	const min: [number, number, number] = [0, 0, 0];
	const max: [number, number, number] = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const span = mesh.bbox.max[k] - mesh.bbox.min[k];
		const pad = span * pad_frac;
		min[k] = mesh.bbox.min[k] - pad;
		max[k] = mesh.bbox.max[k] + pad;
	}
	const inv_dx = N / (max[0] - min[0]);
	const inv_dy = N / (max[1] - min[1]);
	const inv_dz = N / (max[2] - min[2]);
	const dx = 1 / inv_dx;
	const dy = 1 / inv_dy;
	const dz = 1 / inv_dz;

	const { nodes, tets } = mesh;
	const p = new Float64Array(12);

	for (let t = 0; t < n_tets; t++) {
		const i0 = tets[t * 4 + 0];
		const i1 = tets[t * 4 + 1];
		const i2 = tets[t * 4 + 2];
		const i3 = tets[t * 4 + 3];

		for (let k = 0; k < 3; k++) {
			p[0 + k] = nodes[i0 * 3 + k];
			p[3 + k] = nodes[i1 * 3 + k];
			p[6 + k] = nodes[i2 * 3 + k];
			p[9 + k] = nodes[i3 * 3 + k];
		}

		const xmin = Math.min(p[0], p[3], p[6], p[9]);
		const xmax = Math.max(p[0], p[3], p[6], p[9]);
		const ymin = Math.min(p[1], p[4], p[7], p[10]);
		const ymax = Math.max(p[1], p[4], p[7], p[10]);
		const zmin = Math.min(p[2], p[5], p[8], p[11]);
		const zmax = Math.max(p[2], p[5], p[8], p[11]);
		const ix_lo = Math.max(0, Math.floor((xmin - min[0]) * inv_dx));
		const ix_hi = Math.min(N - 1, Math.floor((xmax - min[0]) * inv_dx));
		const iy_lo = Math.max(0, Math.floor((ymin - min[1]) * inv_dy));
		const iy_hi = Math.min(N - 1, Math.floor((ymax - min[1]) * inv_dy));
		const iz_lo = Math.max(0, Math.floor((zmin - min[2]) * inv_dz));
		const iz_hi = Math.min(N - 1, Math.floor((zmax - min[2]) * inv_dz));

		// Inverse barycentric matrix from
		//   [p1−p0, p2−p0, p3−p0] · (λ1, λ2, λ3)ᵀ = (x − p0)
		const ax = p[3] - p[0], ay = p[4] - p[1], az = p[5] - p[2];
		const bx = p[6] - p[0], by = p[7] - p[1], bz = p[8] - p[2];
		const cx = p[9] - p[0], cy = p[10] - p[1], cz = p[11] - p[2];
		const det =
			ax * (by * cz - bz * cy) -
			ay * (bx * cz - bz * cx) +
			az * (bx * cy - by * cx);
		if (Math.abs(det) < 1e-30) continue;
		const inv_det = 1 / det;
		const m00 = (by * cz - bz * cy) * inv_det;
		const m01 = (az * cy - ay * cz) * inv_det;
		const m02 = (ay * bz - az * by) * inv_det;
		const m10 = (bz * cx - bx * cz) * inv_det;
		const m11 = (ax * cz - az * cx) * inv_det;
		const m12 = (az * bx - ax * bz) * inv_det;
		const m20 = (bx * cy - by * cx) * inv_det;
		const m21 = (ay * cx - ax * cy) * inv_det;
		const m22 = (ax * by - ay * bx) * inv_det;

		// Inner-loop accelerator: λ_i is linear in (wx, wy, wz), so over a
		// constant-y/z slab the only term that changes with ix is m·k·dx.
		// Pre-compute the per-step deltas, then march l1/l2/l3 by addition
		// only — three add+compare per voxel instead of nine mult+add.
		const dl1 = m00 * dx;
		const dl2 = m01 * dx;
		const dl3 = m02 * dx;
		const wx_start = min[0] + (ix_lo + 0.5) * dx - p[0];

		for (let iz = iz_lo; iz <= iz_hi; iz++) {
			const wz = min[2] + (iz + 0.5) * dz - p[2];
			for (let iy = iy_lo; iy <= iy_hi; iy++) {
				const wy = min[1] + (iy + 0.5) * dy - p[1];
				const base_yz = (iz * N + iy) * N;
				let l1 = m00 * wx_start + m10 * wy + m20 * wz;
				let l2 = m01 * wx_start + m11 * wy + m21 * wz;
				let l3 = m02 * wx_start + m12 * wy + m22 * wz;
				for (let ix = ix_lo; ix <= ix_hi; ix++) {
					if (l1 >= 0 && l1 <= 1 && l2 >= 0 && l2 <= 1 && l3 >= 0 && l3 <= 1) {
						const l0 = 1 - l1 - l2 - l3;
						if (l0 >= 0) {
							const voxel = base_yz + ix;
							tet_indices[voxel] = t;
							const off = voxel * 4;
							bary[off + 0] = l0;
							bary[off + 1] = l1;
							bary[off + 2] = l2;
							bary[off + 3] = l3;
						}
					}
					l1 += dl1;
					l2 += dl2;
					l3 += dl3;
				}
			}
		}
	}

	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t_start;
		console.log(`[volume] build_static N=${N} n_tets=${n_tets} ${dt.toFixed(1)} ms`);
	}
	return { tet_indices, bary, resolution: N, min, max };
}

/**
 * Fill an RGBA32F voxel buffer with `(A, B, C, occ)` from a per-node phasor
 * field. One call per frequency change in the FD pipeline.
 */
export function volume_eval_phasor(
	grid: VolumeGridStatic,
	mesh: MeshData,
	field_abc: Float32Array,
): Float32Array {
	const t_start = typeof performance !== 'undefined' ? performance.now() : 0;
	const N = grid.resolution;
	const n_voxels = N * N * N;
	const out = new Float32Array(n_voxels * 4);
	if (!field_abc || field_abc.length === 0) return out;

	const tets = mesh.tets;
	const tet_indices = grid.tet_indices;
	const bary = grid.bary;

	for (let v = 0; v < n_voxels; v++) {
		const t = tet_indices[v];
		if (t === TET_OUTSIDE) continue;
		const base = t * 4;
		const i0 = tets[base + 0] * 3;
		const i1 = tets[base + 1] * 3;
		const i2 = tets[base + 2] * 3;
		const i3 = tets[base + 3] * 3;
		const bo = v * 4;
		const l0 = bary[bo + 0];
		const l1 = bary[bo + 1];
		const l2 = bary[bo + 2];
		const l3 = bary[bo + 3];
		const off = v * 4;
		out[off + 0] =
			l0 * field_abc[i0]     + l1 * field_abc[i1]     +
			l2 * field_abc[i2]     + l3 * field_abc[i3];
		out[off + 1] =
			l0 * field_abc[i0 + 1] + l1 * field_abc[i1 + 1] +
			l2 * field_abc[i2 + 1] + l3 * field_abc[i3 + 1];
		out[off + 2] =
			l0 * field_abc[i0 + 2] + l1 * field_abc[i1 + 2] +
			l2 * field_abc[i2 + 2] + l3 * field_abc[i3 + 2];
		out[off + 3] = 1.0;
	}
	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t_start;
		console.log(`[volume] eval_phasor N=${N} ${dt.toFixed(1)} ms`);
	}
	return out;
}

/**
 * Fill an RGBA32F voxel buffer with `(s², s², 0, occ)` from a per-node
 * scalar magnitude. The (s², s², 0) encoding makes the phasor shader yield
 * a phase-independent constant `s²` (matches the `td_abc_from_mag` trick the
 * point cloud uses today). One call per animation frame in the TD pipeline.
 */
export function volume_eval_scalar(
	grid: VolumeGridStatic,
	mesh: MeshData,
	scalar: Float32Array,
): Float32Array {
	const t_start = typeof performance !== 'undefined' ? performance.now() : 0;
	const N = grid.resolution;
	const n_voxels = N * N * N;
	const out = new Float32Array(n_voxels * 4);
	if (!scalar || scalar.length === 0) return out;

	const tets = mesh.tets;
	const tet_indices = grid.tet_indices;
	const bary = grid.bary;

	for (let v = 0; v < n_voxels; v++) {
		const t = tet_indices[v];
		if (t === TET_OUTSIDE) continue;
		const base = t * 4;
		const n0 = tets[base + 0];
		const n1 = tets[base + 1];
		const n2 = tets[base + 2];
		const n3 = tets[base + 3];
		const bo = v * 4;
		const s =
			bary[bo + 0] * scalar[n0] +
			bary[bo + 1] * scalar[n1] +
			bary[bo + 2] * scalar[n2] +
			bary[bo + 3] * scalar[n3];
		const s2 = s * s;
		const off = v * 4;
		out[off + 0] = s2;
		out[off + 1] = s2;
		out[off + 2] = 0;
		out[off + 3] = 1.0;
	}
	if (typeof performance !== 'undefined') {
		const dt = performance.now() - t_start;
		console.log(`[volume] eval_scalar N=${N} ${dt.toFixed(1)} ms`);
	}
	return out;
}

/**
 * Colormap auto-range from the voxel buffer. Same 99th-percentile clip the
 * point cloud uses: outlier port-driven voxels saturate at the top, the bulk
 * spans the full colour range. Operates on |F| = √((A+B)/2), so the result
 * matches the existing shader's log10 scale.
 */
export function volume_energy_range(values: Float32Array): {
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
} {
	const RANGE_PERCENTILE = 0.99;
	const energies: number[] = [];
	const n = values.length / 4;
	for (let i = 0; i < n; i++) {
		const off = i * 4;
		if (values[off + 3] <= 0) continue;
		const e2 = 0.5 * (values[off] + values[off + 1]);
		if (e2 > 0) energies.push(e2);
	}
	if (energies.length === 0) {
		return {
			log_floor: 0,
			log_range: 1,
			field_range: { min: 1, max: 1, decades: 0 },
		};
	}
	energies.sort((a, b) => a - b);
	const lo_idx = Math.min(energies.length - 1, Math.floor(energies.length * (1 - RANGE_PERCENTILE)));
	const hi_idx = Math.min(energies.length - 1, Math.floor(energies.length * RANGE_PERCENTILE));
	const e_max = Math.sqrt(energies[hi_idx]);
	const e_min = Math.max(Math.sqrt(energies[lo_idx]), e_max * 1e-3);
	const log_max = Math.log10(e_max);
	const log_min = Math.log10(e_min);
	return {
		log_floor: log_min,
		log_range: Math.max(log_max - log_min, 0.5),
		field_range: { min: e_min, max: e_max, decades: log_max - log_min },
	};
}
