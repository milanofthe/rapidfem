/**
 * Resample a per-node tet-mesh phasor field onto a regular 3D grid for
 * volume rendering.
 *
 * Input: per-node interleaved `[A, B, C]` triple (same layout as `viz_sample`
 * consumes). Output: four Float32Array views packed as RGBA32F voxels with
 * R=A, G=B, B=C, A=occupancy (1.0 inside mesh, 0.0 outside). The shader
 * evaluates |E(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2C·cos·sin and modulates by
 * the alpha channel so air space contributes nothing.
 *
 * Algorithm: scatter, not gather. For each tet, expand its node-space BBox
 * into voxel index ranges, then for each voxel in that range test barycentric
 * inside-ness and write the linear interpolant. Overlap at shared faces
 * agrees because the P1 field is continuous, so write-order is irrelevant.
 *
 * Cost: O(n_tets · avg_voxels_per_tet). For a 25k-tet mesh at 128³
 * resolution, that's ~2M inside-tests, well under a second on the main
 * thread. 256³ pushes a 32 MB upload + ~16M tests which is borderline —
 * stick with 128³ by default.
 */
import type { MeshData } from './msh';

export interface VolumeGrid {
	/** Packed RGBA32F voxels: [A0,B0,C0,occ0, A1,B1,C1,occ1, ...] */
	data: Float32Array;
	/** Grid resolution N (cubic: N×N×N voxels). */
	resolution: number;
	/** World-space BBox the grid spans, padded to include the full mesh. */
	min: [number, number, number];
	max: [number, number, number];
}

const DEFAULT_RESOLUTION = 128;

/**
 * Resample the per-node phasor field onto a cubic regular grid covering the
 * mesh BBox.
 */
export function resample_field_to_grid(
	mesh: MeshData,
	field_abc: Float32Array,
	resolution: number = DEFAULT_RESOLUTION,
): VolumeGrid {
	const N = resolution;
	const n_voxels = N * N * N;
	const data = new Float32Array(n_voxels * 4);

	const n_tets = mesh.tets.length / 4;
	if (n_tets === 0 || field_abc.length === 0) {
		return {
			data,
			resolution: N,
			min: [...mesh.bbox.min],
			max: [...mesh.bbox.max],
		};
	}

	// Pad the grid BBox by one voxel on each side so boundary tets fully fit.
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

	// Scratch storage for the current tet.
	const p = new Float64Array(12);   // 4 nodes × 3 coords
	const f = new Float32Array(12);   // 4 nodes × (A, B, C)

	for (let t = 0; t < n_tets; t++) {
		const i0 = tets[t * 4 + 0];
		const i1 = tets[t * 4 + 1];
		const i2 = tets[t * 4 + 2];
		const i3 = tets[t * 4 + 3];

		// Tet vertex positions.
		for (let k = 0; k < 3; k++) {
			p[0 + k] = nodes[i0 * 3 + k];
			p[3 + k] = nodes[i1 * 3 + k];
			p[6 + k] = nodes[i2 * 3 + k];
			p[9 + k] = nodes[i3 * 3 + k];
		}
		// Per-vertex phasor.
		for (let k = 0; k < 3; k++) {
			f[0 + k] = field_abc[i0 * 3 + k];
			f[3 + k] = field_abc[i1 * 3 + k];
			f[6 + k] = field_abc[i2 * 3 + k];
			f[9 + k] = field_abc[i3 * 3 + k];
		}

		// Tet BBox → voxel index range.
		let xmin = Math.min(p[0], p[3], p[6], p[9]);
		let xmax = Math.max(p[0], p[3], p[6], p[9]);
		let ymin = Math.min(p[1], p[4], p[7], p[10]);
		let ymax = Math.max(p[1], p[4], p[7], p[10]);
		let zmin = Math.min(p[2], p[5], p[8], p[11]);
		let zmax = Math.max(p[2], p[5], p[8], p[11]);
		const ix_lo = Math.max(0, Math.floor((xmin - min[0]) * inv_dx));
		const ix_hi = Math.min(N - 1, Math.floor((xmax - min[0]) * inv_dx));
		const iy_lo = Math.max(0, Math.floor((ymin - min[1]) * inv_dy));
		const iy_hi = Math.min(N - 1, Math.floor((ymax - min[1]) * inv_dy));
		const iz_lo = Math.max(0, Math.floor((zmin - min[2]) * inv_dz));
		const iz_hi = Math.min(N - 1, Math.floor((zmax - min[2]) * inv_dz));

		// Precompute the inverse barycentric matrix for the tet. Solving
		//   [p1−p0, p2−p0, p3−p0] · (λ1, λ2, λ3)ᵀ = (x − p0)
		// gives λ1..3; λ0 = 1 − λ1 − λ2 − λ3. Inside iff all four ≥ 0.
		const ax = p[3] - p[0], ay = p[4] - p[1], az = p[5] - p[2];
		const bx = p[6] - p[0], by = p[7] - p[1], bz = p[8] - p[2];
		const cx = p[9] - p[0], cy = p[10] - p[1], cz = p[11] - p[2];
		const det =
			ax * (by * cz - bz * cy) -
			ay * (bx * cz - bz * cx) +
			az * (bx * cy - by * cx);
		if (Math.abs(det) < 1e-30) continue;
		const inv_det = 1 / det;
		// Cofactors of M = [a, b, c] so M⁻¹ = adj(M)ᵀ / det.
		const m00 = (by * cz - bz * cy) * inv_det;
		const m01 = (az * cy - ay * cz) * inv_det;
		const m02 = (ay * bz - az * by) * inv_det;
		const m10 = (bz * cx - bx * cz) * inv_det;
		const m11 = (ax * cz - az * cx) * inv_det;
		const m12 = (az * bx - ax * bz) * inv_det;
		const m20 = (bx * cy - by * cx) * inv_det;
		const m21 = (ay * cx - ax * cy) * inv_det;
		const m22 = (ax * by - ay * bx) * inv_det;

		// Voxel-center start position for the inner z-loop (ix_lo, iy_lo, iz_lo).
		for (let iz = iz_lo; iz <= iz_hi; iz++) {
			const wz = min[2] + (iz + 0.5) * dz - p[2];
			for (let iy = iy_lo; iy <= iy_hi; iy++) {
				const wy = min[1] + (iy + 0.5) * dy - p[1];
				const base_yz = (iz * N + iy) * N;
				for (let ix = ix_lo; ix <= ix_hi; ix++) {
					const wx = min[0] + (ix + 0.5) * dx - p[0];
					const l1 = m00 * wx + m10 * wy + m20 * wz;
					if (l1 < 0 || l1 > 1) continue;
					const l2 = m01 * wx + m11 * wy + m21 * wz;
					if (l2 < 0 || l2 > 1) continue;
					const l3 = m02 * wx + m12 * wy + m22 * wz;
					if (l3 < 0 || l3 > 1) continue;
					const l0 = 1 - l1 - l2 - l3;
					if (l0 < 0) continue;
					const off = (base_yz + ix) * 4;
					data[off + 0] = l0 * f[0] + l1 * f[3] + l2 * f[6] + l3 * f[9];
					data[off + 1] = l0 * f[1] + l1 * f[4] + l2 * f[7] + l3 * f[10];
					data[off + 2] = l0 * f[2] + l1 * f[5] + l2 * f[8] + l3 * f[11];
					data[off + 3] = 1.0;
				}
			}
		}
	}

	return { data, resolution: N, min, max };
}

/**
 * Auto-range for the voxel field, analogous to `field_energy_range` in
 * viz.ts. Returns clipped log10 bounds on |E|_node ≈ √((A+B)/2) sampled at
 * occupied voxels, with the same 99th-percentile clip rationale: port-driven
 * outliers saturate at the top, bulk uses the full color span.
 */
export function volume_energy_range(grid: VolumeGrid): {
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
} {
	const RANGE_PERCENTILE = 0.99;
	const energies: number[] = [];
	const data = grid.data;
	const n = grid.resolution ** 3;
	for (let i = 0; i < n; i++) {
		const off = i * 4;
		if (data[off + 3] <= 0) continue;
		const e2 = 0.5 * (data[off] + data[off + 1]);
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
