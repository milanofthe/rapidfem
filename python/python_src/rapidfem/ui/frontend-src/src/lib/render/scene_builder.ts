/**
 * Shared scene builder — translates a mesh payload into GL state via
 * `canvas3d` primitives. Used by both `MeshViewer.svelte` (the in-app
 * viewer) and `embed/fem-viewer.ts` (the standalone web-component
 * embed) so the two produce bit-identical renderings.
 *
 * Pipeline (matches `MeshViewer.rebuild()` exactly):
 *
 *   1. Named surface tris (PEC walls, ports, ground, etc.):
 *      group by tri_phys, classify by name, color by kind+name
 *   2. Implicit volume hulls (substrate, air, PML shells):
 *      per-volume outer faces from tet_phys, dielectric color,
 *      polygon-offset to push behind coplanar conductor surfaces
 *   3. Optional wireframe: every named-surface tri's edges,
 *      one dim color
 *   4. Optional field point cloud: caller-supplied
 *
 * No Svelte deps — pure TS. Safe to call from a web component or a
 * Svelte rebuild $effect alike.
 */

import {
	addMesh, addLineMesh, setBBox, clearVolume,
	type GLState,
} from './canvas3d';
import { buildVolumeBoundaries, buildTriSoupF64 } from './mesh_scene';
import {
	volume_build_static, volume_eval_phasor, volume_energy_range,
} from '../volume_resample';
import type { MeshData } from '../msh';

// ── Mesh-payload contract ────────────────────────────────────────────

export interface SceneMesh {
	nodes: number[] | Float32Array | Float64Array;
	tris: number[];
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Map<number, string> | Record<string, string>;
	phys_dim?: Map<number, number> | Record<string, number>;
	bbox: { min: [number, number, number]; max: [number, number, number] };
}

// Allow `phys_names` from JSON (object keyed by string) or from the
// in-app code path (Map keyed by number).
function physName(m: SceneMesh, tag: number): string {
	if (m.phys_names instanceof Map) return m.phys_names.get(tag) ?? '';
	return (m.phys_names as Record<string, string>)[String(tag)] ?? '';
}

function physDim(m: SceneMesh, tag: number): number {
	if (!m.phys_dim) return 2;
	if (m.phys_dim instanceof Map) return m.phys_dim.get(tag) ?? 2;
	const v = (m.phys_dim as Record<string, number>)[String(tag)];
	return v ?? 2;
}

// ── Material classification + coloring ───────────────────────────────

export type Kind = 'dielectric' | 'conductor' | 'port' | 'gnd';

export function classify(name: string): Kind | null {
	if (name === 'abc' || name.startsWith('_mat_')) return null;
	if (name === 'substrate' || name === 'oxide' || name === 'air') return 'dielectric';
	if (name.endsWith('_gnd') || name === 'gnd' || name === 'ground') return 'gnd';
	if (
		name === 'p1' || name === 'p2' || /^p\d+$/.test(name) ||
		name.startsWith('port') || name.endsWith('_port')
	) return 'port';
	return 'conductor';
}

function hex(s: string): [number, number, number] {
	return [
		parseInt(s.slice(1, 3), 16) / 255,
		parseInt(s.slice(3, 5), 16) / 255,
		parseInt(s.slice(5, 7), 16) / 255,
	];
}

const FIXED_CONDUCTOR_COLORS: Record<string, string> = {
	met5: '#e8944a', met4: '#f0b86a', met3: '#c4c46b',
	met2: '#9bc28b', met1: '#7b9fb8', li1:  '#5a8caa',
	via5: '#d9513c', via4: '#e5634f', via3: '#bf4233',
	via2: '#9d3526', via1: '#7c281b', mcon: '#aa6b40',
};

export function colorFor(kind: Kind, name: string): [number, number, number] {
	if (kind === 'dielectric') {
		if (name === 'substrate') return hex('#4a9ec2');
		if (name === 'oxide') return hex('#7b5e8a');
		return hex('#5a6470');
	}
	if (kind === 'gnd') return hex('#5aad78');
	if (kind === 'port') return hex('#d9513c');           // accent
	return hex(FIXED_CONDUCTOR_COLORS[name] ?? '#e8944a'); // accent-secondary default
}

// ── Push one tri-group with high-precision normals ───────────────────

function pushGroup(
	state: GLState,
	mesh: SceneMesh,
	idx: number[],
	color: [number, number, number],
	tag: number,
	depthOffset: [number, number] | undefined,
	fieldNorm: Float32Array | null,
): void {
	const ntri = idx.length / 3;
	const { positions, normals } = buildTriSoupF64(mesh.nodes, idx);
	let scalars: Float32Array | undefined;
	if (fieldNorm) {
		scalars = new Float32Array(ntri * 3);
		for (let t = 0; t < ntri; t++) {
			for (let v = 0; v < 3; v++) {
				scalars[t * 3 + v] = fieldNorm[idx[t * 3 + v]];
			}
		}
	}
	addMesh(state, positions, normals, color, tag, depthOffset, scalars);
}

// ── Wireframe edges of every named surface tri ───────────────────────

function buildWireEdges(mesh: SceneMesh): Float32Array {
	const seen = new Set<bigint>();
	const out: number[] = [];
	const push = (a: number, b: number) => {
		const lo = a < b ? a : b, hi = a < b ? b : a;
		const k = (BigInt(lo) << 32n) | BigInt(hi);
		if (seen.has(k)) return;
		seen.add(k);
		out.push(
			mesh.nodes[a * 3], mesh.nodes[a * 3 + 1], mesh.nodes[a * 3 + 2],
			mesh.nodes[b * 3], mesh.nodes[b * 3 + 1], mesh.nodes[b * 3 + 2],
		);
	};
	const n_tris = mesh.tris.length / 3;
	for (let t = 0; t < n_tris; t++) {
		const a = mesh.tris[t * 3], b = mesh.tris[t * 3 + 1], c = mesh.tris[t * 3 + 2];
		push(a, b); push(b, c); push(c, a);
	}
	return Float32Array.from(out);
}

// ── Public API ───────────────────────────────────────────────────────

export interface BuildSceneConfig {
	showFaces?: boolean;           // named surfaces + volume hulls (default true)
	showWire?: boolean;            // edge wireframe (default false)
	showField?: boolean;           // point cloud (caller sets it separately)
	/** Optional per-node normalised field magnitude for vertex-tinted faces.
	 *  Drives the inferno colormap on the mesh surfaces in field mode. */
	fieldNorm?: Float32Array | null;
}

/**
 * Wipe the GL state's previous scene contents and rebuild from `mesh`.
 *
 * NOTE: the caller is responsible for `clearMeshes()` BEFORE this — we
 * don't do it here so callers can compose multiple scenes (geometry +
 * wireframe overlay) if they want. We do call `setBBox` so the camera
 * fitter has fresh bounds.
 *
 * The field point cloud is not set here; the in-app viewer hands it
 * off to a worker and the embed builds a synchronous tet-centroid
 * sample — both call setPointCloud themselves.
 */
export const WIRE_TAG = -1;

export function buildScene(
	state: GLState,
	mesh: SceneMesh,
	config: BuildSceneConfig = {},
): { faceTags: number[]; wireTag: number | null } {
	const showFaces = config.showFaces ?? true;
	const showWire = config.showWire ?? false;
	const fieldNorm = config.fieldNorm ?? null;
	const faceTags: number[] = [];
	let wireTag: number | null = null;

	setBBox(state, mesh.bbox.min, mesh.bbox.max);

	if (showFaces) {
		// 1) Named surface tris (conductors / ports / gnd / ABC are skipped).
		const bySurf = new Map<number, number[]>();
		const n_tris = mesh.tri_phys.length;
		for (let i = 0; i < n_tris; i++) {
			const tag = mesh.tri_phys[i];
			if (!tag || physDim(mesh, tag) !== 2) continue;
			let arr = bySurf.get(tag);
			if (!arr) { arr = []; bySurf.set(tag, arr); }
			arr.push(mesh.tris[i * 3], mesh.tris[i * 3 + 1], mesh.tris[i * 3 + 2]);
		}
		for (const [tag, idx] of bySurf.entries()) {
			const name = physName(mesh, tag);
			const kind = classify(name);
			if (!kind) continue;
			pushGroup(state, mesh, idx, colorFor(kind, name), tag, undefined, fieldNorm);
			faceTags.push(tag);
		}
		// 2) Implicit volume hulls — substrate / air / PML shells. Push
		//    behind via polygon offset so coplanar conductors win the
		//    depth test cleanly. (In field-mode the colormap renders all
		//    surfaces by |E| anyway so the offset doesn't matter.)
		const volBoundaries = buildVolumeBoundaries(mesh);
		for (const [vtag, idx] of volBoundaries.entries()) {
			const name = physName(mesh, vtag);
			if (!name || name.startsWith('_mat_')) continue;
			const offset: [number, number] | undefined = fieldNorm ? undefined : [2, 2];
			pushGroup(state, mesh, idx, colorFor('dielectric', name), vtag, offset, fieldNorm);
			faceTags.push(vtag);
		}
	}

	if (showWire) {
		const edges = buildWireEdges(mesh);
		// Dim grey — matches the line color MeshViewer uses for its mesh
		// wireframe overlay so embed + in-app look identical.
		addLineMesh(state, edges, hex('#3a3a44'), WIRE_TAG);
		wireTag = WIRE_TAG;
	}

	return { faceTags, wireTag };
}

/** Convenience: wipe the field volume. Callers use this when toggling
 *  out of field mode. */
export function clearFieldCloud(state: GLState): void {
	clearVolume(state);
}

// ── Volumetric field resampling ───────────────────────────────────────
//
// Replacement for the old `sampleFieldCloud` point-cloud sampler. Builds the
// per-voxel `(A, B, C, occ)` buffer the new volume raycaster consumes. The
// pipeline is shared with the in-app viewer (`volume_resample.ts`); this
// helper just adapts the embed's `SceneMesh` (`tets` as a plain `number[]`)
// into the strict `MeshData` shape the resampler expects.

function sceneToMeshData(m: SceneMesh): MeshData {
	const nodes = m.nodes instanceof Float64Array
		? m.nodes
		: new Float64Array(m.nodes as ArrayLike<number>);
	const tets = m.tets instanceof Uint32Array ? m.tets : new Uint32Array(m.tets);
	return {
		nodes,
		tris: new Uint32Array(0),
		tri_phys: new Int32Array(0),
		tets,
		tet_phys: new Int32Array(0),
		phys_names: new Map(),
		phys_dim: new Map(),
		bbox: m.bbox,
	};
}

/** Resample a per-node `(A, B, C)` field onto a regular 3D grid and return
 *  the packed RGBA32F voxel buffer plus the world-space BBox and a robust
 *  colour range. The embed treats this output as its cached field artefact
 *  (one entry per `(freq, port, resolution)` key). */
export function buildVolumeCloud(
	mesh: SceneMesh,
	fieldAbc: number[] | Float32Array,
	resolution = 128,
): {
	voxels: Float32Array;
	resolution: number;
	min: [number, number, number];
	max: [number, number, number];
	maxE2: number;
	minE2: number;
} {
	const md = sceneToMeshData(mesh);
	const f = fieldAbc instanceof Float32Array
		? fieldAbc
		: new Float32Array(fieldAbc as ArrayLike<number>);
	const grid = volume_build_static(md, resolution);
	const voxels = volume_eval_phasor(grid, md, f);
	const range = volume_energy_range(voxels);
	return {
		voxels,
		resolution: grid.resolution,
		min: grid.min,
		max: grid.max,
		maxE2: range.field_range.max * range.field_range.max,
		minE2: range.field_range.min * range.field_range.min,
	};
}
