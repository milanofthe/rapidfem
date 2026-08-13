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
	addMesh, addLineMesh, setBBox, setPointCloud,
	type GLState,
} from './canvas3d';
import { buildVolumeBoundaries, buildTriSoupF64 } from './mesh_scene';

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

// Physical groups arrive under two naming schemes and both have to be
// understood here. `Geometry.mesh()` names a group after the material class
// plus a counter — "dielectric_1", "air_2", "abc_1" — while hand-built and
// older models use bare names like "substrate" / "oxide" / "gnd". Matching
// only the bare names (as this did) drops every generated group through to
// the `conductor` fallback: dielectrics render in conductor colour and the
// absorbing boundary, which should never be drawn at all, gets painted over
// the model.
export function classify(name: string): Kind | null {
	const base = name.replace(/_\d+$/, '');
	if (base === 'abc' || name.startsWith('_mat_')) return null;
	if (
		base === 'substrate' || base === 'oxide' || base === 'air' ||
		base === 'dielectric' || base === 'anisotropic' || base === 'pml' ||
		base === 'debye' || base === 'drude'
	) return 'dielectric';
	if (base === 'gnd' || base === 'ground' || name.endsWith('_gnd')) return 'gnd';
	if (
		/^p\d+$/.test(name) || base === 'port' ||
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
		// Same two naming schemes as classify(); strip the counter suffix.
		const base = name.replace(/_\d+$/, '');
		if (base === 'substrate') return hex('#4a9ec2');
		if (base === 'oxide') return hex('#7b5e8a');
		if (base === 'air') return hex('#3f4855');       // dimmer than solids
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
	cull = false,
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
	addMesh(state, positions, normals, color, tag, depthOffset, scalars, cull);
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
		// Named surfaces get a one-unit nudge TOWARDS the camera, so that a
		// conductor sitting exactly ON a volume boundary (ground plane on a
		// substrate face, port plate on a slab face) wins the depth test.
		// The nudge lives here, on the coplanar special case, rather than on
		// the hulls: offsetting every hull backwards also pushed back the
		// dielectrics that genuinely sit IN FRONT of a conductor, so the
		// conformal passivation lost against the surface-impedance shell it
		// encloses and flickered through it. With CULL_FACE off (sheets need
		// both sides) the cap's underside is exactly coplanar with the shell,
		// which is what made it fight in the first place.
		// Sheets: no offset, no culling. These groups come straight from the
		// mesh's triangle list and have no reliable winding, so both sides
		// have to draw; they sit at their true depth.
		for (const [tag, idx] of bySurf.entries()) {
			const name = physName(mesh, tag);
			const kind = classify(name);
			if (!kind) continue;
			pushGroup(state, mesh, idx, colorFor(kind, name), tag, undefined, fieldNorm);
			faceTags.push(tag);
		}
		// 2) Implicit volume hulls — substrate / air / passivation / PML.
		//    buildVolumeBoundaries orients every face outward, so these can
		//    be backface culled: only the side turned towards the camera
		//    draws, and a shell's far side stops competing with whatever it
		//    encloses.
		//
		//    They still take a small constant push backwards so a conductor
		//    lying exactly ON a hull face (ground plane on a substrate
		//    surface) wins that tie. factor stays 0: glPolygonOffset scales
		//    it by the polygon's depth slope, so a non-zero factor grows
		//    with the viewing angle and would overrun genuine separations
		//    at grazing angles.
		const volBoundaries = buildVolumeBoundaries(mesh);
		const hullOffset: [number, number] | undefined = fieldNorm ? undefined : [0, 2];
		for (const [vtag, idx] of volBoundaries.entries()) {
			const name = physName(mesh, vtag);
			if (!name || name.startsWith('_mat_')) continue;
			pushGroup(state, mesh, idx, colorFor('dielectric', name), vtag, hullOffset, fieldNorm, true);
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

/** Convenience: wipe the field point cloud. Callers use this when toggling
 *  out of field mode. */
export function clearFieldCloud(state: GLState): void {
	setPointCloud(state, new Float32Array(0), new Float32Array(0));
}

// ── Volumetric field point sampling ──────────────────────────────────
//
// Same algorithm as `$lib/viz.ts` (the sampler the in-app MeshViewer uses),
// but in-line and synchronous for the embed which doesn't carry a worker.
// Two-stage draw: tet ∝ vol·w_max[t], then uniform-barycentric proposal
// inside the tet accepted with prob w(x)/w_max[t]. Marginal density per
// unit volume is the piecewise-linear w(x) — no density jumps at shared
// tet faces (the "visible tet edges" artifact of the old per-tet-mean
// weighting). See `viz.ts` for the full derivation.

/** Energy coverage floor — matches `viz.ts:ENERGY_FLOOR`. Close to 1 means
 *  almost-uniform spatial coverage with only a small linear bias toward
 *  high-energy regions; see `viz.ts` for the full derivation. */
const ENERGY_FLOOR = 0.7;

function buildTetVolumes(mesh: SceneMesh): Float64Array {
	const tets = mesh.tets;
	const nodes = mesh.nodes;
	const n = tets.length / 4;
	const vols = new Float64Array(n);
	for (let t = 0; t < n; t++) {
		const a = tets[t * 4], b = tets[t * 4 + 1],
		      c = tets[t * 4 + 2], d = tets[t * 4 + 3];
		const ax = nodes[a * 3], ay = nodes[a * 3 + 1], az = nodes[a * 3 + 2];
		const bx = nodes[b * 3], by = nodes[b * 3 + 1], bz = nodes[b * 3 + 2];
		const cx = nodes[c * 3], cy = nodes[c * 3 + 1], cz = nodes[c * 3 + 2];
		const dx = nodes[d * 3], dy = nodes[d * 3 + 1], dz = nodes[d * 3 + 2];
		const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
		const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
		const e3x = dx - ax, e3y = dy - ay, e3z = dz - az;
		const det = e1x * (e2y * e3z - e2z * e3y)
		          - e1y * (e2x * e3z - e2z * e3x)
		          + e1z * (e2x * e3y - e2y * e3x);
		vols[t] = Math.abs(det) / 6;
	}
	return vols;
}

function bsearchCdf(cdf: Float64Array, target: number): number {
	let lo = 0, hi = cdf.length - 1;
	while (lo < hi) {
		const mid = (lo + hi) >>> 1;
		if (cdf[mid] < target) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}

/** Uniform barycentric weights for a tetrahedron (sorted-triple trick). */
function uniformBary(out: [number, number, number, number]): void {
	let s = Math.random();
	let t = Math.random();
	let u = Math.random();
	if (s > t) [s, t] = [t, s];
	if (t > u) [t, u] = [u, t];
	if (s > t) [s, t] = [t, s];
	out[0] = s;
	out[1] = t - s;
	out[2] = u - t;
	out[3] = 1 - u;
}

/** Energy-weighted random sampling of N field points, with (A, B, C) phasor
 *  coefficients interpolated by barycentric weights. Same output shape
 *  `viz.ts:viz_sample` produces (plus maxE2/minE2 the embed uses for its
 *  colour range) — drop the positions / abc straight into `setPointCloud`. */
export function sampleFieldCloud(
	mesh: SceneMesh,
	fieldAbc: number[] | Float32Array,
	n: number,
): { positions: Float32Array; abc: Float32Array; maxE2: number; minE2: number } {
	const vols = buildTetVolumes(mesh);
	const tets = mesh.tets;
	const nodes = mesh.nodes;
	const nTets = vols.length;
	const nNodes = fieldAbc.length / 3;

	// Per-node time-averaged energy e_i = 0.5·(A_i + B_i).
	const eNode = new Float64Array(nNodes);
	let eGlobalMax = 0;
	for (let i = 0; i < nNodes; i++) {
		const e = 0.5 * (fieldAbc[i * 3] + fieldAbc[i * 3 + 1]);
		const ec = e > 0 ? e : 0;
		eNode[i] = ec;
		if (ec > eGlobalMax) eGlobalMax = ec;
	}
	if (eGlobalMax <= 0) {
		return {
			positions: new Float32Array(0),
			abc: new Float32Array(0),
			maxE2: 1, minE2: 1,
		};
	}
	const invEGlobal = 1 / eGlobalMax;

	// w(x) = ENERGY_FLOOR + (1−ENERGY_FLOOR)·e(x)/e_global_max  is affine in
	// the barycentric coords, so w_max inside a tet = w at its hottest corner.
	const wMaxTet = new Float64Array(nTets);
	for (let t = 0; t < nTets; t++) {
		let em = 0;
		for (let k = 0; k < 4; k++) {
			const en = eNode[tets[t * 4 + k]];
			if (en > em) em = en;
		}
		wMaxTet[t] = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * em * invEGlobal;
	}

	// Tet-pick CDF: weight = vol · w_max_tet.
	const cdf = new Float64Array(nTets);
	let acc = 0;
	for (let t = 0; t < nTets; t++) {
		acc += vols[t] * wMaxTet[t];
		cdf[t] = acc;
	}
	const totalWeight = acc || 1;

	const MAX_PROPOSALS = 64;
	const positions = new Float32Array(n * 3);
	const abc = new Float32Array(n * 3);
	const w: [number, number, number, number] = [0, 0, 0, 0];
	for (let i = 0; i < n; i++) {
		const ti = bsearchCdf(cdf, Math.random() * totalWeight);
		const a = tets[ti * 4], b = tets[ti * 4 + 1],
		      c = tets[ti * 4 + 2], d = tets[ti * 4 + 3];
		const ea = eNode[a], eb = eNode[b], ec_ = eNode[c], ed = eNode[d];
		const wmax = wMaxTet[ti];

		for (let attempt = 0; attempt < MAX_PROPOSALS; attempt++) {
			uniformBary(w);
			const eLocal = w[0] * ea + w[1] * eb + w[2] * ec_ + w[3] * ed;
			const wLocal = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * eLocal * invEGlobal;
			if (Math.random() * wmax <= wLocal) break;
		}
		positions[i * 3] = (
			w[0] * nodes[a * 3] + w[1] * nodes[b * 3] +
			w[2] * nodes[c * 3] + w[3] * nodes[d * 3]
		);
		positions[i * 3 + 1] = (
			w[0] * nodes[a * 3 + 1] + w[1] * nodes[b * 3 + 1] +
			w[2] * nodes[c * 3 + 1] + w[3] * nodes[d * 3 + 1]
		);
		positions[i * 3 + 2] = (
			w[0] * nodes[a * 3 + 2] + w[1] * nodes[b * 3 + 2] +
			w[2] * nodes[c * 3 + 2] + w[3] * nodes[d * 3 + 2]
		);
		const A = w[0] * fieldAbc[a * 3]     + w[1] * fieldAbc[b * 3]     + w[2] * fieldAbc[c * 3]     + w[3] * fieldAbc[d * 3];
		const B = w[0] * fieldAbc[a * 3 + 1] + w[1] * fieldAbc[b * 3 + 1] + w[2] * fieldAbc[c * 3 + 1] + w[3] * fieldAbc[d * 3 + 1];
		const C = w[0] * fieldAbc[a * 3 + 2] + w[1] * fieldAbc[b * 3 + 2] + w[2] * fieldAbc[c * 3 + 2] + w[3] * fieldAbc[d * 3 + 2];
		abc[i * 3] = A; abc[i * 3 + 1] = B; abc[i * 3 + 2] = C;
	}
	// Robust colormap range from the per-node energy distribution — see
	// the matching helper in `viz.ts`. Avoids the embed's auto-range
	// getting locked onto a single port-edge outlier.
	const { minE2, maxE2 } = nodeEnergyPercentile(fieldAbc);
	return { positions, abc, maxE2, minE2 };
}

/** 99th/1st percentile of per-node `|F|² ≈ (A + B)/2`. */
function nodeEnergyPercentile(fieldAbc: number[] | Float32Array): { minE2: number; maxE2: number } {
	const nNodes = fieldAbc.length / 3;
	const energies: number[] = [];
	for (let ni = 0; ni < nNodes; ni++) {
		const e2 = 0.5 * (fieldAbc[ni * 3] + fieldAbc[ni * 3 + 1]);
		if (e2 > 0) energies.push(e2);
	}
	if (energies.length === 0) return { minE2: 1, maxE2: 1 };
	energies.sort((a, b) => a - b);
	const last = energies.length - 1;
	const lo = energies[Math.min(last, Math.floor(energies.length * 0.01))];
	const hi = energies[Math.min(last, Math.floor(energies.length * 0.99))];
	return { minE2: lo, maxE2: hi };
}
