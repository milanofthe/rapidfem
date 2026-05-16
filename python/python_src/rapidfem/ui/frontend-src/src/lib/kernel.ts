/**
 * HTTP-based kernel client.
 *
 * Each notebook file gets a long-lived worker subprocess (managed by
 * rapidfem.ui.runner). Cell execution is:
 *   1. POST /api/cell/run  → server kicks off subprocess, returns cell_id
 *   2. POST /api/cell/poll → long-polls (100 ms) for stream/display/error/done
 *      events until `done: true` lands in the response.
 *
 * The previous WS-based path is gone — see commits leading to
 * feature/subprocess-worker for the rationale (Werkzeug+wsproto+deflate
 * frame-encoding bugs that bit on >64 KiB messages).
 */

import { api_base, type GeometryPayload, type MeshPayload, type PythonError } from './api';

export type StreamKind = 'stdout' | 'stderr';

export type KernelEvent =
	| { type: 'stream'; stream: StreamKind; value: string }
	| { type: 'display'; kind: 'geometry'; name: string; payload: GeometryPayload }
	| { type: 'display'; kind: 'mesh'; name: string; payload: MeshPayload }
	| { type: 'display'; kind: 'result'; name: string; payload: SolveResultPayload }
	| { type: 'display'; kind: 'error'; name: string; error: PythonError }
	| { type: 'error'; id?: string; error: string; traceback?: string }
	| { type: 'done'; id: string; ok: boolean }
	| { type: 'reset-ack' }
	| { type: 'worker-exit' };

export interface SolveResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_driven: number;
	n_freq: number;
	n_dofs: number;
	n_tets: number;
	/** True when this is an eigenmode result — `frequencies[i]` is the i-th
	 *  resonant frequency, `sparams` is empty, the field slider becomes a
	 *  mode-index slider. */
	eigenmode?: boolean;
	/** Per-mode Q factor (only present when eigenmode=true). */
	q_factors?: number[];
	solve_time_s: number;
	fields?: (number[] | null)[][];
	/** Conduction current density J = σE per (freq, port), same shape as `fields`. */
	fields_j?: (number[] | null)[][] | null;
	/** Magnetic field H = ∇×E / (jωμ) per (freq, port), same shape as `fields`. */
	fields_h?: (number[] | null)[][] | null;
}

export interface ExecuteOptions {
	cell_id: string;
	file: string;
	code: string;
	reset?: boolean;
	onStarted?: () => void;
	onStream?: (stream: StreamKind, line: string) => void;
	onDisplay?: (kind: 'geometry' | 'mesh' | 'result', payload: unknown, name: string) => void;
}

export interface ExecuteResult {
	ok: boolean;
	error?: PythonError;
}

async function post_json<T>(path: string, body: object): Promise<T> {
	const res = await fetch(api_base() + path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) {
		const text = await res.text().catch(() => '');
		throw new Error(`${path}: HTTP ${res.status} ${text.slice(0, 200)}`);
	}
	return res.json() as Promise<T>;
}

export class KernelClient {
	async execute(opts: ExecuteOptions): Promise<ExecuteResult> {
		console.log('[kernel] execute', { cell_id: opts.cell_id, file: opts.file, code_len: opts.code.length });
		try {
			const run = await post_json<{ ok: boolean; cell_id?: string; error?: string }>(
				'/api/cell/run',
				{
					file: opts.file,
					code: opts.code,
					reset: !!opts.reset,
					cell_id: opts.cell_id,
				},
			);
			if (!run.ok || !run.cell_id) {
				console.warn('[kernel] /api/cell/run rejected', run);
				return {
					ok: false,
					error: { type: 'RunError', message: run.error ?? 'cell-run failed', traceback: '' },
				};
			}
			opts.onStarted?.();
			const r = await this.drain_until_done(opts);
			if (!r.ok) console.warn('[kernel] execute returning ok=false', r);
			return r;
		} catch (e) {
			console.error('[kernel] execute caught:', e);
			return {
				ok: false,
				error: { type: 'NetworkError', message: String(e), traceback: '' },
			};
		}
	}

	private async drain_until_done(opts: ExecuteOptions): Promise<ExecuteResult> {
		const target = opts.cell_id;
		let last_error: PythonError | undefined;
		// Poll loop. The server's /api/cell/poll long-polls 100 ms so this
		// stays cheap and responsive.
		while (true) {
			let resp: { messages: KernelEvent[]; done: boolean };
			try {
				resp = await post_json<{ messages: KernelEvent[]; done: boolean }>(
					'/api/cell/poll',
					{ file: opts.file },
				);
			} catch (e) {
				return {
					ok: false,
					error: { type: 'NetworkError', message: String(e), traceback: '' },
				};
			}
			for (const evt of resp.messages) {
				if (evt.type === 'stream') {
					// Server sends raw chunks; split into lines so the
					// notebook log panel renders cleanly.
					for (const line of evt.value.split('\n')) {
						if (line.length > 0) opts.onStream?.(evt.stream, line);
					}
				} else if (evt.type === 'display') {
					if (evt.kind === 'error') {
						opts.onStream?.('stderr', `${evt.error.type}: ${evt.error.message}`);
					} else {
						// onDisplay callbacks touch reactive state (mesh, S-params,
						// fields). A throw in there used to bubble all the way up
						// and mark the cell failed — but the cell itself already
						// finished cleanly on the worker. Log + skip instead so
						// the user sees the diagnostic, the cell stays OK, and
						// the next display events keep flowing.
						try {
							opts.onDisplay?.(evt.kind, evt.payload, evt.name);
						} catch (err) {
							console.error('[kernel] onDisplay threw:', err, evt);
							opts.onStream?.('stderr',
								`[ui] display "${evt.kind}" failed to render: ${err}`);
						}
					}
				} else if (evt.type === 'error') {
					last_error = {
						type: 'ExecError',
						message: evt.error,
						traceback: evt.traceback ?? '',
					};
					opts.onStream?.('stderr', evt.error);
					if (evt.traceback) opts.onStream?.('stderr', evt.traceback);
				} else if (evt.type === 'done') {
					if (evt.id === target) {
						console.log('[kernel] cell done', { cell_id: target, ok: evt.ok, has_error: !!last_error });
						return { ok: evt.ok, error: last_error };
					}
					console.warn('[kernel] done event for non-matching cell', {
						received_id: evt.id, target,
					});
					opts.onStream?.('stderr',
						`[ui] WARNING: received done for cell_id="${evt.id}" but expected "${target}" — cell may show stale status`);
				} else if (evt.type === 'worker-exit') {
					console.warn('[kernel] worker exited');
					opts.onStream?.('stderr', '[ui] worker process exited');
					return {
						ok: false,
						error: { type: 'WorkerExit', message: 'Worker process exited', traceback: '' },
					};
				}
			}
			// If the server says we're done but we never saw our own `done`,
			// something raced — bail with whatever error we caught.
			if (resp.done && resp.messages.every((e) => e.type !== 'done' || e.id !== target)) {
				return last_error
					? { ok: false, error: last_error }
					: { ok: true };
			}
			// Empty-but-not-done poll → idle wait until next tick. The server
			// already blocks ~100 ms so we don't add a JS-side delay here.
		}
	}

	async reset(file: string): Promise<void> {
		try {
			await post_json('/api/cell/reset', { file });
		} catch (e) {
			console.warn('[kernel] reset failed:', e);
		}
	}

	/** Send SIGINT to the worker subprocess — the cell-run's `exec` will
	 *  raise `KeyboardInterrupt`, propagate to the error path, and emit an
	 *  `error` event back through the normal poll stream. */
	async interrupt(file: string): Promise<boolean> {
		try {
			const r = await post_json<{ ok: boolean }>('/api/cell/interrupt', { file });
			return !!r.ok;
		} catch (e) {
			console.warn('[kernel] interrupt failed:', e);
			return false;
		}
	}
}

// ── Static-demo replay client ──────────────────────────────────────────
//
// Used when the frontend is built with VITE_STATIC_MODE=1. Loads the bake
// artefacts produced by `scripts/bake_demo.py` from `static/demo/` and
// replays them through the same execute() callbacks as the live WS path,
// so consumers (Cell, Notebook, +page.svelte) don't have to branch.

import { IS_STATIC_MODE, DEMO_BASE } from './static_mode';

interface BakedFieldsStub {
	$bin: true;
	magic: number;
	version: number;
	n_freq: number;
	n_port: number;
	stride: number;
	url: string;
}

interface BakedDisplayEvent {
	kind: 'geometry' | 'mesh' | 'result' | 'error';
	name: string;
	payload?: Record<string, unknown>;
	error?: PythonError;
}

interface BakedCell {
	marker: string | null;
	code: string;
	status: 'ok' | 'error';
	stream_lines: { stream: StreamKind; line: string }[];
	display_events: BakedDisplayEvent[];
	error?: PythonError;
}

interface BakedExample {
	name: string;
	filename: string;
	source: string;
	cells: BakedCell[];
}

interface ManifestEntry {
	name: string;
	filename: string;
	json: string;
	bin_files: string[];
	n_cells: number;
}

interface Manifest {
	version: number;
	baked_at: number;
	examples: ManifestEntry[];
}

class StaticKernelClient {
	private manifest_promise: Promise<Manifest> | null = null;
	private examples = new Map<string, Promise<BakedExample>>();
	private bins = new Map<string, Promise<ArrayBuffer>>();

	private load_manifest(): Promise<Manifest> {
		if (!this.manifest_promise) {
			this.manifest_promise = fetch(`${DEMO_BASE}manifest.json`)
				.then((r) => {
					if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
					return r.json() as Promise<Manifest>;
				});
		}
		return this.manifest_promise;
	}

	private async load_example(filename: string): Promise<BakedExample> {
		// File path keys are stored as bare filenames in the manifest, but
		// callers may pass a longer "<path>" — match on basename.
		const base = filename.split(/[\\/]/).pop() ?? filename;
		let p = this.examples.get(base);
		if (p) return p;
		p = this.load_manifest().then(async (m) => {
			const entry = m.examples.find((e) => e.filename === base);
			if (!entry) throw new Error(`baked example not found: ${base}`);
			const r = await fetch(`${DEMO_BASE}${entry.json}`);
			if (!r.ok) throw new Error(`example fetch failed: ${r.status}`);
			return r.json() as Promise<BakedExample>;
		});
		this.examples.set(base, p);
		return p;
	}

	private async load_bin(url: string): Promise<ArrayBuffer> {
		let p = this.bins.get(url);
		if (p) return p;
		p = fetch(`${DEMO_BASE}${url}`).then((r) => {
			if (!r.ok) throw new Error(`bin fetch failed: ${r.status} ${url}`);
			return r.arrayBuffer();
		});
		this.bins.set(url, p);
		return p;
	}

	/** Hydrate the binary field stub back into the nested-array shape the
	 *  live UI's `fields_raw` expects: ``(number[] | null)[][]``. */
	private async hydrate_fields(stub: BakedFieldsStub): Promise<(number[] | null)[][]> {
		const buf = await this.load_bin(stub.url);
		const dv = new DataView(buf);
		const magic = dv.getUint32(0, true);
		const version = dv.getUint32(4, true);
		if (magic !== stub.magic || version !== stub.version) {
			throw new Error(`field bin header mismatch (got ${magic.toString(16)}/${version})`);
		}
		const n_freq = dv.getUint32(8, true);
		const n_port = dv.getUint32(12, true);
		const stride = dv.getUint32(16, true);
		const mask_off = 20;
		const mask = new Uint8Array(buf, mask_off, n_freq * n_port);
		// The mask block is zero-padded so the float block starts on a
		// 4-byte boundary — required by Float32Array's offset constraint.
		const mask_padded = (mask.byteLength + 3) & ~3;
		const floats_off = mask_off + mask_padded;
		const all_floats = new Float32Array(buf, floats_off);

		const out: (number[] | null)[][] = [];
		let cursor = 0;
		for (let fi = 0; fi < n_freq; fi++) {
			const row: (number[] | null)[] = [];
			for (let pi = 0; pi < n_port; pi++) {
				if (mask[fi * n_port + pi] === 0) {
					row.push(null);
				} else {
					row.push(Array.from(all_floats.subarray(cursor, cursor + stride)));
					cursor += stride;
				}
			}
			out.push(row);
		}
		return out;
	}

	/** Find the baked cell matching the source the editor just sent. */
	private find_cell(example: BakedExample, code: string): BakedCell | null {
		const eq = (a: string, b: string) => a === b || a.trimEnd() === b.trimEnd();
		const exact = example.cells.find((c) => eq(c.code, code));
		if (exact) return exact;
		// Fallback: trimmed match in case Notebook.serialize() added/dropped
		// a trailing newline.
		return example.cells.find((c) => c.code.trim() === code.trim()) ?? null;
	}

	execute(opts: ExecuteOptions): Promise<ExecuteResult> {
		return (async () => {
			opts.onStarted?.();
			let example: BakedExample;
			try {
				example = await this.load_example(opts.file);
			} catch (err) {
				opts.onStream?.('stderr', String(err));
				return { ok: false, error: { type: 'StaticDemoError', message: String(err), traceback: '' } };
			}
			const cell = this.find_cell(example, opts.code);
			if (!cell) {
				opts.onStream?.('stderr', '[static-demo] no baked record for this cell');
				return { ok: false, error: { type: 'StaticDemoError', message: 'no cell match', traceback: '' } };
			}

			for (const s of cell.stream_lines) opts.onStream?.(s.stream, s.line);

			for (const ev of cell.display_events) {
				if (ev.kind === 'error') {
					if (ev.error) opts.onStream?.('stderr', `${ev.error.type}: ${ev.error.message}`);
					continue;
				}
				let payload = ev.payload ?? {};
				if (ev.kind === 'result') {
					const f = (payload as Record<string, unknown>).fields as BakedFieldsStub | undefined | null;
					if (f && (f as BakedFieldsStub).$bin) {
						try {
							const hydrated = await this.hydrate_fields(f as BakedFieldsStub);
							payload = { ...payload, fields: hydrated };
						} catch (err) {
							opts.onStream?.('stderr', `[static-demo] field load failed: ${err}`);
						}
					}
				}
				opts.onDisplay?.(ev.kind, payload, ev.name);
			}

			if (cell.status === 'error') {
				return { ok: false, error: cell.error };
			}
			return { ok: true };
		})();
	}

	reset(_file: string) {
		/* no-op in static mode — the namespace is whatever was baked */
	}
}

let _singleton: KernelClient | StaticKernelClient | null = null;
export function get_kernel(): KernelClient | StaticKernelClient {
	if (!_singleton) {
		_singleton = IS_STATIC_MODE ? new StaticKernelClient() : new KernelClient();
	}
	return _singleton;
}
