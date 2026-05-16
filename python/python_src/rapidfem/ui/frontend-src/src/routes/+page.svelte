<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { IS_STATIC_MODE } from '$lib/static_mode';

	// Local `rapidfem serve` has no use for the landing page — go straight
	// to /notebook. The landing route is only meaningful in the GH-Pages
	// build (VITE_STATIC_MODE=1) where this is the fem.rapidpassives.org
	// entry point. At build time IS_STATIC_MODE is false in the local
	// serve bundle, so the {#if} below drops the landing DOM from the
	// prerendered HTML too — no flash of unwanted content.
	onMount(() => {
		if (!IS_STATIC_MODE) {
			const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
			goto(`${base}/notebook`, { replaceState: true });
			return;
		}
		// Static-demo only: define <fem-viewer> for the card grid.
		if (customElements.get('fem-viewer')) return;
		const script = document.createElement('script');
		script.src = `${import.meta.env.BASE_URL || '/'}embed/fem-viewer.js`.replace(/\/+/g, '/');
		document.head.appendChild(script);
	});

	const examples = [
		{
			name: 'wr90',
			label: 'WR-90 Waveguide',
			desc: 'Rectangular waveguide section, 21-pt sweep across the X-band TE₁₀ mode',
		},
		{
			name: 'wr90_pml',
			label: 'WR-90 + PML',
			desc: 'PML as a matched load — |S₁₁| at the numerical floor across the whole band',
		},
		{
			name: 'iris_filter',
			label: 'Iris Bandpass',
			desc: 'Three inductive irises in WR-90 form a 3rd-order Chebyshev passband near 10 GHz',
			fieldFreq: 24,  // 10.72 GHz on the 8.2-12.4 GHz / 41-pt sweep
		},
		{
			name: 'patch_antenna',
			label: 'Patch Antenna',
			desc: 'Edge-fed 2.4 GHz patch on FR-4 with a 5-slab PML enclosure and far-field pattern',
			fieldFreq: 0,   // 2.0 GHz, low end of the sweep — strongest near-field
			fieldMode: 'log',
		},
		{
			name: 'coax_step',
			label: 'Coax Step',
			desc: '50 → 75 ohm coaxial impedance discontinuity, native coax TEM ports',
			fieldFreq: 0,   // 1 GHz, the low end of the sweep
		},
		{
			name: 'microstrip_line',
			label: 'Microstrip Z₀',
			desc: '50 ohm microstrip on RO4003C, narrowed to the λ_g/2 sweet spot for clean S-params',
			fieldMode: 'log',
		},
	] as Array<{
		name: string; label: string; desc: string;
		fieldFreq?: number; fieldMode?: 'lin' | 'log';
	}>;

	const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
	const notebook_url = `${base}/notebook`;
	const embed_test_url = `${base}/embed/test`;

	const install_cmd = 'pip install rapidfem';
	let copied = false;
	let copy_timer: ReturnType<typeof setTimeout> | null = null;
	function copyInstall() {
		try {
			navigator.clipboard.writeText(install_cmd);
			copied = true;
			if (copy_timer) clearTimeout(copy_timer);
			copy_timer = setTimeout(() => { copied = false; }, 1400);
		} catch {}
	}
</script>

<svelte:head>
	<title>RapidFEM &mdash; frequency-domain Maxwell FEM in Rust</title>
	<meta name="description" content="Open-source frequency-domain electromagnetic FEM solver. Nedelec-2 edge elements, complex-symmetric sparse linear algebra, Python API, browser notebook UI." />
</svelte:head>

{#if IS_STATIC_MODE}
<div class="page">
	<header>
		<a class="brand" href="/">
			<img src="{base}/favicon.svg" alt="RapidFEM" class="logo" />
		</a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab" href={notebook_url}>Notebook</a>
			<a class="tab" href={embed_test_url}>Embed</a>
		</nav>
	</header>
	<div class="landing">
		<div class="hero">
			<h1>RapidFEM</h1>
			<p>Frequency-domain electromagnetic FEM in Rust. Nédélec-2 edge elements, complex-symmetric sparse solvers, Python API, browser notebook UI.</p>
		</div>
		<div class="quickstart">
			<div class="install-line">
				<span class="install-prompt">$</span>
				<code class="install-cmd">{install_cmd}</code>
				<button class="copy-btn" on:click={copyInstall} aria-label="Copy install command">
					{copied ? '✓ copied' : 'copy'}
				</button>
			</div>
			<a class="cta" href={notebook_url}>
				<span>Open the Notebook</span>
				<span class="cta-arrow">→</span>
			</a>
		</div>
		<div class="cards">
			{#each examples as ex}
				<a class="card" href={`${notebook_url}?example=${ex.name}`}>
					<div class="card-preview">
						{@html `<fem-viewer src="${base}/demo/${ex.name}.json" rotate cycle speed="0.6" field-samples="5000"${ex.fieldFreq !== undefined ? ` field-freq="${ex.fieldFreq}"` : ''}${ex.fieldMode ? ` field-mode="${ex.fieldMode}"` : ''} width="100%" height="240px"></fem-viewer>`}
					</div>
					<div class="card-info">
						<h3>{ex.label}</h3>
						<p>{ex.desc}</p>
					</div>
				</a>
			{/each}
		</div>
		<a class="embed-hint" href={embed_test_url}>
			<span class="embed-tag">&lt;fem-viewer&gt;</span>
			<span>Embed FEM results on your website</span>
		</a>
	</div>
	<footer class="landing-footer">
		<a href="https://github.com/milanofthe/rapidfem" target="_blank" rel="noopener">GitHub</a>
		<span class="sep">/</span>
		<a href="https://pypi.org/project/rapidfem/" target="_blank" rel="noopener">PyPI</a>
		<span class="sep">/</span>
		<a href={notebook_url}>Notebook</a>
		<span class="sep">/</span>
		<a href="https://rapidpassives.org" target="_blank" rel="noopener">RapidPassives</a>
		<span class="sep">/</span>
		<a href="https://milanrother.com" target="_blank" rel="noopener">Milan Rother</a>
	</footer>
</div>
{/if}

<style>
	/* Notebook's palette + scale (global vars from app.css) — but a
	 * lifted mid-gray bg so the tiles read as cards instead of floating
	 * voids on the viewer-dark default. Fixed viewport height with flex
	 * column so the header stays anchored at the top, the footer at the
	 * bottom, and `.landing` scrolls internally — matches rapidpassives. */
	.page {
		background: #1c1c21;
		height: 100vh;
		display: flex;
		flex-direction: column;
	}

	/* ── Header (matches /demo notebook header) ──────────────────────── */
	header {
		display: flex;
		align-items: center;
		padding: 0 var(--space-xl);
		gap: var(--space-md);
		height: 36px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		text-decoration: none;
	}
	.logo {
		height: 22px;
		width: auto;
		display: block;
	}
	.nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}
	.tabs {
		display: flex;
		gap: 0;
		height: 100%;
	}
	header .tab {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-size: var(--fs-xs);
		font-weight: 600;
		font-family: var(--font-mono);
		letter-spacing: 0.5px;
		color: var(--text-dim);
		text-decoration: none;
		transition: color var(--transition);
	}
	header .tab:hover { color: var(--text-muted); }
	header .tab.active { color: var(--accent); }
	.landing {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 32px;
		padding: 40px;
		overflow-y: auto;
	}
	.hero {
		text-align: center;
		margin-bottom: 16px;
	}
	.hero h1 {
		font-size: 28px;
		font-weight: 700;
		color: var(--accent);
		font-family: var(--font-mono);
		letter-spacing: 2px;
		margin-bottom: 10px;
	}
	.hero p {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 480px;
		font-family: var(--font-mono);
		line-height: 1.5;
	}
	.quickstart {
		display: flex;
		align-items: stretch;
		gap: 14px;
		flex-wrap: wrap;
		justify-content: center;
		margin-top: -8px;
	}
	.install-line {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 0 12px;
		height: 36px;
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.install-prompt {
		color: var(--text-dim);
		user-select: none;
	}
	.install-cmd {
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.3px;
	}
	.copy-btn {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		letter-spacing: 0.5px;
		padding: 3px 8px;
		cursor: pointer;
		text-transform: uppercase;
		transition: color var(--transition), border-color var(--transition);
	}
	.copy-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
	}
	.cta {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 0 18px;
		height: 36px;
		background: var(--accent);
		color: var(--bg);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 700;
		letter-spacing: 0.6px;
		text-decoration: none;
		text-transform: uppercase;
		transition: filter var(--transition), transform var(--transition);
	}
	.cta:hover {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}
	.cta-arrow {
		font-size: 14px;
		line-height: 1;
		transition: transform var(--transition);
	}
	.cta:hover .cta-arrow { transform: translateX(2px); }
	.quickstart-note {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
		text-align: center;
		max-width: 560px;
		line-height: 1.6;
		margin-top: -12px;
	}
	.quickstart-note a {
		color: var(--text-muted);
		text-decoration: none;
		border-bottom: 1px solid var(--border);
		transition: color var(--transition), border-color var(--transition);
	}
	.quickstart-note a:hover {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}
	.cards {
		display: flex;
		gap: 18px;
		flex-wrap: wrap;
		justify-content: center;
		max-width: 1100px;
	}
	.card {
		width: 320px;
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		text-decoration: none;
		color: inherit;
		transition: border-color var(--transition), transform var(--transition);
		display: flex;
		flex-direction: column;
	}
	.card:hover {
		border-color: var(--accent);
		transform: translateY(-2px);
	}
	.card-preview {
		width: 100%;
		overflow: hidden;
	}
	.card-info {
		padding: 12px 14px;
		border-top: 1px solid var(--border-subtle);
	}
	.card-info h3 {
		font-size: var(--fs-sm);
		font-weight: 600;
		color: var(--accent);
		font-family: var(--font-mono);
		margin-bottom: 4px;
	}
	.card-info p {
		font-size: var(--fs-xs);
		color: var(--text-dim);
		line-height: 1.4;
		font-family: var(--font-mono);
	}
	.embed-hint {
		display: flex;
		align-items: center;
		gap: 10px;
		text-decoration: none;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
		transition: color var(--transition);
	}
	.embed-hint:hover { color: var(--accent); }
	.embed-tag {
		font-weight: 600;
		color: var(--text-muted);
		border: 1px solid var(--border);
		padding: 3px 8px;
		transition: border-color var(--transition), color var(--transition);
	}
	.embed-hint:hover .embed-tag {
		border-color: var(--accent);
		color: var(--accent);
	}
	.landing-footer {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 0 16px;
		height: 36px;
		background: var(--bg-surface);
		border-top: 1px solid var(--border);
		flex-shrink: 0;
	}
	.sep {
		color: var(--border);
		font-size: var(--fs-xs);
	}
	.landing-footer a {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		text-decoration: none;
		letter-spacing: 0.3px;
		transition: color var(--transition);
	}
	.landing-footer a:hover { color: var(--accent); }
</style>
