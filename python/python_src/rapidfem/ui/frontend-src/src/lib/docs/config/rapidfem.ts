// Central configuration for the RapidFEM documentation site.
// Single-package — all site content is defined here.

export const site = {
	name: 'RapidFEM',
	tagline: 'Electromagnetic FEM solver: frequency and time domain',
	description:
		'An electromagnetic FEM solver written in Rust, distributed as a Python package. ' +
		'A frequency-domain backend (second-kind Nédélec edge elements, complex-symmetric ' +
		'sparse linear algebra) and a time-domain DGTD backend (a discontinuous-Galerkin ' +
		'Maxwell operator advanced by an exponential integrator). Waveguide and lumped ports ' +
		'for both.'
};

// External links. `demo` points at the existing static notebook demo that
// already ships the pre-baked examples — the docs do not re-render them.
export const external = {
	github: 'https://github.com/milanofthe/rapidfem',
	pypi: 'https://pypi.org/project/rapidfem',
	demo: 'https://fem.rapidpassives.org'
};

export interface Feature {
	title: string;
	description: string;
}

export const features: Feature[] = [
	{
		title: 'Nedelec-2 Elements',
		description: '20 DOFs per tetrahedron, a vector edge basis for the curl-curl form of Maxwell.'
	},
	{
		title: 'Excitations',
		description: 'Rectangular waveguide ports, lumped TEM ports, and absorbing boundaries of order 1 and 2.'
	},
	{
		title: 'Anisotropic PML',
		description: 'Stretched-coordinate perfectly matched layer for open-domain radiation problems.'
	},
	{
		title: 'Sparse Solvers',
		description: 'Pure-Rust rslab complex-symmetric LDLT, vendored, no native dependencies.'
	},
	{
		title: 'Frequency Sweep',
		description: 'Assembles E/B once, analyses the pattern once, solves neighbouring points by COCG on an earlier factorisation.'
	},
	{
		title: 'Eigenmode Solver',
		description: 'Shift-invert Lanczos on the complex-symmetric system for resonant mode analysis.'
	},
	{
		title: 'Adaptive Refinement',
		description: 'Residual error estimator with Dörfler marking, exports a size field for re-meshing.'
	},
	{
		title: 'Time-Domain DGTD',
		description: 'A nodal discontinuous-Galerkin Maxwell operator: broadband transients and modal-port S-parameters from a single run.'
	},
	{
		title: 'Exponential Integrator',
		description: 'Matrix-free Krylov / ETD time stepping, exact for the linear system at any step size with no CFL stability limit.'
	},
	{
		title: 'Model-Order Reduction',
		description: 'Krylov projection compiles the DGTD operator into a compact reduced model for fast repeated propagation.'
	},
	{
		title: 'Output Formats',
		description: 'Touchstone (.sNp), VTK field export, and far-field NFFT radiation patterns.'
	}
];

export interface InstallOption {
	name: string;
	command: string;
	note?: string;
}

export const installation: InstallOption[] = [
	{ name: 'Solver', command: 'pip install rapidfem' },
	{ name: 'Solver + local UI', command: 'pip install rapidfem[ui]' }
];

export interface QuickStart {
	title: string;
	description: string;
	code: string;
}

// Two entry points sharing one geometry / material / physics API: the
// frequency-domain sweep and the time-domain DGTD backend. The landing
// page presents them as switchable tabs.
export const quickstart: QuickStart[] = [
	{
		title: 'Frequency domain',
		description:
			'Build geometry, attach materials and physics to entities, then run any number of analyses on the same Problem.',
		code: `import numpy as np
import rapidfem as rf

# Build geometry; attach materials + physics directly to entities
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
air = g.box(22.86e-3, 10.16e-3, 30e-3, position=(-11.43e-3, -5.08e-3, 0),
            material=rf.Air())

rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)

g.mesh()

# Define the problem once, run any number of analyses on it
prob = rf.ProblemFD(g)
result = prob.sweep(np.linspace(8e9, 12e9, 21))
print(result.frequencies.shape, result.sparams.shape)`
	},
	{
		title: 'Time domain',
		description:
			'The same geometry compiles into a DGTD time-domain problem. Drive a port with a band-limited pulse, watch the full transient evolve in 3D, advanced by exact CFL-free exponential time stepping.',
		code: `import rapidfem as rf

# The same geometry / material / physics API as the FD backend
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
air = g.box(22.86e-3, 10.16e-3, 30e-3, position=(-11.43e-3, -5.08e-3, 0),
            material=rf.Air())

p_in  = rf.RectWaveguidePort(air.faces.min(axis="z"))
p_out = rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)

g.mesh()

# Discontinuous-Galerkin TD problem; CFL-free exponential time stepping
ptd = rf.ProblemTD(g, order=2, flux="upwind")

# Drive the input port with a band-limited TE10 pulse and propagate
pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=10e9)
traj = ptd.transient(port=p_in, waveform=pulse, dt=3e-12, steps=320)

rf.show(traj)                                            # 3D field animation
rf.show(ptd.port_signals(traj, [p_in, p_out], dt=3e-12)) # port waveforms`
	}
];

export interface ApiModule {
	name: string;
	description: string;
}

// Curated module order and descriptions for the API reference page.
// Modules not listed here still render — sorted alphabetically after these.
export const apiModules: ApiModule[] = [
	{ name: 'rapidfem', description: 'Top-level package: Geometry, Problem, and re-exported helpers' },
	{ name: 'rapidfem.geometry', description: 'Geometry construction, entities, face selection, meshing' },
	{ name: 'rapidfem.materials', description: 'Material models: Air, Dielectric, lossy and dispersive permittivity' },
	{ name: 'rapidfem.physics', description: 'Ports and boundary conditions: waveguide, lumped, PEC, ABC, PML' },
	{ name: 'rapidfem.excitation', description: 'Time-domain excitation waveforms: Gaussian and modulated pulses' },
	{ name: 'rapidfem.problem.fd', description: 'Frequency-domain solver: driven sweep, eigenmode, far-field' },
	{ name: 'rapidfem.problem.td', description: 'Time-domain DGTD solver: exponential stepping, transients, modal-port S-parameters, model-order reduction' },
	{ name: 'rapidfem.io', description: 'Result containers, Touchstone, VTK, and far-field export' },
	{ name: 'rapidfem.rfic', description: 'RFIC layout import: geometry from JSON layer stacks' }
];

// Sidebar navigation. API/Examples paths use the version when known.
export interface SidebarItem {
	title: string;
	path: string;
	icon?: string;
	external?: boolean;
}

export function getSidebarItems(version?: string): SidebarItem[] {
	const v = version || 'latest';
	return [
		{ title: 'Overview', path: '', icon: 'home' },
		{ title: 'API Reference', path: `${v}/api`, icon: 'braces' }
	];
}
