# rapidfem developer tasks. `make help` lists targets.
# The test suite has two pillars (see python/tests/README.md):
#   1. sympy kernel goldens   — Rust tests pinned to symbolic ground truth
#   2. phenomenon geometries  — Python tests vs analytical / conservation laws

.PHONY: help test test-rust test-py test-py-fast build gen-goldens

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: test-rust build test-py ## run everything (rust goldens + rebuild + full physics)

test-rust: ## fast: core/fd unit + all sympy kernel goldens (incl. TD)
	cargo test --release -p rapidfem-core -p rapidfem-fd
	cargo test -p rapidfem-td \
	  --test dg_basis_golden_test \
	  --test rhs_curl_golden_test \
	  --test propagator_expm_test \
	  --test lserk4_convergence_test

build: ## rebuild the python extension (release; needed after any Rust change)
	cd python && maturin develop --release

test-py-fast: ## fast python tests only (renormalization math, no FEM solves)
	cd python && python3 -m pytest -m "not slow" -q

test-py: ## full phenomenon suite (real FEM solves, several minutes)
	cd python && python3 -m pytest -q

gen-goldens: ## regenerate every sympy-derived Rust golden test
	python3 derivations/nedelec2/emit_element_golden.py
	python3 derivations/nedelec2/emit_coefficients_test.py
	python3 derivations/nedelec2/emit_tri_mass_golden.py
	python3 derivations/nedelec2/emit_interp_golden.py
	python3 derivations/waveguide/emit_waveguide_golden.py
	python3 derivations/materials/emit_debye_golden.py
	python3 derivations/materials/emit_material_golden.py
	python3 derivations/td_dg/emit_dg_basis_golden.py
	python3 derivations/td_dg/emit_rhs_curl_golden.py
	python3 derivations/td_dg/emit_propagator_golden.py
