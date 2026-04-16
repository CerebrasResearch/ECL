.PHONY: help env install dev test lint format typecheck clean figures tables appendix all-experiments

# NFS disk space is limited — always use python -B to suppress .pyc writes
PYTHON = PYTHONPATH=src python -B

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda environment
	bash create_env.sh

install:  ## Install package in editable mode (WARNING: writes to NFS, prefer PYTHONPATH=src)
	pip install -e .

dev:  ## Install with dev + model dependencies
	pip install -e ".[all]"
	pre-commit install

test:  ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=ecl --cov-report=term-missing --cov-report=html

lint:  ## Run linter
	ruff check src/ tests/ scripts/

format:  ## Auto-format code
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck:  ## Run type checker
	mypy src/ecl/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ---------- Main paper results (Section 11) ----------

figures:  ## Generate all main figures (fig01-fig12)
	@for f in scripts/fig[0-9]*.py; do echo "=== Running $$f ==="; $(PYTHON) "$$f"; done

tables:  ## Generate all main tables (tab01-tab07)
	@for f in scripts/tab[0-9]*.py; do echo "=== Running $$f ==="; $(PYTHON) "$$f"; done

# ---------- Appendix results ----------

appendix:  ## Generate appendix figures and tables (figA*, tabA*)
	@for f in scripts/figA*.py scripts/tabA*.py; do echo "=== Running $$f ==="; $(PYTHON) "$$f"; done

# ---------- Individual experiment recipes ----------

fig01:  ## Exp 1a: Influence profiles — promoter loci
	$(PYTHON) scripts/fig01_influence_profiles_promoter.py

fig02:  ## Exp 1b: Influence profiles — enhancer loci
	$(PYTHON) scripts/fig02_influence_profiles_enhancer.py

fig03:  ## Exp 2a: Cumulative influence curves with bootstrap CIs
	$(PYTHON) scripts/fig03_cumulative_influence.py

fig04:  ## Exp 3: Directional ECL (upstream/downstream)
	$(PYTHON) scripts/fig04_directional_ecl.py

fig05:  ## Exp 4b: Perturbation scatter — substitution vs shuffle
	$(PYTHON) scripts/fig05_perturbation_scatter.py

fig06:  ## Exp 5: Trained vs random weights
	$(PYTHON) scripts/fig06_trained_vs_random.py

fig07:  ## Exp 7a: Locus-class violin plots
	$(PYTHON) scripts/fig07_locus_class_violin.py

fig08:  ## Exp 7b: ECL vs gene length
	$(PYTHON) scripts/fig08_ecl_vs_gene_length.py

fig09:  ## Exp 8a: Paired model comparison plot
	$(PYTHON) scripts/fig09_model_comparison_paired.py

fig10:  ## Exp 9a: Biological validation profiles
	$(PYTHON) scripts/fig10_biological_validation.py

fig11:  ## Exp 10a: Interaction influence heatmap
	$(PYTHON) scripts/fig11_interaction_heatmap.py

fig12:  ## Exp 10b: gNIAH sensitivity curves
	$(PYTHON) scripts/fig12_gniah_sensitivity.py

tab01:  ## Table 1: Model taxonomy (static)
	$(PYTHON) scripts/tab01_model_taxonomy.py

tab02:  ## Table 2: ECL estimates with bootstrap CIs
	$(PYTHON) scripts/tab02_ecl_estimates.py

tab03:  ## Table 3: Context utilization ratios
	$(PYTHON) scripts/tab03_utilization.py

tab04:  ## Table 4: Perturbation sensitivity
	$(PYTHON) scripts/tab04_perturbation_sensitivity.py

tab05:  ## Table 5: Multi-scale block ECL
	$(PYTHON) scripts/tab05_multiscale_block.py

tab06:  ## Table 6: Pairwise model comparison
	$(PYTHON) scripts/tab06_pairwise_comparison.py

tab07:  ## Table 7: Biological validation
	$(PYTHON) scripts/tab07_biological_validation.py

figA1:  ## Appendix: CDS analysis
	$(PYTHON) scripts/figA1_cds_analysis.py

figA2:  ## Appendix: ECD analysis
	$(PYTHON) scripts/figA2_ecd_analysis.py

tabA1:  ## Appendix: Hyperparameter sensitivity
	$(PYTHON) scripts/tabA1_hyperparameter_sensitivity.py

# ---------- Aggregate targets ----------

all-experiments: figures tables appendix  ## Run ALL experiment scripts (main + appendix)
