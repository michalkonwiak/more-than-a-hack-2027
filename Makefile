PYTHON := .venv/bin/python
DATA_DIR ?= ../More Than A Hack 2027 - Initial Challenge - Dataset

.PHONY: install check reference candidate reproduce

install:
	uv venv --python 3.12
	uv pip sync requirements.lock --require-hashes

check:
	.venv/bin/ruff check possession tests
	.venv/bin/ruff format --check possession tests
	$(PYTHON) -m pytest -q

reference:
	$(PYTHON) -m possession.reference --data-dir "$(DATA_DIR)" --verify

candidate:
	$(MAKE) check
	OPENBLAS_NUM_THREADS=1 $(PYTHON) -m possession.external_infer --data-dir "$(DATA_DIR)" --verify
	$(PYTHON) evaluation/validate_submission.py artifacts/candidates/external_idsse_v1/submission.csv

reproduce:
	$(MAKE) check
	$(MAKE) reference
	$(PYTHON) evaluation/validate_submission.py artifacts/reference_baseline/submission.csv
