.PHONY: test smoke-public audit-public tables per-author chips-r-detail

PYTHON ?= python
TABLE_OUTPUT ?= experiments/reproduce/tables/table_rost_results.tex

test:
	$(PYTHON) -m unittest discover -s tests

smoke-public: test audit-public per-author chips-r-detail

audit-public:
	$(PYTHON) scripts/check_release_inventory.py

tables:
	$(PYTHON) scripts/generate_paper_tables.py \
		--metrics experiments/runs/rost_cv5_full/metrics.json \
		--output $(TABLE_OUTPUT)

per-author:
	$(PYTHON) scripts/rost_per_author.py

chips-r-detail:
	$(PYTHON) scripts/chips_r_detail.py
