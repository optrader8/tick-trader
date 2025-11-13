# Repository Guidelines

## Project Structure & Module Organization
Tick Trader ships as an editable Python package rooted in `src/`. Market ingestion and schema helpers live in `src/data/`, feature calculators in `src/features/`, neural/ensemble models in `src/models/`, and the orchestration entry point is `src/training/trainer.py`. `src/prediction/realtime.py` handles live inference, `src/storage/{parquet,redis_cache}.py` handles persistence, and `src/evaluation/{metrics,backtest}.py` captures KPIs. Central settings sit in `config/config.yaml`; raw/processed/features/models artifacts belong under `data/` per that file. Keep docs in `README.md` and mirror every module in `tests/` (`tests/test_storage`, `tests/test_models`, etc.) for parity.

## Build, Test, and Development Commands
- `python -m venv .venv && .\.venv\Scripts\activate`: provision a local Python ≥3.8 runtime aligned with `setup.py`.
- `pip install -r requirements.txt && pip install -e .`: install dependencies plus the package in editable mode for `import training.trainer`.
- `pytest -q`: run the suite; focus with `pytest tests/test_models -k transformer` when iterating on a module.
- `python -m coverage run -m pytest && coverage html`: optional coverage report before reviews.
- Prototype training flows in a REPL/notebook by loading configs via `yaml.safe_load(open("config/config.yaml"))` and instantiating `ModelTrainer`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, snake_case modules/functions, and CapWords classes as seen in `src/models/lstm.py`. Favor type hints (`typing.Dict`, `np.ndarray`) and module-level loggers (`logging.getLogger(__name__)`). Keep docstrings descriptive (triple-double quotes) and group imports stdlib → third-party → local. Persist notebooks in `data/notebooks/` and gate large artifacts via `.gitignore`. Commit JSON/YAML configs only when documenting all new keys.

## Testing Guidelines
Author pytest tests beside their domains (e.g., new storage adapter ⇒ `tests/test_storage/test_new_adapter.py`). Name tests `test_<behavior>` and use fixtures/tmp_path for Redis/Parquet adapters. Target ≥80 % statement coverage for new modules and assert both success paths and failure modes (`ModelTrainingError`, config validation). Always run `pytest` (or filtered subsets) before opening a PR and paste the command/output in the PR body.

## Commit & Pull Request Guidelines
Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`) as shown in `git log`. Keep scope focused and mention the subsystem (`feat(models): add transformer dropout`). PRs should summarize intent, link issues, reference config knobs touched, and attach evidence (test logs, backtest snippets, screenshots of metric deltas) so reviewers can replay results quickly.

## Security & Configuration Tips
Never commit credentials, API keys, or proprietary ticks; point secrets to environment variables consumed by the loaders. Update `config/config.yaml` instead of scattering literals, and document any new keys inside the PR. When using Redis or Parquet sinks, prefer local ports and describe required firewall/proxy changes in a `Security` subsection of the PR.
