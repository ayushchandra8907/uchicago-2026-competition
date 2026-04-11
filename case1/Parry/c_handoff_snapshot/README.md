# Market C Handoff Snapshot

This folder is a curated copy of the private `/Users/parrynall/UChicago-Trading_2`
workspace so Saavan can inspect the Market C work from the shared GitHub repo.

## Files

- `C.py`: Current standalone C-only bot architecture used as the base for the
  repo version in `case1/Parry/prediction_C.py`. It includes the newer C fair
  model, CPI/macro handling, CSV diagnostics, and earnings-shock lifecycle work.
- `prediction.py`: Older standalone C-only implementation. Treat this as
  historical context only; it is useful for understanding the earlier fair/rate
  model and simpler execution loop, but it is not the current source of truth.
- `logs/`: Curated run logs from the private workspace. These were selected
  because they show the main performance and behavior cases we discussed while
  building C.

## Important Notes

- The current repo bot remains `case1/Parry/prediction_C.py`.
- These source snapshots have been sanitized to use `UCHICAGO_USERNAME` and
  `UCHICAGO_PASSWORD` environment variables instead of hardcoded credentials.
- This is intentionally not a full folder dump: `.venv`, `__pycache__`, generated
  files, and most logs are excluded.
