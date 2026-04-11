# Curated Market C Logs

These logs were copied from `/Users/parrynall/UChicago-Trading_2/logs` as a small
reference set for Market C debugging and future integration work.

- `c_only_run_20260410_112113.csv`: Early poor run where C ignored important CPI
  and C earnings events.
- `c_only_run_20260410_113917.csv`: Follow-up poor run used to identify missing
  CPI/C earnings handling and repeated blocked entries.
- `c_only_run_20260410_122214.csv`: Strong run, roughly the first major 45k PnL
  example used as the success baseline.
- `c_only_run_20260410_125247.csv`: Weaker follow-up run used to compare against
  the 45k baseline and identify regressions.
- `c_only_run_20260410_140905.csv`: Another strong roughly 44k run; useful for
  studying position-hold timing and short-side behavior.
- `c_only_run_20260410_183404.csv`: Later live run showing earnings-shock
  shorting behavior, profit taking, max-hold exits, and remaining hold-time
  concerns.
- `c_only_run_20260410_200812.csv`: Newest large C-only log from the private
  workspace, useful for inspecting the latest standalone behavior.
