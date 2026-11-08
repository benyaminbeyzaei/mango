Use this directory for investigation-specific code if useful. You are not required
to place all modifications here; substantive changes may naturally involve src/.

The experiment platform is the `mlsrag` package in `platform/` (see `platform/README.md`;
hand-over notes in `platform/HANDOVER.md`). Run results it writes go to `results/`, one folder per run.

**Evaluators:** `web/src/content/evaluator-guide.md` (also the web client's Guide page) shows how to
reproduce the baseline numbers with the course scripts on the cluster, and how to run our matrices
on a full split with `mlsrag matrix ... --fresh --allow-full --yes`, with the estimated cost.
