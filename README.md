# IITM-MLOps-Course
This repository contains any code/documents that is relevant for the graded assignments for the MLOps course.
Quick test & CI notes
---------------------

To run the week-2 tests locally you need the DVC-tracked data. From the repository root:

```bash
cd week-2
dvc pull
pytest -q tests
```

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on `ubuntu-latest`. The workflow installs DVC, pulls data (`dvc pull`) and runs the pytest suite in `week-2`. The workflow is fail-fast (`pytest -x`).
