# Contributing

Thanks for taking a look. This is a personal research/portfolio project, but
issues and pull requests are welcome — bug reports, reproducibility notes, and
small improvements especially.

## Ground rules

- **Config over constants.** Every tunable value (paths, model IDs, hyperparameters,
  UI strings, colors, filter cutoffs) lives in [`config.yaml`](config.yaml) and is
  loaded through the pydantic schema in [`utils/config.py`](utils/config.py). Code
  should read `cfg.section.key`, never a hard-coded literal. This is enforced by
  `tests/test_no_magic_strings.py`.
- **Tests must pass.** Run the suite before opening a PR:

  ```bash
  .venv/bin/python -m pytest -m "not integration"
  ```

  Tests marked `integration` need model weights / datasets that aren't in the repo;
  they're expected to skip locally when those artifacts are absent.
- **Keep files small and focused.** One responsibility per module; prefer splitting
  over growing a large file.
- **Match the surrounding style.** Type hints, clear names, and docstrings where the
  intent isn't obvious.

## Development setup

See the [Setup](README.md#setup) section of the README. In short: create the
Python 3.14 virtual environment, then run `./setup.sh`.

## Reporting bugs

Please include your OS, Python version, the command you ran, and the full traceback.
A minimal reproduction is worth a lot.
