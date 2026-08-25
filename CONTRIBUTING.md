# Contributing

Contributions are welcome through GitHub issues and pull requests.

1. Create the development environment with `micromamba create -f environment.yml`.
2. Activate it with `micromamba activate utherm-release`.
3. Install the checkout with `python -m pip install --no-deps -e .`.
4. Run `python -B -m study_code.preflight .` and `pytest -q` before opening a pull request.

Please add tests for behavioral changes and update the scientific documentation
when changing equations, assumptions, defaults, or coupling behavior. Do not
commit input datasets, generated simulation outputs, credentials, machine-specific
paths, or internal diagnostic reports.
