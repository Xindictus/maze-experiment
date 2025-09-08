## Local Setup

1. Install `uv` dependency manager. Follow installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

2. Install `Python` dependencies:

   ```shell
   uv sync
   ```

3. Enable `venv`:

   ```shell
   . .venv/bin/activate
   ```

4. Run an experiment

   ```shell
   python ...
   ```

## Code maintainance

1. Code is formatted using `black`. To format the codebase run on root level:

    ```shell
    black .
    ```

2. The default linter used is `ruff`. To run checks against the codebase run:

    ```shell
    ruff check .
    ```

   To auto-fix the issues run:

    ```shell
    ruff check . --fix
    ```

## Instructions

TODO
