@setup:
    uv sync
    uv run modal token new

@run:
    uv run modal run demo.py

@lint:
    uv run ruff check .

@format:
    uv run ruff format .
