@setup:
    uv sync
    uv run modal token new

@run:
    uv run modal run pipeline.py

@validate:
    uv run modal run validate.py

@test:
    uv run python -m unittest discover -s tests -v

@lint:
    uvx ruff==0.12.4 check .

@format:
    uvx ruff==0.12.4 format .
