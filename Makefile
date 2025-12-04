COMPOSE_FILE = deploy/docker-compose.local.yml
PROJECT_NAME = prism

.PHONY: help install up down ingest api clean

help:
	@echo "Usage: make [target]"
	@echo "  install   Install dependencies with uv"
	@echo "  up        Start local infrastructure (Kafka, Milvus, Triton)"
	@echo "  down      Stop infrastructure"
	@echo "  ingest    Run the video ingestion producer"
	@echo "  api       Start the search API"
	@echo "  clean     Remove cache and artifacts"

install:
	uv python install 3.12
	uv sync --all-extras

format:
	uv run ruff format .

up:
	docker-compose -f $(COMPOSE_FILE) up -d

down:
	docker-compose -f $(COMPOSE_FILE) down

ingest:
	uv run python -m src.prism.ingestion.producer

api:
	uv run uvicorn src.prism.api.app:app --reload

