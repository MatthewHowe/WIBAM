PYTHON = python3
PIP = pip3

.DEFAULT_GOAL = run

build:
	@bash scripts/build_docker.bash

run:
	@bash scripts/run.bash $(filter-out $@, $(MAKECMDGOALS))

stop:
	@docker container kill $$(docker ps -q)

lint:
	@bash scripts/lint.bash
	@echo "✅✅✅✅✅ Lint is good! ✅✅✅✅✅"