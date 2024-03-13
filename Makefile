.DEFAULT_GOAL := help
.PHONY: clean help

## Dependencies
### Install dependencies for MacOS (Metal device)
install-metal:
#	CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python 
	CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

### Install optional package whisper-mps
poetry-install-whisper-mps:
	poetry run pip install whisper-mps

## Poetry commands
### Export requirements.txt and requirements-dev.txt
poetry-export-requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls --only main
	@poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --without-urls --only dev

### Create git tag and git push
poetry-version-tag:
	@git tag v$$(poetry version -s)
	@git push --tags

### lint
poetry-lint:
	@poetry run pylint ./chatnerds

### autoflake detect (remove unused imports and variables)
poetry-autoflake:
	@poetry run autoflake --remove-unused-variables --remove-all-unused-imports --recursive --verbose ./chatnerds

### autoflake fix (remove unused imports and variables)
poetry-autoflake-fix:
	@poetry run autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive --verbose ./chatnerds

### black (prettifier)
poetry-black:
	@poetry run black ./chatnerds

poetry-fix: poetry-autoflake-fix poetry-black

poetry-importtime:
	@poetry run python -X importtime ./chatnerds/cli/cli.py 2> importtime_cli.log
	@poetry run tuna --sort-by cumtime --reverse importtime_cli.log

### Detect and show dependencies
poetry-deptry:
	@poetry run deptry .

# show help: Renders automatically categories (##) and targets (###). Regular comments (#) ignored
# Based on: https://gist.github.com/prwhite/8168133?permalink_comment_id=2278355#gistcomment-2278355
TARGET_COLOR := $(shell tput -Txterm setaf 6)
BOLD := $(shell tput -Txterm bold)
RESET := $(shell tput -Txterm sgr0)
help:
	@echo ''
	@echo 'Usage:'
	@echo '  make ${TARGET_COLOR}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^### (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 4, RLENGTH); \
      printf "  ${TARGET_COLOR}%-20s${RESET} %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
