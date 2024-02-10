.DEFAULT_GOAL := help
.PHONY: clean help

## Dependencies
### Install dependencies for MacOS (Metal device)
install-metal:
	CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python>=2.38.0 --no-cache-dir
#	CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DLLAMA_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python

## Poetry commands
### Export requirements.txt and requirements-dev.txt
poetry-export-requirements:
	@poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls --only main
	@poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --without-urls --only dev

### Create git tag and git push
poetry-version-tag:
	@git tag v$$(poetry version -s)
	@git push --tags

### autoflake detect (remove unused imports and variables)
poetry-autoflake:
	@poetry run autoflake --remove-unused-variables --remove-all-unused-imports --recursive --verbose ./chatnerds

### autoflake fix (remove unused imports and variables)
poetry-autoflake-fix:
	@poetry run autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive --verbose ./chatnerds

### black
poetry-black:
	@poetry run black ./chatnerds

### lint
poetry-lint:
	@poetry run pylint ./chatnerds

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
