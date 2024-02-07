#!/bin/bash

VENV_PATH=$(pipenv --venv)

# Check local repo exists in the parent directory
if [[ ! -d "$VENV_PATH" ]]; then
  echo "\nerror: venv path does not exist: $VENV_PATH"
  exit 1
fi

# confirm [<prompt>]
# source: https://mike632t.wordpress.com/2017/07/06/bash-yes-no-prompt/
confirm() {
  local _prompt _default _response
 
  if [ "$1" ]; then _prompt="$1"; else _prompt="Are you sure"; fi
  _prompt="$_prompt [y/n] ?"
 
  # Loop forever until the user enters a valid response (Y/N or Yes/No).
  while true; do
    read -r -p "$_prompt " _response
    case "$_response" in
      [Yy][Ee][Ss]|[Yy]) # Yes or Y (case-insensitive).
        return 0
        ;;
      [Nn][Oo]|[Nn])  # No or N.
        # exit
        return 1
        ;;
      *) # Anything else (including a blank) is invalid.
        ;;
    esac
  done
}

if ! confirm "You are going to delete the environment '${VENV_PATH}' and recreate it. Do you want to continue"; then exit; fi

# Delete the virtual environment and recreate it
# pipenv --rm && \
rm -rf $VENV_PATH && \
pipenv shell && \
pipenv install --dev

# pyenv shell 3.10.12 && \
# pipenv shell --python 3.10.12 
