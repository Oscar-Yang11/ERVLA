SHELL=/bin/bash

format:
	pre-commit run --all-files

.PHONY: install format
