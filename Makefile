CC := gcc
CFLAGS := -O3 -ffast-math -march=native -funroll-loops -fomit-frame-pointer
LDFLAGS := -lm
SHELL := /bin/bash
UV := uv
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python

SRC := src/main.c
BIN := artifacts/sarvam1
BENCH_SRC := src/benchmark.c
BENCH_BIN := artifacts/sarvam1_benchmark
WEIGHTS := artifacts/sarvam1.bin
TOKENIZER := artifacts/tokenizer.bin
TOKENIZER_MODEL := artifacts/tokenizer.model

TOKENIZER_SCRIPT := scripts/tokenizer.py
WEIGHTS_SCRIPT := scripts/weights.py

EXTRA_GOALS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
PROMPT ?=
TOKENS ?= 50

.PHONY: all check-uv venv install export build benchmark-build run benchmark clean clean-artifacts

all: build

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv is not installed. install it from https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

venv: check-uv
	@if [ ! -x "$(PYTHON)" ]; then \
		$(UV) venv $(VENV_DIR); \
	fi

install: venv
	$(UV) pip install --python $(PYTHON) torch --index-url https://download.pytorch.org/whl/cpu
	$(UV) pip install --python $(PYTHON) transformers accelerate sentencepiece numpy

clean-artifacts:
	rm -rf artifacts

artifacts:
	mkdir -p artifacts

export: install clean-artifacts artifacts
	$(UV) run --python $(PYTHON) $(TOKENIZER_SCRIPT)
	$(UV) run --python $(PYTHON) $(WEIGHTS_SCRIPT)

build: $(BIN)

$(BIN): $(SRC)
	mkdir -p artifacts
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

benchmark-build: $(BENCH_BIN)

$(BENCH_BIN): $(BENCH_SRC) $(SRC)
	mkdir -p artifacts
	$(CC) $(CFLAGS) -DSARVAM_LIB_ONLY -o $(BENCH_BIN) $(BENCH_SRC) $(SRC) $(LDFLAGS)

run: build
	@args='$(EXTRA_GOALS)'; \
	prompt='$(PROMPT)'; \
	tokens='$(TOKENS)'; \
	if [ ! -f "$(WEIGHTS)" ] || [ ! -f "$(TOKENIZER)" ]; then \
		echo "missing artifacts. run: make export"; \
		exit 1; \
	fi; \
	if [ -n "$$args" ]; then \
		set -- $$args; \
		last="$${!#}"; \
		if [[ "$$last" =~ ^[0-9]+$$ ]]; then \
			tokens="$$last"; \
			if [ "$$#" -gt 1 ]; then \
				prompt="$${*:1:$$(($$#-1))}"; \
			else \
				prompt=""; \
			fi; \
		else \
			prompt="$$*"; \
		fi; \
	fi; \
	if [ -z "$$prompt" ]; then \
		echo "usage: make run \"your prompt\" <number of tokens>"; \
		echo "or: make run PROMPT=\"your prompt\" TOKENS=50"; \
		exit 1; \
	fi; \
	$(BIN) $(WEIGHTS) $(TOKENIZER) "$$prompt" "$$tokens"

benchmark: benchmark-build
	@args='$(EXTRA_GOALS)'; \
	prompt='$(PROMPT)'; \
	tokens='$(TOKENS)'; \
	if [ ! -f "$(WEIGHTS)" ] || [ ! -f "$(TOKENIZER)" ]; then \
		echo "missing artifacts. run: make export"; \
		exit 1; \
	fi; \
	if [ -n "$$args" ]; then \
		set -- $$args; \
		last="$${!#}"; \
		if [[ "$$last" =~ ^[0-9]+$$ ]]; then \
			tokens="$$last"; \
			if [ "$$#" -gt 1 ]; then \
				prompt="$${*:1:$$(($$#-1))}"; \
			else \
				prompt=""; \
			fi; \
		else \
			prompt="$$*"; \
		fi; \
	fi; \
	if [ -z "$$prompt" ]; then \
		echo "usage: make benchmark \"your prompt\" <number of tokens>"; \
		echo "or: make benchmark PROMPT=\"your prompt\" TOKENS=50"; \
		exit 1; \
	fi; \
	$(BENCH_BIN) $(WEIGHTS) $(TOKENIZER) "$$prompt" "$$tokens"

clean:
	rm -f $(BIN) $(BENCH_BIN)

%:
	@:
