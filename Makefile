help:
	@echo "Make what? Try 'make build', 'make build-strict', 'make serve', 'make clean' or 'make all'."

all: build

build:
	jupyter-book build .

build-strict:
	jupyter-book build --nitpick --warningiserror .

serve: build
	python tools/serve.py

clean:
	rm -rf _build/
