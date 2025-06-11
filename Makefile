all: build

build:
	jupyter-book build .

build-strict:
	jupyter-book build --nitpick --warningiserror .

serve: build
	python tools/serve.py

clean:
	rm -rf _build/
