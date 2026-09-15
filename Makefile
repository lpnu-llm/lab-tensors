.PHONY: test watch notebook

test:
	uvx pytest matmul.py

watch:
	uvx --with pytest pytest-watcher --now --clear --ff . -- matmul.py

notebook:
	uvx jupytext --to notebook matmul.py
