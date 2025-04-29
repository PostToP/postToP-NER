.PHONY: docs clean-docs

docs:
	$(MAKE) -C docs html

clean-docs:
	$(MAKE) -C docs clean