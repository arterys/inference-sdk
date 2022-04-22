.PHONY: wheel

wheel:
	@python setup.py bdist_wheel
	@mv dist/*.whl .
	@rm -r build dist
	@cp *.whl arterys_sdk/inference_test_tool/packages/