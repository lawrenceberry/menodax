--8<-- "README.md:examples"

Each one runs on a GPU, compares against a Diffrax (and, where it is tractable,
a scipy) baseline built from the same right-hand side, and is checked without a
device by `tests/test_examples.py`.
