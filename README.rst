==========
python-hll
==========


.. image:: https://img.shields.io/pypi/v/python_hll.svg
        :target: https://pypi.python.org/pypi/python_hll

.. image:: https://img.shields.io/travis/JonathanAquino/python_hll.svg
        :target: https://travis-ci.org/JonathanAquino/python_hll

.. image:: https://readthedocs.org/projects/python-hll/badge/?version=latest
        :target: https://python-hll.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/JonathanAquino/python_hll/shield.svg
     :target: https://pyup.io/repos/github/JonathanAquino/python_hll/
     :alt: Updates

A Python implementation of `HyperLogLog <http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>`_
whose goal is to be `storage compatible <https://github.com/aggregateknowledge/hll-storage-spec>`_
with `java-hll <https://github.com/aggregateknowledge/java-hll>`_, `js-hll <https://github.com/aggregateknowledge/js-hll>`_
and `postgresql-hll <https://github.com/citusdata/postgresql-hll>`_.

**NOTE:** This is a fairly literal translation/port of `java-hll <https://github.com/aggregateknowledge/java-hll>`_
to Python. Internally, bytes are represented as Java-style bytes (-128 to 127) rather than Python-style bytes (0 to 255).
Also this implementation is quite slow: for example, in Java ``HLLSerializationTest`` takes 12 seconds to run
while in Python ``test_hll_serialization`` takes 1.5 hours to run (about 400x slower).

* Runs on: Python 2.7 and 3
* Free software: MIT license
* Documentation: https://python-hll.readthedocs.io.


Getting started
---------------
::

    $ mkvirtualenv python_hll
    $ python setup.py develop
    $ pip install -r requirements_dev.txt

Run tests::

    $ make lint
    $ make test-fast

To run one test file or one test::

    $ py.test --capture=no tests/test_sparse_hll.py
    $ py.test --capture=no tests/test_sparse_hll.py::test_add

To run slow tests::

    $ make test
