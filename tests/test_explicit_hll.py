#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from python_hll.hlltype import HLLType
from python_hll.hll import HLL
from python_hll.serialization import SerializationUtil

"""Unit tests for BitVector."""


def test_add_basic():
    """
    Tests basic set semantics of ``HLL.add_raw()``.
    """
    # Adding a single positive value to an empty set should work.
    hll = new_hll(128)  # arbitrary
    hll.add_raw(1)  # positive
    assert hll.cardinality() == 1

    # Adding a single negative value to an empty set should work.
    hll = new_hll(128)  # arbitrary
    hll.add_raw(-1)  # negative
    assert hll.cardinality() == 1

    # Adding a duplicate value to a set should be a no-op.
    hll = new_hll(128)  # arbitrary
    hll.add_raw(1)  # positive
    hll.add_raw(1)  # dupe
    assert hll.cardinality() == 1


def test_union():
    """
    Tests ``HLL.union()``.
    """
    # Unioning two distinct sets should work
    hll_a = new_hll(128)  # arbitrary
    hll_b = new_hll(128)  # arbitrary
    hll_a.add_raw(1)
    hll_a.add_raw(2)
    hll_b.add_raw(3)

    hll_a.union(hll_b)
    assert hll_a.cardinality() == 3

    # Unioning two sets whose union doesn't exceed the cardinality cap should not promote
    hll_a = new_hll(128)  # arbitrary
    hll_b = new_hll(128)  # arbitrary
    hll_a.add_raw(1)
    hll_a.add_raw(2)
    hll_b.add_raw(1)

    hll_a.union(hll_b)
    assert hll_a.cardinality() == 2
    assert hll_a.get_type() == HLLType.EXPLICIT

    # Unioning two sets whose union exceeds the cardinality cap should promote
    hll_a = new_hll(128)  # arbitrary
    hll_b = new_hll(128)  # arbitrary
    for i in range(0, 128):
        hll_a.add_raw(i)
        hll_b.add_raw(i+128)

    hll_a.union(hll_b)
    assert hll_a.get_type() == HLLType.SPARSE


def test_clear():
    """
    Tests ``HLL.clear()``
    """
    hll = new_hll(128)  # arbitrary
    hll.add_raw(1)
    hll.clear()
    assert hll.cardinality() == 0


def test_to_from_bytes():
    """
    Tests ``HLL.to_bytes() and ``HLL.from_bytes().
    """
    schema_version = SerializationUtil.DEFAULT_SCHEMA_VERSION
    type = HLLType.EXPLICIT
    padding = schema_version.padding_bytes(type)
    bytes_per_word = 8

    # Should work on an empty set
    hll = new_hll(128)
    bytes = hll.to_bytes(schema_version)
    assert len(bytes) == padding  # no elements, just padding

    in_hll = HLL.from_bytes(bytes)
    assert_elements_equal(hll, in_hll)

    # Should work on a partially filled set
    hll = new_hll(128)
    for i in range(0, 3):
        hll.add_raw(i)

    bytes = hll.to_bytes(schema_version)
    assert len(bytes) == padding + bytes_per_word * 3

    in_hll = HLL.from_bytes(bytes)
    assert_elements_equal(hll, in_hll)

    # Should work on a full set
    explicit_threshold = 128
    hll = new_hll(explicit_threshold)

    for i in range(0, explicit_threshold):
        hll.add_raw(27 + i)

    bytes = hll.to_bytes(schema_version)
    assert len(bytes) == padding + bytes_per_word * explicit_threshold

    in_hll = HLL.from_bytes(bytes)
    assert_elements_equal(hll, in_hll)


def test_random_values():
    """
    Tests correctness against `set()`.
    """
    explicit_threshold = 4096
    canonical = set()
    hll = new_hll(explicit_threshold)

    seed = 1  # constant so results are reproducible
    random.seed(seed)
    max_java_long = 9223372036854775807
    for i in range(0, explicit_threshold):
        random_long = random.randint(1, max_java_long)
        canonical.add(random_long)
        hll.add_raw(random_long)
    canonical_cardinality = len(canonical)
    assert hll.cardinality() == canonical_cardinality


def test_promotion():
    """
    Tests promotion to ``HLLType.SPARSE`` and ``HLLType.FULL``.
    """
    explicit_threshold = 128
    hll = HLL.create_for_testing(11, 5, explicit_threshold, 256, HLLType.EXPLICIT)
    for i in range(0, explicit_threshold + 1):
        hll.add_raw(i)
    assert hll.get_type() == HLLType.SPARSE

    hll = HLL(11, 5, 4, False, HLLType.EXPLICIT)  # expthresh=4 => explicit_threshold=8
    for i in range(0, 9):
        hll.add_raw(i)
    assert hll.get_type() == HLLType.FULL


# ------------------------------------------------------------
# assertion helpers


def assert_elements_equal(hll_a, hll_b):
    """
    Asserts that values in both sets are exactly equal.
    """
    assert hll_a._explicit_storage == hll_b._explicit_storage


def new_hll(explicit_threshold):
    """
    Builds a ``HLLType.EXPLICIT`` ``HLL`` instance with the specified
    explicit threshold.

    :param explicit_threshold: explicit threshold to use for the constructed
           ``HLL``. This must be greater than zero.
    :type explicit_threshold: int
    :returns: A default-sized ``HLLType.EXPLICIT`` empty ``HLL`` instance. This
              will never be ``None``.
    :rtype: HLL
    """
    return HLL.create_for_testing(11, 5, explicit_threshold, 256, HLLType.EXPLICIT)
