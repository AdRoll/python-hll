#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from math import ceil, log
import random
from python_hll.hlltype import HLLType
from python_hll.hll import HLL
from python_hll.hllutil import HLLUtil
from python_hll.serialization import SerializationUtil
from python_hll.util import BitUtil
import probabilistic_test_util

"""Tests ``HLL`` of type ``HLLType.SPARSE``."""

log2m = 11


def test_add():
    """
    Tests ``HLL.add_raw()``.
    """
    # ------------------------------------------------------------
    # insert an element with register value 1 (minimum set value)
    register_index = 0
    register_value = 1
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(register_value))

    # ------------------------------------------------------------
    # insert an element with register value 31 (maximum set value)
    register_index = 0
    register_value = 31
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(register_value))

    # ------------------------------------------------------------
    # insert an element that could overflow the register (past 31)
    register_index = 0
    register_value = 36
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(31))  # register max

    # ------------------------------------------------------------
    # insert duplicate elements, observe no change
    register_index = 0
    register_value = 1
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)
    hll.add_raw(raw_value)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(register_value))  # register max

    # ------------------------------------------------------------
    # insert elements that increase a register's value
    register_index = 0
    register_value = 1
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)

    register_value_2 = 2
    raw_value_2 = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value_2)
    hll.add_raw(raw_value_2)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(register_value_2))

    # ------------------------------------------------------------
    # insert elements that have lower register values, observe no change
    register_index = 0
    register_value = 2
    raw_value = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value)

    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(raw_value)

    register_value_2 = 1
    raw_value_2 = probabilistic_test_util.construct_hll_value(log2m, register_index, register_value_2)
    hll.add_raw(raw_value_2)

    assert_one_register_set(hll, register_index, BitUtil.to_signed_byte(register_value))


def test_small_range_smoke():
    """
    Smoke test for ``HLL.cardinality()`` and the proper use of the small
    range correction.
    """
    log2m = 11
    m = BitUtil.left_shift_int(1, log2m)
    regwidth = 5

    # ------------------------------------------------------------
    # only one register set
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.SPARSE)
    hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, 0, 1))

    cardinality = hll.cardinality()

    # Trivially true that small correction conditions hold: one register
    # set implies zeroes exist, and estimator trivially smaller than 5m/2.
    # Small range correction: m * log(m/V)
    expected = ceil(m * log(m / (m - 1)))  # # of zeroes
    assert cardinality == expected

    # ------------------------------------------------------------
    # all but one register set
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.SPARSE)
    for i in range(0, m - 1):
        hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, 1))

    # Trivially true that small correction conditions hold: all but
    # one register set implies a zero exists, and estimator trivially
    # smaller than 5m/2 since it's alpha / ((m-1)/2)
    cardinality = hll.cardinality()

    # Small range correction: m * log(m/V)
    expected = ceil(m * log(m / 1))  # # of zeroes
    assert cardinality == expected


def test_normal_range_smoke():
    """
    Smoke test for HLL.cardinality() and the proper use of the
    uncorrected estimator.
    """
    log2m = 11
    m = BitUtil.left_shift_int(1, log2m)
    regwidth = 5
    # regwidth = 5, so hash space is
    # log2m + (2^5 - 1 - 1), so L = log2m + 30
    L = log2m + 30

    # all registers at 'medium' value
    hll = HLL.create_for_testing(log2m, regwidth, 128, m, HLLType.SPARSE)

    register_value = 7  # chosen to ensure neither correction kicks in
    for i in range(0, m):
        hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, register_value))

    cardinality = hll.cardinality()

    # Simplified estimator when all registers take same value: alpha / (m/2^val)
    estimator = HLLUtil.alpha_m_squared(m) / (m / (2 ** register_value))

    # Assert conditions for uncorrected range
    assert estimator <= (2 ** L) / 30
    assert estimator > (5 * m / 2)

    expected = ceil(estimator)
    assert cardinality == expected


def test_large_range_smoke():
    """
    Smoke test for ``HLL.cardinality()`` and the proper use of the large
    range correction.
    """
    log2m = 11
    m = BitUtil.left_shift_int(1, log2m)
    regwidth = 5
    # regwidth = 5, so hash space is
    # log2m + (2^5 - 1 - 1), so L = log2m + 30
    L = log2m + 30

    # all registers at large value
    hll = HLL.create_for_testing(log2m, regwidth, 128, m, HLLType.SPARSE)

    register_value = 31  # chosen to ensure large correction kicks in
    for i in range(0, m):
        hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, register_value))

    cardinality = hll.cardinality()

    # Simplified estimator when all registers take same value: alpha / (m/2^val)
    estimator = HLLUtil.alpha_m_squared(m) / (m / (2 ** register_value))

    # Assert conditions for large range
    assert estimator > (2**L) / 30

    # Large range correction: -2^32 * log(1 - E/2^32)
    try:
        expected = ceil(-1.0 * (2**L) * log(1.0 - estimator / (2**L)))
    except ValueError:
        expected = 0
    assert cardinality == expected


def test_union():
    """
    Tests ``HLL.union()``.
    """
    log2m = 11  # arbitrary
    sparse_threshold = 256  # arbitrary

    # ------------------------------------------------------------
    # two empty multisets should union to an empty set
    hll_a = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_b = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)

    hll_a.union(hll_b)

    assert hll_a.get_type() == HLLType.SPARSE
    assert hll_a.cardinality() == 0

    # ------------------------------------------------------------
    # two disjoint multisets should union properly
    hll_a = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_a.add_raw(probabilistic_test_util.construct_hll_value(log2m, 1, 1))
    hll_b = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_a.add_raw(probabilistic_test_util.construct_hll_value(log2m, 2, 1))

    hll_a.union(hll_b)

    assert hll_a.get_type() == HLLType.SPARSE  # unchanged
    assert hll_a.cardinality() == 3  # precomputed
    assert_register_present(hll_a, 1, BitUtil.to_signed_byte(1))
    assert_register_present(hll_a, 2, BitUtil.to_signed_byte(1))

    # ------------------------------------------------------------
    # two exactly overlapping multisets should union properly
    hll_a = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_a.add_raw(probabilistic_test_util.construct_hll_value(log2m, 1, 10))
    hll_b = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_a.add_raw(probabilistic_test_util.construct_hll_value(log2m, 1, 13))

    hll_a.union(hll_b)

    assert hll_a.get_type() == HLLType.SPARSE  # unchanged
    assert hll_a.cardinality() == 2  # precomputed
    assert_one_register_set(hll_a, 1, BitUtil.to_signed_byte(13))  # max(10,13)

    # ------------------------------------------------------------
    # overlapping multisets should union properly
    hll_a = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_b = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    # register index = 3
    raw_value_a = probabilistic_test_util.construct_hll_value(log2m, 3, 11)

    # register index = 4
    raw_value_b = probabilistic_test_util.construct_hll_value(log2m, 4, 13)
    raw_value_b_prime = probabilistic_test_util.construct_hll_value(log2m, 4, 21)

    # register index = 5
    raw_value_c = probabilistic_test_util.construct_hll_value(log2m, 5, 14)

    hll_a.add_raw(raw_value_a)
    hll_a.add_raw(raw_value_b)

    hll_b.add_raw(raw_value_b_prime)
    hll_b.add_raw(raw_value_c)

    hll_a.union(hll_b)
    # union should have three registers set, with partition B set to the
    # max of the two registers
    assert_register_present(hll_a, 3, BitUtil.to_signed_byte(11))
    assert_register_present(hll_a, 4, BitUtil.to_signed_byte(21))  # max(21,13)
    assert_register_present(hll_a, 5, BitUtil.to_signed_byte(14))

    # ------------------------------------------------------------
    # too-large unions should promote
    hll_a = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)
    hll_b = HLL.create_for_testing(log2m, 5, 128, sparse_threshold, HLLType.SPARSE)

    # fill up sets to maxCapacity
    for i in range(0, sparse_threshold):
        hll_a.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, 1))
        hll_b.add_raw(probabilistic_test_util.construct_hll_value(log2m, i + sparse_threshold, 1))  # non-overlapping

    hll_a.union(hll_b)

    assert hll_a.get_type() == HLLType.FULL


def test_clear():
    """
    Tests ``HLL.clear()``.
    """
    hll = HLL.create_for_testing(log2m, 5, 128, 256, HLLType.SPARSE)
    hll.add_raw(1)
    hll.clear()
    assert hll.cardinality() == 0


def test_to_from_bytes():
    """
    Tests ``HLL.to_bytes()`` and ``HLL.from_bytes()``.
    """
    log2m = 11  # arbitrary
    regwidth = 5  # arbitrary
    sparse_threshold = 256  # arbitrary
    short_word_length = 16  # log2m + regwidth = 11 + 5

    schema_version = SerializationUtil.DEFAULT_SCHEMA_VERSION
    type = HLLType.SPARSE
    padding = schema_version.padding_bytes(type)

    # ------------------------------------------------------------
    # Should work on an empty element
    hll = HLL.create_for_testing(log2m, regwidth, 128, sparse_threshold, HLLType.SPARSE)
    bytes = hll.to_bytes(schema_version)

    # output should just be padding since no registers are used
    assert len(bytes) == padding

    in_hll = HLL.from_bytes(bytes)

    # assert register values correct
    assert_elements_equal(hll, in_hll)

    # ------------------------------------------------------------
    # Should work on a partially filled element
    hll = HLL.create_for_testing(log2m, regwidth, 128, sparse_threshold, HLLType.SPARSE)

    for i in range(0, 3):
        raw_value = probabilistic_test_util.construct_hll_value(log2m, i, (i + 9))
        hll.add_raw(raw_value)

    bytes = hll.to_bytes(schema_version)

    assert len(bytes) == padding + probabilistic_test_util.get_required_bytes(short_word_length, 3)  # register_count

    in_hll = HLL.from_bytes(bytes)

    # assert register values correct
    assert_elements_equal(hll, in_hll)

    # ------------------------------------------------------------
    # Should work on a full set
    hll = HLL.create_for_testing(log2m, regwidth, 128, sparse_threshold, HLLType.SPARSE)

    for i in range(0, sparse_threshold):
        raw_value = probabilistic_test_util.construct_hll_value(log2m, i, (i % 9) + 1)
        hll.add_raw(raw_value)

    bytes = hll.to_bytes(schema_version)

    # 'short words' should be 12 bits + 5 bits = 17 bits long
    assert len(bytes) == padding + probabilistic_test_util.get_required_bytes(short_word_length, sparse_threshold)

    in_hll = HLL.from_bytes(bytes)

    # assert register values correct
    assert_elements_equal(hll, in_hll)


def test_random_values():
    log2m = 11  # arbitrary
    regwidth = 5  # arbitrary
    sparse_threshold = 256  # arbitrary

    seed = 1
    random.seed(seed)
    max_java_long = 9223372036854775807

    for run in range(0, 100):
        hll = HLL.create_for_testing(log2m, regwidth, 128, sparse_threshold, HLLType.SPARSE)

        map = {}

        for i in range(0, sparse_threshold):
            raw_value = random.randint(1, max_java_long)

            register_index = probabilistic_test_util.get_register_index(raw_value, log2m)
            register_value = probabilistic_test_util.get_register_value(raw_value, log2m)
            if map.get(register_index, 0) < register_value:
                map[register_index] = register_value

            hll.add_raw(raw_value)

        for key in map.keys():
            expected_register_value = map.get(key, 0)
            assert_register_present(hll, key, expected_register_value)

# ------------------------------------------------------------
# assertion helpers


def assert_register_present(hll, register_index, register_value):
    """
    Asserts that the register at the specified index is set to the specified value.
    """
    sparse_probabilistic_storage = hll._sparse_probabilistic_storage
    assert sparse_probabilistic_storage.get(register_index, 0) == register_value


def assert_one_register_set(hll, register_index, register_value):
    """
    Asserts that only the specified register is set and has the specified value.
    """
    sparse_probabilistic_storage = hll._sparse_probabilistic_storage
    assert len(sparse_probabilistic_storage) == 1
    assert sparse_probabilistic_storage.get(register_index, 0) == register_value


def assert_elements_equal(hll_a, hll_b):
    sparse_probabilistic_storage_a = hll_a._sparse_probabilistic_storage
    sparse_probabilistic_storage_b = hll_b._sparse_probabilistic_storage
    assert len(sparse_probabilistic_storage_a) == len(sparse_probabilistic_storage_b)
    for index in sparse_probabilistic_storage_a.keys():
        assert sparse_probabilistic_storage_a.get(index) == sparse_probabilistic_storage_b.get(index)
