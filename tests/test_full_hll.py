#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pytest
from math import ceil, log
from python_hll.hlltype import HLLType
from python_hll.hll import HLL
from python_hll.hllutil import HLLUtil
from python_hll.serialization import SerializationUtil
from python_hll.util import BitUtil
import probabilistic_test_util

"""Tests ``HLL`` of type ``HLLType.FULL``."""


def test_small_range_smoke():
    """
    Smoke test for HLL.cardinality() and the proper use of the
    small range correction.
    """
    log2m = 11
    m = BitUtil.left_shift_int(1, log2m)
    regwidth = 5

    # only one register set
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
    hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, 0, 1))
    cardinality = hll.cardinality()

    # Trivially true that small correction conditions hold: one register
    # set implies zeroes exist, and estimator trivially smaller than 5m/2.
    # Small range correction: m * log(m/V)
    expected = ceil(m * log(m / (m - 1)))  # # of zeroes
    assert cardinality == expected

    # all but one register set
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
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
    Smoke test for ``HLL.cardinality()`` and the proper use of the
    uncorrected estimator.
    """
    log2m = 11
    regwidth = 5

    # regwidth = 5, so hash space is
    # log2m + (2^5 - 1 - 1), so L = log2m + 30
    L = log2m + 30
    m = BitUtil.left_shift_int(1, log2m)
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)

    # all registers at 'medium' value
    register_value = 7  # chosen to ensure neither correction kicks in
    for i in range(0, m):
        hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, register_value))

    cardinality = hll.cardinality()

    # Simplified estimator when all registers take same value: alpha / (m/2^val)
    estimator = HLLUtil.alpha_m_squared(m) / (m / (2**register_value))

    assert estimator <= (2**L)/30
    assert estimator > (5 * m / 2)

    expected = ceil(estimator)
    assert cardinality == expected


def test_large_range_smoke():
    """
    Smoke test for ``HLL.cardinality()`` and the proper use of the large
    range correction.
    """
    log2m = 12
    regwidth = 5
    # regwidth = 5, so hash space is
    # log2m + (2^5 - 1 - 1), so L = log2m + 30
    L = log2m + 30
    m = BitUtil.left_shift_int(1, log2m)
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)

    register_value = 31  # chosen to ensure large correction kicks in
    for i in range(0, m):
        hll.add_raw(probabilistic_test_util.construct_hll_value(log2m, i, register_value))

    cardinality = hll.cardinality()

    # Simplified estimator when all registers take same value: alpha / (m/2^val)
    estimator = HLLUtil.alpha_m_squared(m) / (m / (2**register_value))

    # Assert conditions for large range

    assert estimator > (2**L) / 30

    # Large range correction: -2^L * log(1 - E/2^L)
    try:
        expected = ceil(-1.0 * (2 ** L) * log(1.0 - estimator / (2 ** L)))
    except ValueError:
        expected = 0
    assert cardinality == expected


def test_register_value():
    """
    Tests the bounds on a register's value for a given raw input value.
    """
    log2m = 4  # small enough to make testing easy (add_raw() shifts by one byte)

    # register width 4 (the minimum size)
    regwidth = 4
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
    bit_vector = hll._probabilistic_storage

    # lower-bounds of the register
    hll.add_raw(0x000000000000001)  # 'j'=1
    assert bit_vector.get_register(1) == 0

    hll.add_raw(0x0000000000000012)  # 'j'=2
    assert bit_vector.get_register(2) == 1

    hll.add_raw(0x0000000000000023)  # 'j'=3
    assert bit_vector.get_register(3) == 2

    hll.add_raw(0x0000000000000044)  # 'j'=4
    assert bit_vector.get_register(4) == 3

    hll.add_raw(0x0000000000000085)  # 'j'=5
    assert bit_vector.get_register(5) == 4

    # upper-bounds of the register
    # NOTE:  bear in mind that BitVector itself does ensure that
    #        overflow of a register is prevented
    hll.add_raw(0x0000000000010006)  # 'j'=6
    assert bit_vector.get_register(6) == 13

    hll.add_raw(0x0000000000020007)  # 'j'=7
    assert bit_vector.get_register(7) == 14

    hll.add_raw(0x0000000000040008)  # 'j'=8
    assert bit_vector.get_register(8) == 15

    hll.add_raw(0x0000000000080009)  # 'j'=9
    assert bit_vector.get_register(9) == 15  # overflow

    # sanity checks to ensure that no other bits above the lowest-set
    # bit matters
    # NOTE:  same as case 'j = 6' above
    hll.add_raw(0x000000000003000A)  # 'j'=10
    assert bit_vector.get_register(10) == 13

    hll.add_raw(0x000000000011000B)  # 'j'=11
    assert bit_vector.get_register(11) == 13

    # ------------------------------------------------------------
    # register width 5

    regwidth = 5
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
    bit_vector = hll._probabilistic_storage

    # lower-bounds of the register
    hll.add_raw(0x0000000000000001)  # 'j'=1
    assert bit_vector.get_register(1) == 0

    hll.add_raw(0x0000000000000012)  # 'j'=2
    assert bit_vector.get_register(2) == 1

    hll.add_raw(0x0000000000000023)  # 'j'=3
    assert bit_vector.get_register(3) == 2

    hll.add_raw(0x0000000000000044)  # 'j'=4
    assert bit_vector.get_register(4) == 3

    hll.add_raw(0x0000000000000085)  # 'j'=5
    assert bit_vector.get_register(5) == 4

    # upper-bounds of the register
    # NOTE:  bear in mind that BitVector itself does ensure that
    #        overflow of a register is prevented
    hll.add_raw(0x0000000100000006)  # 'j'=6
    assert bit_vector.get_register(6) == 29

    hll.add_raw(0x0000000200000007)  # 'j'=7
    assert bit_vector.get_register(7) == 30

    hll.add_raw(0x0000000400000008)  # 'j'=8
    assert bit_vector.get_register(8) == 31

    hll.add_raw(0x0000000800000009)  # 'j'=9
    assert bit_vector.get_register(9) == 31  # overflow


def test_clear():
    """
    Tests HLL.clear().
    """
    regwidth = 5
    log2m = 4  # 16 registers per counter
    m = BitUtil.left_shift_int(1, log2m)

    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
    bit_vector = hll._probabilistic_storage
    for i in range(0, m):
        bit_vector.set_register(i, i)

    hll.clear()
    for i in range(0, m):
        assert bit_vector.get_register(i) == 0  # default value of register


# ------------------------------------------------------------
# Serialization


def test_to_from_bytes():
    log2m = 11  # arbitrary
    regwidth = 5

    schema_version = SerializationUtil.DEFAULT_SCHEMA_VERSION
    type = HLLType.FULL
    padding = schema_version.padding_bytes(type)
    data_byte_count = probabilistic_test_util.get_required_bytes(regwidth, BitUtil.left_shift_int(1, log2m))  # aka 2^log2m = m
    expected_byte_count = padding + data_byte_count

    # Should work on an empty element
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)
    bytes = hll.to_bytes(schema_version)

    # assert output length is correct
    assert len(bytes) == expected_byte_count

    in_hll = HLL.from_bytes(bytes)
    assert_elements_equal(hll, in_hll)

    # Should work on a partially filled element
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)

    for i in range(0, 3):
        raw_value = probabilistic_test_util.construct_hll_value(log2m, i, (i+9))
        hll.add_raw(raw_value)

    bytes = hll.to_bytes(schema_version)

    assert len(bytes) == expected_byte_count

    in_hll = HLL.from_bytes(bytes)

    # assert register values correct
    assert_elements_equal(hll, in_hll)

    # Should work on a full set
    hll = HLL.create_for_testing(log2m, regwidth, 128, 256, HLLType.FULL)

    for i in range(0, BitUtil.left_shift_int(1, log2m)):
        raw_value = probabilistic_test_util.construct_hll_value(log2m, i, (i % 9) + 1)
        hll.add_raw(raw_value)

    bytes = hll.to_bytes(schema_version)

    # assert output length is correct
    assert len(bytes) == expected_byte_count

    in_hll = HLL.from_bytes(bytes)

    # assert register values correct
    assert_elements_equal(hll, in_hll)


# ------------------------------------------------------------
# Assertion Helpers


def assert_elements_equal(hll_a, hll_b):
    bit_vector_a = hll_a._probabilistic_storage
    bit_vector_b = hll_b._probabilistic_storage

    iter_a = bit_vector_a.register_iterator()
    iter_b = bit_vector_b.register_iterator()

    try:
        while True:
            assert iter_a.next() == iter_b.next()
    except StopIteration:
        pass

    try:
        iter_a.next()
        pytest.fail()
    except StopIteration:
        pass

    try:
        iter_b.next()
        pytest.fail()
    except StopIteration:
        pass
