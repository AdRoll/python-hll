#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import pytest
import sys
from python_hll.util import NumberUtil
from python_hll.hll import HLL
from python_hll.hlltype import HLLType
import probabilistic_test_util

"""
Compares the HLLs to the files in the data directory. See README.txt for
more information about this test data.
"""

LOG2M = 11
REGWIDTH = 5
EXPLICIT_THRESHOLD = 256
SPARSE_THRESHOLD = 850


def test_cumulative_add_cardinality_correction(fastonly):
    do_test_add('cumulative_add_cardinality_correction.csv', fastonly)


def test_cumulative_add_comprehensive_promotion(fastonly):
    do_test_add('cumulative_add_comprehensive_promotion.csv', fastonly)


def test_cumulative_add_sparse_edge(fastonly):
    do_test_add('cumulative_add_sparse_edge.csv', fastonly)


def test_cumulative_add_sparse_random(fastonly):
    do_test_add('cumulative_add_sparse_random.csv', fastonly)


def test_cumulative_add_sparse_step(fastonly):
    do_test_add('cumulative_add_sparse_step.csv', fastonly)


def test_cumulative_union_comprehensive(fastonly):
    do_test_union('cumulative_union_comprehensive.csv', fastonly)


def test_cumulative_union_explicit_explicit(fastonly):
    do_test_union('cumulative_union_explicit_explicit.csv', fastonly)


def test_cumulative_union_explicit_promotion(fastonly):
    do_test_union('cumulative_union_explicit_promotion.csv', fastonly)


def test_cumulative_union_probabilistic_probabilistic(fastonly):
    do_test_union('cumulative_union_probabilistic_probabilistic.csv', fastonly)


def test_cumulative_union_sparse_promotion(fastonly):
    do_test_union('cumulative_union_sparse_promotion.csv', fastonly)


def test_cumulative_union_sparse_sparse(fastonly):
    do_test_union('cumulative_union_sparse_sparse.csv', fastonly)


def test_cumulative_union_sparse_full_representation():
    # I'm not exactly sure how this test is suppossed to work - it's different
    # from the other union tests. For now I will just construct the HLLs in the
    # same way as Java's sparseFullRepresentationTest() and compare the output.

    # The file is generated from IntegrationTestGenerator.java.
    filename = 'cumulative_union_sparse_full_representation.csv'
    with open('tests/data/%s' % filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
    print('')
    print('test_integration: %s: %s rows:' % (filename, len(rows)))

    empty_hll_1 = new_hll(HLLType.EMPTY)
    empty_hll_2 = new_hll(HLLType.EMPTY)
    assert_sparse_full_row_equals(empty_hll_1, empty_hll_2, rows[0], filename, 1)

    full_hll = new_hll(HLLType.FULL)
    full_hll.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, 0, 1))
    sparse_hll = new_hll(HLLType.SPARSE)
    sparse_hll.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, 0, 1))
    assert_sparse_full_row_equals(full_hll, sparse_hll, rows[1], filename, 2)

    full_hll_2 = new_hll(HLLType.FULL)
    full_hll_2.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, 1, 1))
    sparse_hll.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, 1, 1))
    assert_sparse_full_row_equals(full_hll_2, sparse_hll, rows[2], filename, 3)

    full_hll_3 = new_hll(HLLType.FULL)
    for i in range(2, SPARSE_THRESHOLD + 1):
        full_hll_3.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, i, 1))
        sparse_hll.add_raw(probabilistic_test_util.construct_hll_value(LOG2M, i, 1))
    assert_sparse_full_row_equals(full_hll_3, sparse_hll, rows[3], filename, 4)


def assert_sparse_full_row_equals(hll, union_hll, row, filename, line):
    """
    Asserts that the given HLLs match the row in cumulative_union_sparse_full_representation.csv.
    """
    assert float_cardinality(hll) == pytest.approx(float(row['cardinality'])), '%s:%s' % (filename, line)
    assert hll_to_string(hll) == row['HLL'], '%s:%s' % (filename, line)
    assert float_cardinality(union_hll) == pytest.approx(float(row['union_cardinality'])), '%s:%s' % (filename, line)
    assert hll_to_string(union_hll) == row['union_HLL'], '%s:%s' % (filename, line)


def new_hll(type):
    """
    Shortcut for testing constructor, which uses the constants defined at
    the top of the file as default parameters.

    :returns: a new ``HLL`` of specified type, which uses the parameters
              ``LOG2M`` ``REGWIDTH``, ``EXPLICIT_THRESHOLD`` and ``SPARSE_THRESHOLD`` specified above.
    """
    return HLL.create_for_testing(LOG2M, REGWIDTH, EXPLICIT_THRESHOLD, SPARSE_THRESHOLD, type)


def do_test_add(filename, fastonly):
    """
    Tests an "add"-style test file.
    """
    with open('tests/data/%s' % filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line = 1
        rows = [row for row in csv_reader]
        if fastonly:
            rows = rows[0:500]
        print('')
        print('test_integration: %s: %s rows: (each . = 100 rows)' % (filename, len(rows)))
        for row in rows:
            if line == 1:
                hll = string_to_hll(row['multiset'])
                line += 1
                continue
            hll.add_raw(int(row['raw_value']))
            assert float_cardinality(hll) == pytest.approx(float(row['cardinality'])), '%s:%s' % (filename, line)
            assert hll_to_string(hll) == row['multiset'], '%s:%s' % (filename, line)
            hll = string_to_hll(row['multiset'])
            line += 1
            if line % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()


def do_test_union(filename, fastonly):
    """
    Tests an "union"-style test file.
    """
    with open('tests/data/%s' % filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line = 1
        rows = [row for row in csv_reader]
        if fastonly:
            rows = rows[0:500]
        print('')
        print('test_integration: %s: %s rows: (each . = 100 rows)' % (filename, len(rows)))
        for row in rows:
            if line == 1:
                hll = string_to_hll(row['union_multiset'])
                line += 1
                continue
            other_hll = string_to_hll(row['multiset'])
            assert float_cardinality(other_hll) == pytest.approx(float(row['cardinality'])), '%s:%s:multiset' % (filename, line)
            hll.union(other_hll)
            assert float_cardinality(hll) == pytest.approx(float(row['union_cardinality'])), '%s:%s' % (filename, line)
            assert hll_to_string(hll) == row['union_multiset'], '%s:%s' % (filename, line)
            hll = string_to_hll(row['union_multiset'])
            line += 1
            if line % 100 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()


def float_cardinality(hll):
    """
    Returns the algorithm-specific cardinality of the specified ``HLL``
     ``String`` appropriate for comparison with the algorithm-specific
     cardinality provided by the PostgreSQL implementation.
    :param HLL hll: The HLL whose algorithm-specific cardinality is to be printed.
           This cannot be ``None``.
    :returns: the algorithm-specific cardinality of the instance as a PostgreSQL-
              compatible String. This will never be ``None``
    :rtype: float
    """
    if hll.get_type() == HLLType.EMPTY:
        return 0
    elif hll.get_type() == HLLType.EXPLICIT:  # promotion has not yet occurred
        return hll.cardinality()
    elif hll.get_type() == HLLType.SPARSE:
        return hll._sparse_probabilistic_algorithm_cardinality()
    elif hll.get_type() == HLLType.FULL:
        return hll._full_probabilistic_algorithm_cardinality()
    else:
        raise Exception('Unknown HLL type ' + str(hll.get_type()))


def string_to_hll(s):
    """
    Converts a string (with \\x) to an HLL.
    """
    s = s[2:]
    return HLL.from_bytes(NumberUtil.from_hex(s, 0, len(s)))


def hll_to_string(hll):
    """
    Converts an HLL to a string (with \\x)
    """
    bytes = hll.to_bytes()
    return '\\x' + NumberUtil.to_hex(bytes, 0, len(bytes))
