#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Serialization smoke-tests."""

import random
import sys
from copy import deepcopy
from python_hll.hlltype import HLLType
from python_hll.hll import HLL
from python_hll.serialization import SerializationUtil
from python_hll.util import BitUtil

# A fixed random seed so that this test is reproducible.
RANDOM_SEED = 1


def test_parameters_byte():
    assert 5 == SerializationUtil.register_width(0x85)
    assert 5 == SerializationUtil.register_count_log2(0x85)

    assert 0x85 == SerializationUtil.pack_parameters_byte(5, 5)


def test_parameters_byte_roundtrip():
    log2m = 5
    regwidth = 6
    expthresh = 0
    sparseon = False
    input_hll = HLL(log2m, regwidth, expthresh, sparseon)

    output_hll = HLL.from_bytes(input_hll.to_bytes())
    assert log2m == output_hll._log2m
    assert regwidth == output_hll._regwidth


def test_cutoff_byte_auto():
    log2m = 5
    regwidth = 3
    expthresh = -1
    sparseon = True
    input_hll = HLL(log2m, regwidth, expthresh, sparseon)

    output_hll = HLL.from_bytes(input_hll.to_bytes())
    assert not output_hll._sparse_off
    assert output_hll._explicit_auto
    assert not output_hll._explicit_off


def test_cutoff_byte_max():
    log2m = 5
    regwidth = 3
    expthresh = 18
    sparseon = True
    input_hll = HLL(log2m, regwidth, expthresh, sparseon)

    output_hll = HLL.from_bytes(input_hll.to_bytes())
    assert not output_hll._sparse_off
    assert not output_hll._explicit_auto
    assert not output_hll._explicit_off


def test_serialization_smoke(fastonly):
    """
    A smoke-test that covers serialization/deserialization of an HLL
    under all possible parameters.
    """
    random.seed(RANDOM_SEED)
    random_count = 250
    max_java_long = 9223372036854775807
    randoms = [random.randint(1, max_java_long) for i in range(0, random_count)]
    assert_cardinality(HLLType.EMPTY, randoms, fastonly)
    assert_cardinality(HLLType.EXPLICIT, randoms, fastonly)
    assert_cardinality(HLLType.SPARSE, randoms, fastonly)
    assert_cardinality(HLLType.FULL, randoms, fastonly)


def assert_cardinality(hll_type, items, fastonly):
    # NOTE: log2m<=16 was chosen as the max log2m parameter so that the test
    #       completes in a reasonable amount of time. Not much is gained by
    #       testing larger values - there are no more known serialization
    #       related edge cases that appear as log2m gets even larger.
    log2m_range = range(HLL.MINIMUM_LOG2M_PARAM, 16 + 1)
    regw_range = range(HLL.MINIMUM_REGWIDTH_PARAM, HLL.MAXIMUM_REGWIDTH_PARAM + 1)
    expthr_range = range(HLL.MINIMUM_EXPTHRESH_PARAM, HLL.MAXIMUM_EXPTHRESH_PARAM + 1)
    if fastonly:
        log2m_range = (HLL.MINIMUM_LOG2M_PARAM, 16)
        regw_range = (HLL.MINIMUM_REGWIDTH_PARAM, HLL.MAXIMUM_REGWIDTH_PARAM)
        expthr_range = (HLL.MINIMUM_EXPTHRESH_PARAM, HLL.MAXIMUM_EXPTHRESH_PARAM)
    for log2m in log2m_range:
        for regw in regw_range:
            for expthr in expthr_range:
                for sparse in [True, False]:
                    hll = HLL(log2m, regw, expthr, sparse, hll_type)
                    for item in items:
                        hll.add_raw(item)
                    copy = HLL.from_bytes(hll.to_bytes())
                    assert copy.cardinality() == hll.cardinality()
                    assert copy.get_type() == hll.get_type()
                    assert copy.to_bytes() == hll.to_bytes()

                    clone = deepcopy(hll)
                    assert clone.cardinality() == hll.cardinality()
                    assert clone.get_type() == hll.get_type()
                    assert clone.to_bytes() == hll.to_bytes()

                    sys.stdout.write('.')
                    sys.stdout.flush()
