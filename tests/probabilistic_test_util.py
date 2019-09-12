#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from python_hll.util import BitUtil
from math import ceil


def construct_hll_value(log2m, register_index, register_value):
    """
    Constructs a value that when added raw to a HLL will set the register at
    ``register_index`` to ``register_value``.

    :param log2m: The log-base-2 of the number of registers in the HLL
    :type log2m: int
    :param register_index: The index of the register to set
    :type register_index: int
    :param register_value: the value to set the register to
    :type register_value: int
    :rtype: int
    """
    partition = register_index
    substream_value = BitUtil.left_shift_long(1, register_value - 1)
    return BitUtil.left_shift_long(substream_value, log2m) | partition


def get_register_index(raw_value, log2m):
    """
    Extracts the HLL register index from a raw value.
    """
    m_bits_mask = BitUtil.left_shift_int(1, log2m) - 1
    j = raw_value & m_bits_mask
    return j


def get_register_value(raw_value, log2m):
    """
    Extracts the HLL register value from a raw value.
    """
    substream_value = BitUtil.unsigned_right_shift_long(raw_value, log2m)
    if substream_value == 0:
        # The paper does not cover p(0x0), so the special value 0 is used.
        # 0 is the original initialization value of the registers, so by
        # doing this the HLL simply ignores it. This is acceptable
        # because the probability is 1/(2^(2^register_size_in_bits)).
        p_w = 0
    else:
        p_w = BitUtil.to_signed_byte(min(1 + BitUtil.least_significant_bit(substream_value), 31))
    return p_w


def get_required_bytes(short_word_length, register_count):
    """
    Returns the number of bytes required to pack ``register_count``
    registers of width ``short_word_length``.
    """
    return ceil((register_count * short_word_length) / 8)
