#!/usr/bin/env python
# -*- coding: utf-8 -*-

from python_hll.util import BitVector

"""Unit tests for BitVector."""


def test_get_set_register():
    """
    Tests ``BitVector.get_register()`` and ``BitVector.set_register()``.
    """
    # NOTE: registers are only 5bits wide
    vector1 = BitVector(5, 2**7)  # width=5, count=2^7
    vector2 = BitVector(5, 2**7)
    vector3 = BitVector(5, 2**7)
    vector4 = BitVector(5, 2**7)
    for i in range(0, 2**7):
        vector1.set_register(i, 0x1F)
        vector2.set_register(i, (i & 0x1F))
        vector3.set_register(i, ((127 - i) & 0x1F))
        vector4.set_register(i, 0x15)

    for i in range(0, 2 ** 7):
        assert vector1.get_register(i) == 0x1F
        assert vector2.get_register(i) == i & 0x1F
        assert vector3.get_register(i) == (127 - i) & 0x1F
        assert vector4.get_register(i) == 0x15
