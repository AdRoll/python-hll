#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for BigEndianAscendingWordSerializer """

from python_hll.serialization import BigEndianAscendingWordSerializer


def test_constructor_error():
    """
    Test for contructors
    """

    # Word length is too short
    try:
        BigEndianAscendingWordSerializer(0, 1, 0)
        assert False, "Should complain about too-short words."
    except ValueError as e:
        assert 'Word length must be >= 1 and <= 64. (was: 0)' == str(e)

    # Word length is too long
    try:
        BigEndianAscendingWordSerializer(65, 1, 0)
        assert False, "Should complain about too-long words."
    except ValueError as e:
        assert "Word length must be" in str(e)

    # Word Count is negative
    try:
        BigEndianAscendingWordSerializer(5, -1, 0)
        assert False, "Should complain about negative word count."
    except ValueError as e:
        assert "Word count must be" in str(e)

    # Byte padding is negative
    try:
        BigEndianAscendingWordSerializer(5, 1, -1)
        assert False, "Should complain about negative byte padding."
    except ValueError as e:
        assert "Byte padding must be" in str(e)


def test_early_get_bytes():
    """
    Tests runtime exception thrown at premature call
    """

    serializer = BigEndianAscendingWordSerializer(5, 1, 0)
    try:
        serializer.get_bytes()
        assert False, "Should throw."
    except ValueError as r:
        assert "Not all words" in str(r)


def test_smoke_explicit_params():
    """
    Smoke test for typical parameters
    """
    short_word_length = 64

    # Should work on empty sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 0, 0)
    assert serializer.get_bytes() == []

    # Should work on byte-divisible sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 2, 0)
    serializer.write_word(-4995993186629670228)  # 0xBAAAAAAAAAAAAAACL
    serializer.write_word(-8070450532247928847)  # 0x8FFFFFFFFFFFFFF1L

    # Bytes:
    #   ======
    #   0xBA 0xAA 0xAA 0xAA 0xAA 0xAA 0xAA 0xAC
    #   0x8F 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xF1
    #  -70 -86 ...                        -84
    #  -113 -1 ...                        -15

    all_bytes = serializer.get_bytes()
    expected_bytes = [-70, -86, -86, -86, -86, -86, -86, -84, -113, -1, -1, -1, -1, -1, -1, -15]
    assert all_bytes == expected_bytes

    # Should pad the array correctly.
    serializer = BigEndianAscendingWordSerializer(short_word_length, 1, 1)
    serializer.write_word(1)
    all_bytes = serializer.get_bytes()
    expected_bytes = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert all_bytes == expected_bytes


def test_smoke_probabilistic_params():
    """
    Smoke Test for typical parameters used in practice.
    """
    short_word_length = 5

    # Should work on an empty sequence with no padding.
    serializer = BigEndianAscendingWordSerializer(short_word_length, 0, 0)
    assert serializer.get_bytes() == []

    # Should work on a non-byte-divisible sequence with no padding.
    serializer = BigEndianAscendingWordSerializer(short_word_length, 3, 0)
    serializer.write_word(9)
    serializer.write_word(31)
    serializer.write_word(1)

    # The values:
    # -----------
    # 9     |31    |1     |padding

    # Corresponding bits:
    # ------------------
    # 0100 1|111 11|00 001|0

    # And the hex/decimal (Are python bytes signed????????):
    # -----------------------------------------------------
    # 0100 1111 -> 0x4F -> 79
    # 1100 0010 -> 0xC2 -> -62

    all_bytes = serializer.get_bytes()
    expected_bytes = [79, -62]
    assert all_bytes == expected_bytes

    # Should work on a byte-divisible sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 8, 0)

    for i in range(1, 9):
        serializer.write_word(i)

    # Values: 1-8
    # Corresponding bits:
    # ------------------
    # 00001
    # 00010
    # 00011
    # 00100
    # 00101
    # 00110
    # 00111
    # 01000

    # And the hex:
    # ------------
    # 0000 1000 => 0x08 => 8
    # 1000 0110 => 0x86 => -122
    # 0100 0010 => 0x62 => 66
    # 1001 1000 => 0x98 => -104
    # 1110 1000 => 0xE8 => -24

    all_bytes = serializer.get_bytes()
    expected_bytes = [8, -122, 66, -104, -24]
    assert all_bytes == expected_bytes

    # Should pad the array correctly
    serializer = BigEndianAscendingWordSerializer(short_word_length, 1, 1)
    serializer.write_word(1)

    # 1 byte leading padding | value 1 | trailing padding
    # 0000 0000 | 0000 1|000
    all_bytes = serializer.get_bytes()
    expected_bytes = [0, 8]
    assert all_bytes == expected_bytes


def test_smoke_sparse_params():
    """
    Smoke test for typical parameters used in practice.
    """
    short_word_length = 17

    # Should work on an empty sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 0, 0)
    assert serializer.get_bytes() == []

    # Should work on a non-byte-divisible sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 3, 0)
    serializer.write_word(9)
    serializer.write_word(42)
    serializer.write_word(75)
    # The values:
    # -----------
    # 9                    |42                   |75                   |padding

    # Corresponding bits:
    # ------------------
    # 0000 0000 0000 0100 1|000 0000 0000 1010 10|00 0000 0000 1001 011|0 0000

    # And the hex/decimal (remember Java bytes are signed):
    # -----------------------------------------------------
    # 0000 0000 -> 0x00 -> 0
    # 0000 0100 -> 0x04 -> 4
    # 1000 0000 -> 0x80 -> -128
    # 0000 1010 -> 0x0A -> 10
    # 1000 0000 -> 0x80 -> -128
    # 0000 1001 -> 0x09 -> 9
    # 0110 0000 -> 0x60 -> 96

    all_bytes = serializer.get_bytes()
    expected_bytes = [0, 4, -128, 10, -128, 9, 96]
    assert all_bytes == expected_bytes

    # Should work on a byte-divisible sequence with no padding
    serializer = BigEndianAscendingWordSerializer(short_word_length, 8, 0)

    for i in range(1, 9):
        serializer.write_word(i)

    # Values: 1-8
    # Corresponding bits:
    # ------------------
    # 0000 0000 0000 0000 1
    # 000 0000 0000 0000 10
    #  00 0000 0000 0000 011
    # 0 0000 0000 0000 0100

    # 0000 0000 0000 0010 1
    # 000 0000 0000 0001 10
    # 00 0000 0000 0000 111
    # 0 0000 0000 0000 1000

    # And the hex:
    # ------------
    # 0000 0000 -> 0x00 -> 0
    # 0000 0000 -> 0x00 -> 0
    # 1000 0000 -> 0x80 -> -128
    # 0000 0000 -> 0x00 -> 0
    # 1000 0000 -> 0x80 -> -128
    # 0000 0000 -> 0x00 -> 0
    # 0110 0000 -> 0x60 -> 96
    # 0000 0000 -> 0x00 -> 0
    # 0100 0000 -> 0x40 -> 64
    # 0000 0000 -> 0x00 -> 0
    # 0010 1000 -> 0x28 -> 40
    # 0000 0000 -> 0x00 -> 0
    # 0001 1000 -> 0x18 -> 24
    # 0000 0000 -> 0x00 -> 0
    # 0000 1110 -> 0x0D -> 14
    # 0000 0000 -> 0x00 -> 0
    # 0000 1000 -> 0x08 -> 8

    all_bytes = serializer.get_bytes()
    expected_bytes = [0, 0, -128, 0, -128, 0, 96, 0, 64, 0, 40, 0, 24, 0, 14, 0, 8]
    assert all_bytes == expected_bytes

    # Should pad the array correctly
    serializer = BigEndianAscendingWordSerializer(short_word_length, 1, 1)
    serializer.write_word(1)

    all_bytes = serializer.get_bytes()
    expected_bytes = [0, 0, 0, -128]
    assert all_bytes == expected_bytes
