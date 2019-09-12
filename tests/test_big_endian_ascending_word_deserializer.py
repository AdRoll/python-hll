#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for BigEndianAscendingWordDeserializer """

from sys import maxsize
import random
from python_hll.serialization import BigEndianAscendingWordDeserializer, BigEndianAscendingWordSerializer
from python_hll.util import BitUtil


def test_constructor_error():
    """
    Error checking tests for constructor.
    """

    # word length too small
    try:
        BigEndianAscendingWordDeserializer(0, 0, [0])
        assert False, "Should complain about too-short words."
    except ValueError as e:
        assert "Word length must be" in str(e)

    # word length too large
    try:
        BigEndianAscendingWordDeserializer(65, 0, [0])
        assert False, "Should complain about too-long words."
    except ValueError as e:
        assert "Word length must be" in str(e)

    # byte padding negative
    try:
        BigEndianAscendingWordDeserializer(5, -1, [0])
    except ValueError as e:
        assert "Byte padding must be" in str(e)


def test_smoke_64_bit_word():
    serializer = BigEndianAscendingWordSerializer(64, 5, 0)

    # Check that the sign bit is being preserved.
    serializer.write_word(-1)
    serializer.write_word(-112894714)

    # CHeck "special values"
    serializer.write_word(0)
    serializer.write_word(maxsize)
    serializer.write_word(-maxsize - 1)

    bytes_ = serializer.get_bytes()

    deserializer = BigEndianAscendingWordDeserializer(64, 0, bytes_)
    assert deserializer.total_word_count() == 5

    assert deserializer.read_word() == -1
    assert deserializer.read_word() == -112894714
    assert deserializer.read_word() == 0
    assert deserializer.read_word() == maxsize
    assert deserializer.read_word() == -maxsize - 1


def test_ascending_smoke(fastonly):
    """
    A smoke/fuzz test for ascending (from zero) word values.
    """
    word_length = 5
    while word_length < 65:
        run_ascending_test(word_length, 3, 1000 if fastonly else 100000)
        word_length += 1


def test_random_smoke(fastonly):
    """
    A smoke/fuzz test for random word values.
    """
    word_length = 5
    while word_length < 65:
        run_random_test(word_length, 3, 1000 if fastonly else 100000, word_length)
        word_length += 1


def run_random_test(word_length, byte_padding, word_count, seed):
    """
    Runs a test which serializes and deserializes random word values.
    """
    random.seed(seed)

    word_mask = ~0 if word_length == 64 else BitUtil.left_shift_long(1, word_length) - 1

    serializer = BigEndianAscendingWordSerializer(word_length, word_count, byte_padding)

    for _ in range(word_count):
        value = random.randint(0, maxsize) & word_mask
        serializer.write_word(value)

    bytes_ = serializer.get_bytes()

    deserializer = BigEndianAscendingWordDeserializer(word_length, byte_padding, bytes_)

    assert deserializer.total_word_count() == word_count

    # verification random
    random.seed(seed)
    for _ in range(word_count):
        assert deserializer.read_word() == (random.randint(0, maxsize) & word_mask)


def run_ascending_test(word_length, byte_padding, word_count):
    """
    Runs a test which serializes and deserializes ascending (from zero) word values.
    """
    word_mask = ~0 if word_length == 64 else BitUtil.left_shift_long(1, word_length) - 1

    serializer = BigEndianAscendingWordSerializer(word_length, word_count, byte_padding)

    for i in range(word_count):
        serializer.write_word(i & word_mask)

    bytes_ = serializer.get_bytes()

    deserializer = BigEndianAscendingWordDeserializer(word_length, byte_padding, bytes_)

    assert deserializer.total_word_count() == word_count

    for i in range(word_count):
        assert deserializer.read_word() == (i & word_mask)
