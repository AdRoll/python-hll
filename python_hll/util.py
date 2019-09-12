# -*- coding: utf-8 -*-

from math import log
import numpy as np


class BitUtil:
    """
    A collection of bit utilities.
    """

    # The set of least-significant bits for a given ``byte``. ``-1``
    # is used if no bits are set (so as to not be confused with "index of zero"
    # meaning that the least significant bit is the 0th (1st) bit).
    LEAST_SIGNIFICANT_BIT = [
        -1, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
    ]

    @classmethod
    def least_significant_bit(cls, value):
        """
        Computes the least-significant bit of the specified ``long``
        that is set to ``1``. Zero-indexed.

        See <http://stackoverflow.com/questions/757059/position-of-least-significant-bit-that-is-set>
        and <http://www-graphics.stanford.edu/~seander/bithacks.html>.

        :param long value: the ``long`` whose least-significant bit is desired.
        :returns: the least-significant bit of the specified ``long``.
                  ``-1`` is returned if there are no bits set.
        :rtype: int
        """

        if value == 0:
            # by contract
            return -1

        elif value & 0xFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 0) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 0

        elif value & 0xFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 8) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 8

        elif value & 0xFFFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 16) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 16

        elif value & 0xFFFFFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 24) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 24

        elif value & 0xFFFFFFFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 32) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 32

        elif value & 0xFFFFFFFFFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 40) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 40

        elif value & 0xFFFFFFFFFFFFFF != 0:
            index = int(cls.unsigned_right_shift_long(value, 48) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 48

        else:
            index = int(cls.unsigned_right_shift_long(value, 56) & 0xFF)
            return cls.LEAST_SIGNIFICANT_BIT[index] + 56

    @classmethod
    def unsigned_right_shift_long(cls, val, n):
        """
        Equivalent to Java >>> on a long value
        """
        return val if n == 0 else int(np.uint64(val) >> np.uint64(n))

    @classmethod
    def unsigned_right_shift_int(cls, val, n):
        """
        Equivalent to Java >>> on an int value
        """
        return val if n == 0 else int(np.uint32(val) >> np.uint32(n))

    @classmethod
    def unsigned_right_shift_byte(cls, val, n):
        """
        Equivalent to Java >>> on a byte value
        """
        return val if n == 0 else int(np.uint32(val) >> np.uint32(n))

    @classmethod
    def to_signed_byte(cls, i):
        """
        Converts a Python byte (unsigned integer from 0 to 255) to a Java byte
        (signed two's complement integer from -128 to 127).
        :type i: byte
        :rtype: byte
        """
        return i if i <= 127 else i - 256

    @classmethod
    def left_shift_long(cls, long_x, int_y):
        """
        Simulates a Java << for a long.

        :param long_x: expected long value in python code
        :param int_y: expected int value in python
        :returns: left shift result for, x << y
        :rtype: long
        """
        x = np.int64(long_x)
        y = np.int(int_y)
        z = np.left_shift(x, y)

        return np.int64(z.item())

    @classmethod
    def left_shift_int(cls, int_x, int_y):
        """
        Simulates a Java << for an integer.

        :param int_x: expected int value in python code
        :param int_y: expected int value in python
        :returns: left shift result for, x << y
        :rtype: int
        """
        x = np.int32(int_x)
        y = np.int(int_y)
        z = np.left_shift(x, y)

        return z.item()

    @classmethod
    def left_shift_byte(cls, byte_x, int_y):
        """
        Simulates a Java << for a byte.

        :param byte_x: expected byte value in python code
        :param int_y: expected int value in python
        :returns: left shift result for, x << y
        :rtype: int
        """
        x = np.int8(byte_x)  # converts to signed byte, since byte is signed in java
        y = np.int(int_y)
        z = np.left_shift(x, y)

        # In Java, (byte)128 << 3 produces an int.
        return z.item()


class LongIterator:
    """
    A ``long``-based iterator.
    """

    LOG2_BITS_PER_WORD = 6
    BITS_PER_WORD = BitUtil.left_shift_int(1, LOG2_BITS_PER_WORD)

    def __init__(self, register_width, words, register_mask, count):
        self._register_width = register_width
        self._words = words
        self._register_mask = register_mask
        self._count = count

        # register setup
        self._register_index = 0
        self._word_index = 0
        self._remaining_word_bits = self.BITS_PER_WORD
        self._word = self._words[self._word_index]

    def __iter__(self):
        return self

    def __next__(self):
        # Python 3 compatibility
        return self.next()

    def next(self):
        if self._register_index >= self._count:
            raise StopIteration

        if self._remaining_word_bits >= self._register_width:
            register = self._word & self._register_mask

            # shift to the next register
            self._word = BitUtil.unsigned_right_shift_long(self._word, self._register_width)
            self._remaining_word_bits -= self._register_width
        else:  # insufficient bits remaining in current word
            self._word_index += 1  # move to the next word

            register = (self._word | BitUtil.left_shift_long(self._words[self._word_index], self._remaining_word_bits)) & self._register_mask

            # shift to the next partial register (word)
            self._word = BitUtil.unsigned_right_shift_long(self._words[self._word_index], self._register_width - self._remaining_word_bits)
            self._remaining_word_bits += self.BITS_PER_WORD - self._register_width

        self._register_index += 1
        return register


class BitVector:
    """
    A vector (array) of bits that is accessed in units ("registers") of ``width``
    bits which are stored as 64bit "words" (``long``'s).  In this context
    a register is at most 64bits.
    """

    # NOTE:  in this context, a word is 64bits

    # rather than doing division to determine how a bit index fits into 64bit
    # words (i.e. longs), bit shifting is used
    LOG2_BITS_PER_WORD = 6  # =>64bits
    BITS_PER_WORD = BitUtil.left_shift_int(1, LOG2_BITS_PER_WORD)
    BITS_PER_WORD_MASK = BITS_PER_WORD - 1

    # ditto from above but for bytes (for output)
    LOG2_BITS_PER_BYTE = 3  # =>8bits
    BITS_PER_BYTE = BitUtil.left_shift_int(1, LOG2_BITS_PER_BYTE)

    BYTES_PER_WORD = 8  # 8 bytes in a long

    def __init__(self, width, count):
        """
        :param int width: the width of each register.  This cannot be negative or
               zero or greater than 63 (the signed word size).
        :param long count: the number of registers.  This cannot be negative or zero
        """
        # 64bit words
        # ceil((width * count)/BITS_PER_WORD)
        self._words = [0] * BitUtil.unsigned_right_shift_long((width * count) + self.BITS_PER_WORD_MASK, self.LOG2_BITS_PER_WORD)
        # the width of a register in bits (this cannot be more than 64 (the word size))
        self._register_width = width
        self._count = count
        self._register_mask = BitUtil.left_shift_long(1, width) - 1

    def get_register(self, register_index):
        """
        :param long register_index: the index of the register whose value is to be
               retrieved.  This cannot be negative.
        :returns: the value at the specified register index
        :rtype: long
        """
        # NOTE:  if this changes then setMaxRegister() must change
        bit_index = register_index * self._register_width
        first_word_index = BitUtil.unsigned_right_shift_long(bit_index, self.LOG2_BITS_PER_WORD)  # aka (bitIndex / BITS_PER_WORD)
        second_word_index = BitUtil.unsigned_right_shift_long(bit_index + self._register_width - 1, self.LOG2_BITS_PER_WORD)  # see above
        bit_remainder = bit_index & self.BITS_PER_WORD_MASK  # aka (bitIndex % BITS_PER_WORD)

        if first_word_index == second_word_index:
            return BitUtil.unsigned_right_shift_long(self._words[first_word_index], bit_remainder) & self._register_mask
        # else -- register spans words
        return BitUtil.unsigned_right_shift_long(self._words[first_word_index], bit_remainder) | \
            BitUtil.left_shift_long(self._words[second_word_index], self.BITS_PER_WORD - bit_remainder) & self._register_mask

    def set_register(self, register_index, value):
        """
        :param long register_index: the index of the register whose value is to be set.
               This cannot be negative
        :param long value: the value to set in the register
        :rtype: long
        """
        # NOTE:  if this changes then setMaxRegister() must change
        bit_index = register_index * self._register_width
        first_word_index = BitUtil.unsigned_right_shift_long(bit_index, self.LOG2_BITS_PER_WORD)  # aka (bitIndex / BITS_PER_WORD)
        second_word_index = BitUtil.unsigned_right_shift_long(bit_index + self._register_width - 1, self.LOG2_BITS_PER_WORD)  # see above
        bit_remainder = bit_index & self.BITS_PER_WORD_MASK  # aka (bitIndex % BITS_PER_WORD)

        if first_word_index == second_word_index:
            # clear then set
            self._words[first_word_index] &= ~BitUtil.left_shift_long(self._register_mask, bit_remainder)
            self._words[first_word_index] |= BitUtil.left_shift_long(value, bit_remainder)
        else:  # register spans words
            # clear then set each partial word
            self._words[first_word_index] &= BitUtil.left_shift_long(1, bit_remainder) - 1
            self._words[first_word_index] |= BitUtil.left_shift_long(value, bit_remainder)

            self._words[second_word_index] &= ~BitUtil.unsigned_right_shift_long(self._register_mask, self.BITS_PER_WORD - bit_remainder)
            self._words[second_word_index] |= BitUtil.unsigned_right_shift_long(value, self.BITS_PER_WORD - bit_remainder)

    def register_iterator(self):
        """
        :returns: a ``LongIterator`` for iterating starting at the register
                  with index zero. This will never be ``None``.
        :rtype: LongIterator
        """
        return LongIterator(self._register_width, self._words, self._register_mask, self._count)

    def set_max_register(self, register_index, value):
        """
        Sets the value of the specified index register if and only if the specified
        value is greater than the current value in the register.  This is equivalent
        to but much more performant than

        ``vector.setRegister(index, Math.max(vector.getRegister(index), value));``

        :param long register_index: the index of the register whose value is to be set.
               This cannot be negative
        :param long value: the value to set in the register if and only if this value
               is greater than the current value in the register
        :returns: True if and only if the specified value is greater
                  than or equal to the current register value. False
                  otherwise.
        :rtype: boolean
        """
        # NOTE:  if this changes then setRegister() must change
        bit_index = register_index * self._register_width
        first_word_index = BitUtil.unsigned_right_shift_long(bit_index, self.LOG2_BITS_PER_WORD)  # aka (bitIndex / BITS_PER_WORD)
        second_word_index = BitUtil.unsigned_right_shift_long(bit_index + self._register_width - 1, self.LOG2_BITS_PER_WORD)  # see above
        bit_remainder = bit_index & self.BITS_PER_WORD_MASK  # aka (bitIndex % BITS_PER_WORD)

        if first_word_index == second_word_index:
            register_value = BitUtil.unsigned_right_shift_long(self._words[first_word_index], bit_remainder) & self._register_mask
        else:  # register spans words
            # # no need to mask since at top of word
            register_value = BitUtil.unsigned_right_shift_long(self._words[first_word_index], bit_remainder) | \
               BitUtil.left_shift_long(self._words[second_word_index], self.BITS_PER_WORD - bit_remainder) & self._register_mask

        # determine which is the larger and update as necessary
        if value > register_value:
            # NOTE:  matches setRegister()
            if first_word_index == second_word_index:
                # clear then set
                self._words[first_word_index] &= ~BitUtil.left_shift_long(self._register_mask, bit_remainder)
                self._words[first_word_index] |= BitUtil.left_shift_long(value, bit_remainder)
            else:  # register spans words
                # clear then set each partial word
                self._words[first_word_index] &= BitUtil.left_shift_long(1, bit_remainder) - 1
                self._words[first_word_index] |= BitUtil.left_shift_long(value,  bit_remainder)

                self._words[second_word_index] &= ~BitUtil.unsigned_right_shift_long(self._register_mask, self.BITS_PER_WORD - bit_remainder)
                self._words[second_word_index] |= BitUtil.unsigned_right_shift_long(value, self.BITS_PER_WORD - bit_remainder)
        # else -- the register value is greater (or equal) so nothing needs to be done

        return value >= register_value

    def fill(self, value):
        """
        Fills this bit vector with the specified bit value.  This can be used to
        clear the vector by specifying ``0``.

        :param long value: the value to set all bits to (only the lowest bit is used)
        :rtype: void
        """
        for i in range(self._count):
            self.set_register(i, value)

    def get_register_contents(self, serializer):
        """
        Serializes the registers of the vector using the specified serializer.

        :param BigEndianAscendingWordSerializer serializer: the serializer to use. This cannot be ``None``.
        :rtype: void
        """
        iterator = self.register_iterator()

        for itr in iterator:
            serializer.write_word(itr)


class NumberUtil:
    """
    A collection of utilities to work with numbers.
    """

    # loge(2) (log-base e of 2)
    LOGE_2 = 0.6931471805599453

    # the hex characters
    HEX = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

    @classmethod
    def log2(cls, value):
        """
        Computes the ``log2`` (log-base-two) of the specified value.

        :param float value: the ``float`` for which the ``log2`` is
               desired.
        :returns: the ``log2`` of the specified value
        :rtype: float
        """
        # REF:  http://en.wikipedia.org/wiki/Logarithmic_scale (conversion of bases)
        return log(value) / cls.LOGE_2

    @classmethod
    def to_hex(cls, bytes, offset, count):
        """
        Converts the specified array of ``byte``'s into a string of
        hex characters (low ``byte`` first).

        :param list bytes: the array of ``byte``'s that are to be converted.
               This cannot be ``None`` though it may be empty.
        :param int offset: the offset in ``bytes`` at which the bytes will
               be taken.  This cannot be negative and must be less than
               ``bytes.length - 1``.
        :param int count: the number of bytes to be retrieved from the specified array.
               This cannot be negative.  If greater than ``bytes.length - offset``
               then that value is used.
        :returns: a string of at most ``count`` characters that represents
                  the specified byte array in hex.  This will never be ``None``
                  though it may be empty if ``bytes`` is empty or ``count``
                  is zero.
        :rtype: string
        """
        if offset >= len(bytes):  # by contract
            raise Exception("Offset is greater than the length, {offset} >= {byte_array_length}"
                            .format(offset=offset, byte_array_length=len(bytes)))
        byte_count = min(len(bytes) - offset, count)
        upper_bound = byte_count + offset

        chars = [None] * (byte_count * 2)   # two chars per byte
        char_index = 0
        for i in range(offset, upper_bound):
            value = bytes[i]
            chars[char_index] = cls.HEX[(BitUtil.unsigned_right_shift_byte(value, 4)) & 0x0F]
            char_index += 1
            chars[char_index] = cls.HEX[value & 0x0F]
            char_index += 1

        return ''.join(chars)

    @classmethod
    def from_hex(cls, string, offset, count):
        """
        Converts the specified array of hex characters into an array of ``byte``'s
        (low ``byte`` first).

        :param string string: the string of hex characters to be converted into ``byte``'s.
               This cannot be ``None`` though it may be blank.
        :param int offset: the offset in the string at which the characters will be
               taken.  This cannot be negative and must be less than ``string.length() - 1``.
        :param int count: the number of characters to be retrieved from the specified
               string.  This cannot be negative and must be divisible by two
               (since there are two characters per ``byte``).
        :returns: the array of ``byte``'s that were converted from the
                  specified string (in the specified range).  This will never be
                  ``None`` though it may be empty if ``string``
                  is empty or ``count`` is zero.
        :rtype: list
        """

        if offset >= len(string):  # by contract
            raise Exception("Offset is greater than the length, {offset} >= {string_length}"
                            .format(offset=offset, string_length=len(string)))
        if (count & 0x01) != 0:  # by contract
            raise Exception("Count is not divisible by two, ({})".format(count))

        char_count = min(len(string) - offset, count)
        upper_bound = offset + char_count

        byte_array = [0] * (BitUtil.unsigned_right_shift_int(char_count, 1))  # aka /2
        byte_index = 0  # beginning
        for i in range(0, upper_bound, 2):
            p1 = BitUtil.left_shift_int(cls._digit(string[i]), 4)
            p2 = cls._digit(string[i+1])
            p = (p1 | p2) & 0xFF

            byte_array[byte_index] = BitUtil.to_signed_byte(p)
            byte_index += 1
        return byte_array

    @classmethod
    def _digit(cls, character):
        """
        :param string character: a hex character to be converted to a ``byte``.
               This cannot be a character other than [a-fA-F0-9].
        :returns: the value of the specified character.  This will be a value ``0``
                  through ``15``.
        :rtype: int
        """
        if character == '0':
            return 0
        elif character == '1':
            return 1
        elif character == '2':
            return 2
        elif character == '3':
            return 3
        elif character == '4':
            return 4
        elif character == '5':
            return 5
        elif character == '6':
            return 6
        elif character == '7':
            return 7
        elif character == '8':
            return 8
        elif character == '9':
            return 9
        elif character in ['a', 'A']:
            return 10
        elif character in ['b', 'B']:
            return 11
        elif character in ['c', 'C']:
            return 12
        elif character in ['d', 'D']:
            return 13
        elif character in ['e', 'E']:
            return 14
        elif character in ['f', 'F']:
            return 15
        else:
            raise Exception("Character is not in [a-fA-F0-9]: ({})".format(character))
