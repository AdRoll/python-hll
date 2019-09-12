# -*- coding: utf-8 -*-
from __future__ import division
from python_hll.hlltype import HLLType
from python_hll.util import BitUtil


class BigEndianAscendingWordDeserializer:
    """
    A corresponding deserializer for BigEndianAscendingWordSerializer.
    """

    # The number of bits per byte.
    BITS_PER_BYTE = 8

    # long mask for the maximum value stored in a byte
    BYTE_MASK = BitUtil.left_shift_long(1, BITS_PER_BYTE) - 1

    # :var int word_length: The length in bits of the words to be read.
    # :var list bytes: The byte array to which the words are serialized.
    # :var int byte_padding: The number of leading padding bytes in 'bytes' to be ignored.
    # :var int word_count: The number of words that the byte array contains.
    # :var int current_word_index: The current read state.

    def __init__(self, word_length, byte_padding, bytes):
        """
        :param int word_length: the length in bits of the words to be deserialized. Must
            be less than or equal to 64 and greater than or equal to 1.
        :param int byte_padding: the number of leading bytes that pad the serialized words.
        :param list bytes: the byte array containing the serialized words. Cannot be ``None``.
        """
        if not 1 <= word_length <= 64:
            raise ValueError("Word length must be >= 1 and <= 64. (was: {word_length})".format(word_length=word_length))

        if byte_padding < 0:
            raise ValueError("Byte padding must be >= zero. (was: {byte_padding})".format(byte_padding=byte_padding))

        self._word_length = word_length
        self._bytes = bytes
        self._byte_padding = byte_padding

        self.data_bytes = (len(bytes) - byte_padding)
        self.data_bits = self.data_bytes * self.BITS_PER_BYTE

        self.word_count = int(self.data_bits/self._word_length)

        self.current_word_index = 0

    def read_word(self):
        """
        Return the next word in the sequence. Should not be called more than ``total_word_count`` times.

        :rtype: long
        """
        word = self._read_word(self.current_word_index)
        self.current_word_index += 1
        return word

    def _read_word(self, position):
        """
        Reads the word at the specific sequence position (zero-indexed).

        :param int position: the zero-indexed position of the word to be read. This must be greater
            than or equal to zero.
        :returns: the value of the serialized word at the specific position.
        :rtype: long
        """
        if position < 0:
            raise ValueError("Array index out of bounds for {position}".format(position=position))

        # First bit of the word
        first_bit_index = (position * self._word_length)
        first_byte_index = (self._byte_padding + int(first_bit_index / self.BITS_PER_BYTE))
        first_byte_skip_bits = int(first_bit_index % self.BITS_PER_BYTE)

        # Last bit of the word
        last_bit_index = (first_bit_index + self._word_length - 1)
        last_byte_index = (self._byte_padding + int(last_bit_index / self.BITS_PER_BYTE))

        bits_after_byte_boundary = int((last_bit_index + 1) % self.BITS_PER_BYTE)

        # If the word terminates at the end of the last byte,, consume the whole
        # last byte.
        if bits_after_byte_boundary == 0:
            last_byte_bits_to_consume = self.BITS_PER_BYTE
        else:
            # Otherwise, only consume what is necessary.
            last_byte_bits_to_consume = bits_after_byte_boundary

        if last_byte_index >= len(self._bytes):
            raise ValueError("Word out of bounds of backing array, {} >= {}".format(last_byte_index, len(self._bytes)))

        # Accumulator
        value = 0

        # -------------------------------------------------------------------
        # First byte
        bits_remaining_in_first_byte = (self.BITS_PER_BYTE - first_byte_skip_bits)
        bits_to_consume_in_first_byte = min(bits_remaining_in_first_byte, self._word_length)
        first_byte = self._bytes[first_byte_index]

        # Mask off the bits to skip in the first byte.
        first_byte_mask = (BitUtil.left_shift_long(1, bits_remaining_in_first_byte) - 1)
        first_byte &= first_byte_mask

        # Right-align relevant bits of first byte.
        first_byte = BitUtil.unsigned_right_shift_long(
            first_byte,
            bits_remaining_in_first_byte - bits_to_consume_in_first_byte
        )

        value |= first_byte

        # If the first byte contains the whold word, short-circuit.
        if first_byte_index == last_byte_index:
            return value

        # -------------------------------------------------------------
        # Middle bytes
        middle_byte_count = int(last_byte_index - first_byte_index - 1)
        for i in range(middle_byte_count):
            middle_byte = self._bytes[first_byte_index + i + 1] & self.BYTE_MASK
            # Push middle byte onto accumulator.
            value = BitUtil.left_shift_long(value, self.BITS_PER_BYTE)
            value |= middle_byte

        # --------------------------------------------------
        # Last byte
        last_byte = (self._bytes[last_byte_index] & self.BYTE_MASK)
        last_byte >>= self.BITS_PER_BYTE - last_byte_bits_to_consume
        value = BitUtil.left_shift_long(value, last_byte_bits_to_consume)
        value |= last_byte
        return value

    def total_word_count(self):
        """
        Returns the number of words that could be encoded in the sequence.

        NOTE:  the sequence that was encoded may be shorter than the value this
               method returns due to padding issues within bytes. This guarantees
               only an upper bound on the number of times ``readWord()``
               can be called.

        :returns: the maximum number of words that could be read from the sequence.
        :rtype: int
        """
        return self.word_count


class BigEndianAscendingWordSerializer:
    """
    A serializer that writes a sequence of fixed bit-width 'words' to a byte array.
    Bitwise OR is used to write words into bytes, so a low bit in a word is also
    a low bit in a byte. However, a high byte in a word is written at a lower index
    in the array than a low byte in a word. The first word is written at the lowest
    array index. Each serializer is one time use and returns its backing byte
    array.

    This encoding was chosen so that when reading bytes as octets in the typical
    first-octet-is-the-high-nibble fashion, an octet-to-binary conversion
    would yield a high-to-low, left-to-right view of the "short words".

    Example:

    Say short words are 5 bits wide. Our word sequence is the values
    ``[31, 1, 5]``. In big-endian binary format, the values are
    ``[0b11111, 0b00001, 0b00101]``. We use 15 of 16 bits in two bytes
    and pad the last (lowest) bit of the last byte with a zero::

        [0b11111000, 0b01001010] = [0xF8, 0x4A]
    """

    # The number of bits per byte.
    BITS_PER_BYTE = 8

    # :var int bits_left_in_byte: Number of bits that remain writable in the current byte.
    # :var int byte_index: Index of byte currently being written to.
    # :var int words_written: Number of words written.

    def __init__(self, word_length, word_count, byte_padding):
        """
        :param int word_length: the length in bits of the words to be serialized. Must
               be greater than or equal to 1 and less than or equal to 64.
        :param int word_count: the number of words to be serialized. Must be greater than
               or equal to zero.
        :param int byte_padding: the number of leading bytes that should pad the
               serialized words. Must be greater than or equal to zero.
        """
        if (word_length < 1) or (word_length > 64):
            raise ValueError('Word length must be >= 1 and <= 64. (was: {})'.format(word_length))
        if word_count < 0:
            raise ValueError('Word count must be >= 0. (was: {})'.format(word_count))
        if byte_padding < 0:
            raise ValueError('Byte padding must be must be >= 0. (was: {})'.format(byte_padding))

        self._word_length = word_length
        self._word_count = word_count

        bits_required = word_length * word_count
        leftover_bits = ((bits_required % self.BITS_PER_BYTE) != 0)
        leftover_bits_inc = 0
        if leftover_bits:
            leftover_bits_inc = 1
        bytes_required = (bits_required / self.BITS_PER_BYTE) + leftover_bits_inc + byte_padding
        self._bytes = [0] * int(bytes_required)

        self._bits_left_in_byte = self.BITS_PER_BYTE
        self._byte_index = byte_padding
        self._words_written = 0

    def write_word(self, word):
        """
        Writes the word to the backing array.

        :param long word: the word to write.
        :rtype: void
        """
        if self._words_written == self._word_count:
            raise ValueError('Cannot write more words, backing array full!')

        bits_left_in_word = self._word_length

        while bits_left_in_word > 0:
            # Move to the next byte if the current one is fully packed.
            if self._bits_left_in_byte == 0:
                self._byte_index += 1
                self._bits_left_in_byte = self.BITS_PER_BYTE

            consumed_mask = ~0 if bits_left_in_word == 64 else (BitUtil.left_shift_long(1, bits_left_in_word) - 1)

            # Fix how many bits will be written in this cycle. Choose the
            #  smaller of the remaining bits in the word or byte.
            number_of_bits_to_write = min(self._bits_left_in_byte, bits_left_in_word)
            bits_in_byte_remaining_after_write = self._bits_left_in_byte - number_of_bits_to_write

            # In general, we write the highest bits of the word first, so we
            # strip the highest bits that were consumed in previous cycles.
            remaining_bits_of_word_to_write = (word & consumed_mask)

            # If the byte can accept all remaining bits, there is no need
            # to shift off the bits that won't be written in this cycle.
            bits_that_the_byte_can_accept = remaining_bits_of_word_to_write

            # If there is more left in the word than can be written to this
            # byte, shift off the bits that can't be written off the bottom.
            if bits_left_in_word > number_of_bits_to_write:
                bits_that_the_byte_can_accept = BitUtil.unsigned_right_shift_long(remaining_bits_of_word_to_write, bits_left_in_word - self._bits_left_in_byte)
            else:
                # If the byte can accept all remaining bits, there is no need
                # to shift off the bits that won't be written in this cycle.
                bits_that_the_byte_can_accept = remaining_bits_of_word_to_write

            # Align the word bits to write up against the byte bits that have
            # already been written. This shift may do nothing if the remainder
            # of the byte is being consumed in this cycle.
            aligned_bits = BitUtil.left_shift_long(bits_that_the_byte_can_accept, bits_in_byte_remaining_after_write)

            # Update the byte with the alignedBits.
            self._bytes[self._byte_index] |= BitUtil.to_signed_byte(aligned_bits)

            # Update state with bit count written.
            bits_left_in_word -= number_of_bits_to_write
            self._bits_left_in_byte = bits_in_byte_remaining_after_write

        self._words_written += 1

    def get_bytes(self):
        """
        Returns the backing array of ``byte``s that contain the serialized words.

        :returns: the serialized words as a list of bytes.
        :rtype: list
        """
        if self._words_written < self._word_count:
            raise ValueError('Not all words have been written! ({}/{})'.format(self._words_written, self._word_count))
        return self._bytes


class HLLMetadata:
    """
    The metadata and parameters associated with a HLL.
    """

    def __init__(self, schema_version, type, register_count_log2, register_width, log2_explicit_cutoff, explicit_off, explicit_auto, sparse_enabled):
        """
        :param int schema_version: the schema version number of the HLL. This must
            be greater than or equal to zero.
        :param HLLType type: the type of the HLL. This cannot be ``None``.
        :param int register_count_log2: the log-base-2 register count parameter for
            probabilistic HLLs. This must be greater than or equal to zero.
        :param int register_width: the register width parameter for probabilistic
            HLLs. This must be greater than or equal to zero.
        :param int log2_explicit_cutoff: the log-base-2 of the explicit cardinality cutoff,
            if it is explicitly defined. (If ``explicit_off`` or ``explicit_auto`` is True then
            this has no meaning.
        :param boolean explicit_off: the flag for 'explicit off'-mode, where the
            ``HLLType.EXPLICIT`` representation is not used. Both this and
            ``explicit_auto`` cannot be True at the same time.
        :param boolean explicit_auto: the flag for 'explicit auto'-mode, where the
            ``HLLType.EXPLICIT`` representation's promotion cutoff is
            determined based on in-memory size automatically. Both this and
            ``explicit_off`` cannot be True at the same time.
        :param boolean sparse_enabled: the flag for 'sparse-enabled'-mode, where the
            ``HLLType.SPARSE`` representation is used.
        """
        self._schema_version = schema_version
        self._type = type
        self._register_count_log2 = register_count_log2
        self._register_width = register_width
        self._log2_explicit_cutoff = log2_explicit_cutoff
        self._explicit_off = explicit_off
        self._explicit_auto = explicit_auto
        self._sparse_enabled = sparse_enabled

    def schema_version(self):
        """
        :returns: the schema version of the HLL. This will never be ``None``.
        :rtype: int
        """
        return self._schema_version

    def hll_type(self):
        """
        :returns: the type of the HLL. This will never be ``None``.
        :rtype: HLLType
        """
        return self._type

    def register_count_log2(self):
        """
        :returns: the log-base-2 of the register count parameter of the HLL. This
                  will always be greater than or equal to 4 and less than or equal
                  to 31.
        :rtype: int
        """
        return self._register_count_log2

    def register_width(self):
        """
        :returns: the register width parameter of the HLL. This will always be
                  greater than or equal to 1 and less than or equal to 8.
        :rtype: int
        """
        return self._register_width

    def log2_explicit_cutoff(self):
        """
        :returns: the log-base-2 of the explicit cutoff cardinality. This will always
                  be greater than or equal to zero and less than 31, per the specification.
        :rtype: int
        """
        return self._log2_explicit_cutoff

    def explicit_off(self):
        """
        :returns: True if the ``HLLType.EXPLICIT`` representation
                  has been disabled. False< otherwise.
        :rtype: boolean
        """
        return self._explicit_off

    def explicit_auto(self):
        """
        :returns: True if the ``HLLType.EXPLICIT`` representation
                  cutoff cardinality is set to be automatically chosen,
                  False otherwise.
        :rtype: boolean
        """
        return self._explicit_auto

    def sparse_enabled(self):
        """
        :returns: True if the HLLType.SPARSE representation is enabled.
        :rtype: boolean
        """
        return self._sparse_enabled

    def __str__(self):
        return "<HLLMetadata schema_version: %s, type: %s, register_count_log2: %s, register_width: %s, log2_explicit_cutoff: %s, explicit_off: %s, explicit_auto: %s>" % (self._schema_version, self._type, self._register_count_log2, self._register_width, self._log2_explicit_cutoff, self._explicit_off, self._explicit_auto)


class SchemaVersionOne:
    """
    A serialization schema for HLLs. Reads and writes HLL metadata to
    and from byte representations.
    """

    # The schema version number for this instance.
    SCHEMA_VERSION = 1

    # Version-specific ordinals (array position) for each of the HLL types
    TYPE_ORDINALS = [
        HLLType.UNDEFINED,
        HLLType.EMPTY,
        HLLType.EXPLICIT,
        HLLType.SPARSE,
        HLLType.FULL
    ]

    # number of header bytes for all HLL types
    HEADER_BYTE_COUNT = 3

    # sentinel values from the spec for explicit off and auto
    EXPLICIT_OFF = 0
    EXPLICIT_AUTO = 63

    def padding_bytes(self, type):
        """
        The number of metadata bytes required for a serialized HLL of the
        specified type.

        :param HLLType type: the type of the serialized HLL
        :returns: the number of padding bytes needed in order to fully accommodate
                  the needed metadata.
        :rtype: int
        """
        return self.HEADER_BYTE_COUNT

    def write_metadata(self, bytes, metadata):
        """
        Writes metadata bytes to serialized HLL.

        :param list bytes: the padded data bytes of the HLL
        :param HLLMetadata metadata: the metadata to write to the padding bytes
        :rtype: void
        """
        type = metadata.hll_type()
        type_ordinal = self._get_ordinal(type)

        explicit_cut_off_value = metadata.log2_explicit_cutoff() + 1

        if metadata.explicit_off():
            explicit_cut_off_value = self.EXPLICIT_OFF
        elif metadata.explicit_auto():
            explicit_cut_off_value = self.EXPLICIT_AUTO

        bytes[0] = SerializationUtil.pack_version_byte(self.SCHEMA_VERSION, type_ordinal)
        bytes[1] = SerializationUtil.pack_parameters_byte(metadata.register_width(), metadata.register_count_log2())
        bytes[2] = SerializationUtil.pack_cutoff_byte(explicit_cut_off_value, metadata.sparse_enabled())

    def read_metadata(self, bytes):
        """
        Reads the metadata bytes of the serialized HLL.

        :param list bytes: the serialized HLL
        :returns: the HLL metadata
        :rtype: HLLMetadata
        """
        version_byte = bytes[0]
        parameters_byte = bytes[1]
        cutoff_byte = bytes[2]

        type_ordinal = SerializationUtil.type_ordinal(version_byte)
        explicit_cut_off_value = SerializationUtil.explicit_cutoff(cutoff_byte)
        explicit_off = (explicit_cut_off_value == self.EXPLICIT_OFF)
        explicit_auto = (explicit_cut_off_value == self.EXPLICIT_AUTO)
        log2_explicit_cutoff = -1 if (explicit_off or explicit_auto) else explicit_cut_off_value - 1

        return HLLMetadata(SchemaVersionOne.SCHEMA_VERSION, self._get_type(type_ordinal), SerializationUtil.register_count_log2(parameters_byte),
                           SerializationUtil.register_width(parameters_byte), log2_explicit_cutoff, explicit_off,
                           explicit_auto, SerializationUtil.sparse_enabled(cutoff_byte))

    def get_serializer(self, type, word_length, word_count):
        """
        Builds an HLL serializer that matches this schema version.

        :param HLLType type: the HLL type that will be serialized. This cannot be ``None``.
        :param int word_length: the length of the 'words' that comprise the data of the
               HLL. Words must be at least 5 bits and at most 64 bits long.
        :param int word_count: the number of 'words' in the HLL's data.
        :returns a byte array serializer used to serialize a HLL according
                 to this schema version's specification.
        :rtype: BigEndianAscendingWordSerializer
        """
        return BigEndianAscendingWordSerializer(word_length, word_count, self.padding_bytes(type))

    def get_deserializer(self, type, word_length, bytes):
        """
        Builds an HLL deserializer that matches this schema version.

        :param HLLType type: the HLL type that will be deserialized. This cannot be ``None``.
        :param int word_length: the length of the 'words' that comprise the data of the
               serialized HLL. Words must be at least 5 bits and at most 64
               bits long.
        :param list bytes: the serialized HLL to deserialize. This cannot be ``None``.
        :returns: a byte array deserializer used to deserialize a HLL serialized
                  according to this schema version's specification.
        :rtype: BigEndianAscendingWordDeserializer
        """
        return BigEndianAscendingWordDeserializer(word_length, self.padding_bytes(type), bytes)

    def schema_version_number(self):
        """
        :returns: the schema version number
        :rtype: int
        """
        return self.SCHEMA_VERSION

    @classmethod
    def _get_ordinal(cls, type):
        """
        Gets the ordinal for the specified ``HLLType``.

        :param HLLType type: the type whose ordinal is desired
        :returns the ordinal for the specified type, to be used in the version byte.
                 This will always be non-negative.
        :rtype: int
        """
        return cls.TYPE_ORDINALS.index(type)

    @classmethod
    def _get_type(cls, ordinal):
        """
        Gets the ``HLLType`` for the specified ordinal.

        :param int ordinal: the ordinal whose type is desired
        :returns: the type for the specified ordinal. This will never be ``None``.
        :rtype: HLLType
        """
        if ordinal < 0 or ordinal >= len(cls.TYPE_ORDINALS):
            raise ValueError('Invalid type ordinal {}. Only 0-{} inclusive allowed'.format(
                ordinal, (len(cls.TYPE_ORDINALS) - 1)))
        return cls.TYPE_ORDINALS[ordinal]


class SerializationUtil:
    """
    A collection of constants and utilities for serializing and deserializing
    HLLs.
    """

    # The number of bits (of the parameters byte) dedicated to encoding the
    # width of the registers.
    REGISTER_WIDTH_BITS = 3

    # A mask to cap the maximum value of the register width.
    REGISTER_WIDTH_MASK = BitUtil.left_shift_int(1, REGISTER_WIDTH_BITS) - 1

    # The number of bits (of the parameters byte) dedicated to encoding
    # ``log2(register_count)``.
    LOG2_REGISTER_COUNT_BITS = 5

    # A mask to cap the maximum value of ``log2(register_count)``.
    LOG2_REGISTER_COUNT_MASK = BitUtil.left_shift_int(1, LOG2_REGISTER_COUNT_BITS) - 1

    # The number of bits (of the cutoff byte) dedicated to encoding the
    # log-base-2 of the explicit cutoff or sentinel values for
    # 'explicit-disabled' or 'auto'.
    EXPLICIT_CUTOFF_BITS = 6

    # A mask to cap the maximum value of the explicit cutoff choice.
    EXPLICIT_CUTOFF_MASK = BitUtil.left_shift_int(1, EXPLICIT_CUTOFF_BITS) - 1

    # Number of bits in a nibble.
    NIBBLE_BITS = 4

    # A mask to cap the maximum value of a nibble.
    NIBBLE_MASK = BitUtil.left_shift_int(1, NIBBLE_BITS) - 1

    # ************************************************************************
    # Serialization utilities

    # Schema version one (v1).
    VERSION_ONE = SchemaVersionOne()

    # The default schema version for serializing HLLs.
    DEFAULT_SCHEMA_VERSION = VERSION_ONE

    # List of registered schema versions, indexed by their version numbers. If
    # an entry is ``None``, then no such schema version is registered.
    # Similarly, registering a new schema version simply entails assigning an
    # SchemaVersion instance to the appropriate index of this array.
    #
    # By default, only SchemaVersionOne is registered. Note that version
    # zero will always be reserved for internal (e.g. proprietary, legacy) schema
    # specifications/implementations and will never be assigned to in by this
    # library.
    REGISTERED_SCHEMA_VERSIONS = [None, VERSION_ONE]

    @classmethod
    def get_schema_version_from_number(cls, schema_version_number):
        """
        :param int schema_version_number: the version number of the ``SchemaVersion``
               desired. This must be a registered schema version number.
        :returns: The ``SchemaVersion`` for the given number. This will never be ``None``.
        :rtype: SchemaVersion
        """
        if schema_version_number >= len(cls.REGISTERED_SCHEMA_VERSIONS) or schema_version_number < 0:
            raise ValueError('Invalid schema version number {}'.format(schema_version_number))
        schema_version = cls.REGISTERED_SCHEMA_VERSIONS[schema_version_number]

        if schema_version is None:
            raise ValueError('Unknown schema version number {}'.format(schema_version_number))
        return schema_version

    @classmethod
    def get_schema_version(cls, bytes):
        """
        Get the appropriate ``SchemaVersion`` for the specified
        serialized HLL.

        :param list bytes: the serialized HLL whose schema version is desired.
        :returns the schema version for the specified HLL. This will never be ``None``.
        :rtype: SchemaVersion
        """
        version_byte = bytes[0]
        schema_version_number = cls.schema_version(version_byte)

        return cls.get_schema_version_from_number(schema_version_number)

    @classmethod
    def pack_version_byte(cls, schema_version, type_ordinal):
        """
        Generates a byte that encodes the schema version and the type ordinal of the HLL.

        The top nibble is the schema version and the bottom nibble is the type ordinal.

        :param int schema_version: the schema version to encode.
        :param int type_ordinal: the type ordinal of the HLL to encode.
        :returns: the packed version byte
        :rtype: byte
        """
        return BitUtil.to_signed_byte(BitUtil.left_shift_int(cls.NIBBLE_MASK & schema_version, cls.NIBBLE_BITS) | (cls.NIBBLE_MASK & type_ordinal))

    @classmethod
    def pack_cutoff_byte(cls, explicit_cutoff, sparse_enabled):
        """
        Generates a byte that encodes the log-base-2 of the explicit cutoff or sentinel values for
        'explicit-disabled' or 'auto', as well as the boolean indicating whether to use ``HLLType.SPARSE``
        in the promotion hierarchy.

        The top bit is always padding, the second highest bit indicates the
        'sparse-enabled' boolean, and the lowest six bits encode the explicit
        cutoff value.

        :param int explicit_cutoff: the explicit cutoff value to encode.
               * If 'explicit-disabled' is chosen, this value should be ``0``.
               * If a cutoff of 2:sup:`n` is desired, for``0 <= n < 31``, this value should be ``n + 1``.
        :param boolean sparse_enabled: whether ``HLLType.SPARSE``
               should be used in the promotion hierarchy to improve HLL
               storage.
        :rtype: byte
        """
        sparse_bit = BitUtil.left_shift_int(1, cls.EXPLICIT_CUTOFF_BITS) if sparse_enabled else 0
        return BitUtil.to_signed_byte(sparse_bit | (cls.EXPLICIT_CUTOFF_MASK & explicit_cutoff))

    @classmethod
    def pack_parameters_byte(cls, register_width, register_count_log2):
        """
        Generates a byte that encodes the parameters of a ``HLLType.FULL`` or ``HLLType.SPARSE`` HLL.

        The top 3 bits are used to encode ``registerWidth - 1``
        (range of ``registerWidth`` is thus 1-9) and the bottom 5
        bits are used to encode ``registerCountLog2``
        (range of ``registerCountLog2`` is thus 0-31).

        :param int register_width: the register width (must be at least 1 and at
               most 9)
        :param int register_count_log2: the log-base-2 of the register count (must
               be at least 0 and at most 31)
        :returns: the packed parameters byte
        :rtype: byte
        """
        width_bits = (register_width - 1) & cls.REGISTER_WIDTH_MASK
        count_bits = register_count_log2 & cls.LOG2_REGISTER_COUNT_MASK
        return BitUtil.to_signed_byte(BitUtil.to_signed_byte(BitUtil.left_shift_int(width_bits, cls.LOG2_REGISTER_COUNT_BITS) | count_bits))

    @classmethod
    def sparse_enabled(cls, cutoff_byte):
        """
        Extracts the 'sparse-enabled' boolean from the cutoff byte of a serialized HLL.

        :param byte cutoff_byte: the cutoff byte of the serialized HLL
        :returns: the 'sparse-enabled' boolean
        :rtype: boolean
        """
        return (BitUtil.unsigned_right_shift_byte(cutoff_byte, cls.EXPLICIT_CUTOFF_BITS) & 1) == 1

    @classmethod
    def explicit_cutoff(cls, cutoff_byte):
        """
        Extracts the explicit cutoff value from the cutoff byte of a serialized HLL.

        :param byte cutoff_byte: the cutoff byte of the serialized HLL
        :returns: the explicit cutoff value
        :rtype: int
        """
        return cutoff_byte & cls.EXPLICIT_CUTOFF_MASK

    @classmethod
    def schema_version(cls, version_byte):
        """
        Extracts the schema version from the version byte of a serialized HLL.

        :param byte version_byte: the version byte of the serialized HLL
        :returns: the schema version of the serialized HLL
        :rtype: int
        """
        return cls.NIBBLE_MASK & BitUtil.unsigned_right_shift_byte(version_byte, cls.NIBBLE_BITS)

    @classmethod
    def type_ordinal(cls, version_byte):
        """
        Extracts the type ordinal from the version byte of a serialized HLL.

        :param byte version_byte: the version byte of the serialized HLL
        :returns: the type ordinal of the serialized HLL
        :rtype: int
        """
        return version_byte & cls.NIBBLE_MASK

    @classmethod
    def register_width(cls, parameters_byte):
        """
        Extracts the register width from the parameters byte of a serialized ``HLLType.FULL`` HLL.

        :param byte parameters_byte: the parameters byte of the serialized HLL
        :returns: the register width of the serialized HLL
        :rtype: int
        """
        return (BitUtil.unsigned_right_shift_byte(parameters_byte, cls.LOG2_REGISTER_COUNT_BITS) & cls.REGISTER_WIDTH_MASK) + 1

    @classmethod
    def register_count_log2(cls, parameters_byte):
        """
        Extracts the log2(register_count) from the parameters byte of a serialized ``HLLType.FULL`` HLL.

        :param byte parameters_byte: the parameters byte of the serialized HLL
        :returns: log2(registerCount) of the serialized HLL
        :rtype: int
        """
        return parameters_byte & cls.LOG2_REGISTER_COUNT_MASK
