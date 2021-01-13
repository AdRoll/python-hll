# -*- coding: utf-8 -*-

from __future__ import division
from copy import deepcopy
from math import ceil, floor

from python_hll.hlltype import HLLType
from python_hll.serialization import SerializationUtil, HLLMetadata
from python_hll.util import NumberUtil, BitVector, BitUtil


class HLL:
    """
    A probabilistic set of hashed ``long`` elements. Useful for computing
    the approximate cardinality of a stream of data in very small storage.

    A modified version of the `'HyperLogLog' data structure and algorithm
    <http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>`_ is used,
    which combines both probabilistic and non-probabilistic techniques to
    improve the accuracy and storage requirements of the original algorithm.

    More specifically, initializing and storing a new HLL will
    allocate a sentinel value symbolizing the empty set (HLLType.EMPTY).
    After adding the first few values, a sorted list of unique integers is
    stored in a HLLType.EXPLICIT hash set. When configured, accuracy can
    be sacrificed for memory footprint: the values in the sorted list are
    "promoted" to a "HLLType.SPARSE" map-based HyperLogLog structure.
    Finally, when enough registers are set, the map-based HLL will be converted
    to a bit-packed "HLLType.FULL" HyperLogLog structure.

    This data structure is interoperable with the implementations found at:

    * `postgresql-hll <https://github.com/aggregateknowledge/postgresql-hll>`_
    * `js-hll <https://github.com/aggregateknowledge/js-hll>`_

    when `properly serialized <https://github.com/aggregateknowledge/postgresql-hll/blob/master/STORAGE.markdown>`_.
    """

    # minimum and maximum values for the log-base-2 of the number of registers
    # in the HLL
    MINIMUM_LOG2M_PARAM = 4
    MAXIMUM_LOG2M_PARAM = 30

    # minimum and maximum values for the register width of the HLL
    MINIMUM_REGWIDTH_PARAM = 1
    MAXIMUM_REGWIDTH_PARAM = 8

    # minimum and maximum values for the 'expthresh' parameter of the
    # constructor that is meant to match the PostgreSQL implementation's
    # constructor and parameter names
    MINIMUM_EXPTHRESH_PARAM = -1
    MAXIMUM_EXPTHRESH_PARAM = 18
    MAXIMUM_EXPLICIT_THRESHOLD = BitUtil.left_shift_int(1, (MAXIMUM_EXPTHRESH_PARAM - 1))  # per storage spec

    # ------------------------------------------------------------
    # STORAGE
    # :var set _explicit_storage: storage used when ``type`` is EXPLICIT, None otherwise
    # :var dict _sparse_probabilistic_storage: storage used when ``type`` is SPARSE, None otherwise
    # :var BitVector _probabilistic_storage: storage used when ``type`` is FULL, None otherwise
    # :var HLLType type: current type of this HLL instance, if this changes then so should the storage used (see above)

    # ------------------------------------------------------------
    # CHARACTERISTIC PARAMETERS
    # NOTE:  These members are named to match the PostgreSQL implementation's parameters.
    # :var int _log2m: log2(the number of probabilistic HLL registers)
    # :var int _regwidth: the size (width) each register in bits

    # ------------------------------------------------------------
    # COMPUTED CONSTANTS
    # ............................................................
    # EXPLICIT-specific constants
    # :var boolean _explicit_off: flag indicating if the EXPLICIT representation should NOT be used
    # :var boolean _explicit_auto: flag indicating that the promotion threshold from EXPLICIT should be
    #              computed automatically. NOTE:  this only has meaning when '_explicit_off' is false.
    # :var int _explicit_threshold: threshold (in element count) at which a EXPLICIT HLL is converted to a
    #           SPARSE or FULL HLL, always greater than or equal to zero and always a power of two OR simply zero
    #           NOTE:  this only has meaning when '_explicit_off' is false
    # ............................................................
    # SPARSE-specific constants
    # :var int _short_word_length: the computed width of the short words
    # :var boolean _sparse_off: flag indicating if the SPARSE representation should not be used
    # :var int _sparse_threshold: threshold (in register count) at which a SPARSE HLL is converted to a
    #          FULL HLL, always greater than zero
    # ............................................................
    # Probabilistic algorithm constants
    # :var int _m: the number of registers, will always be a power of 2
    # :var int _m_bits_mask: a mask of the log2m bits set to one and the rest to zero
    # :var int _value_mask: a mask as wide as a register (see ``from_bytes()``)
    # :var long _long_pw_mask: mask used to ensure that p(w) does not overflow register (see ``__init__()`` and ``add_raw()``)
    # ;var float _alpha_m_squared: alpha * m^2 (the constant in the "'raw' HyperLogLog estimator")
    # :var float _small_estimator_cutoff: the cutoff value of the estimator for using the "small" range cardinality correction formula
    # :var float _large_estimator_cutoff: the cutoff value of the estimator for using the "large" range cardinality correction formula

    def __init__(self, log2m, regwidth, expthresh=-1, sparseon=True, type=HLLType.EMPTY):
        """
        NOTE: Arguments here are named and structured identically to those in the
              PostgreSQL implementation, which can be found
              `here <https://github.com/aggregateknowledge/postgresql-hll/blob/master/README.markdown#explanation-of-parameters-and-tuning>`_.

        :param log2m: log-base-2 of the number of registers used in the HyperLogLog
               algorithm. Must be at least 4 and at most 30.
        :type log2m: int
        :param regwidth: number of bits used per register in the HyperLogLog
               algorithm. Must be at least 1 and at most 8.
        :type regwidth: int
        :param expthresh: tunes when the ``HLLType.EXPLICIT`` to
               ``HLLType.SPARSE`` promotion occurs,
               based on the set's cardinality. Must be at least -1 and at most 18.
               +-----------+--------------------------------------------------------------------------------+
               | expthresh | Meaning                                                                        |
               +===========+================================================================================+
               | -1        | Promote at whatever cutoff makes sense for optimal memory usage. ('auto' mode) |
               +-----------+--------------------------------------------------------------------------------+
               | 0         | Skip ``EXPLICIT`` representation in hierarchy.                                 |
               +-----------+--------------------------------------------------------------------------------+
               | 1-18      | Promote at 2:sup:`expthresh - 1` cardinality                                   |
               +-----------+--------------------------------------------------------------------------------+
        :type expthresh: int
        :param sparseon: Flag indicating if the ``HLLType.SPARSE``
               representation should be used.
        :type sparseon: boolean
        :param type: the type in the promotion hierarchy which this instance should
               start at. This cannot be ``None``.
        :type type: HLLType
        """
        from python_hll.hllutil import HLLUtil

        self._log2m = log2m
        if log2m < HLL.MINIMUM_LOG2M_PARAM or log2m > HLL.MAXIMUM_EXPLICIT_THRESHOLD:
            raise Exception("'log2m' must be at least " + str(HLL.MINIMUM_LOG2M_PARAM) + " and at most " + str(HLL.MAXIMUM_LOG2M_PARAM) + " (was: " + str(log2m) + ")")

        self._regwidth = regwidth
        if regwidth < HLL.MINIMUM_REGWIDTH_PARAM or regwidth > HLL.MAXIMUM_REGWIDTH_PARAM:
            raise Exception("'regwidth' must be at least " + str(HLL.MINIMUM_REGWIDTH_PARAM) + " and at most " + str(HLL.MAXIMUM_REGWIDTH_PARAM) + " (was: " + str(regwidth) + ")")

        self._m = BitUtil.left_shift_int(1, log2m)
        self._m_bits_mask = self._m - 1
        self._value_mask = BitUtil.left_shift_int(1, regwidth) - 1
        self._pw_max_mask = HLLUtil.pw_max_mask(regwidth)
        self._alpha_m_squared = HLLUtil.alpha_m_squared(self._m)
        self._small_estimator_cutoff = HLLUtil.small_estimator_cutoff(self._m)
        self._large_estimator_cutoff = HLLUtil.large_estimator_cutoff(log2m, regwidth)

        if expthresh == -1:
            self._explicit_auto = True
            self._explicit_off = False

            # NOTE:  This math matches the size calculation in the PostgreSQL impl.
            full_representation_size = floor((self._regwidth * self._m + 7) / 8)  # round up to next whole byte
            num_longs = floor(full_representation_size / 8)  # integer division to round down

            if num_longs > HLL.MAXIMUM_EXPLICIT_THRESHOLD:
                self._explicit_threshold = HLL.MAXIMUM_EXPLICIT_THRESHOLD
            else:
                self._explicit_threshold = num_longs
        elif expthresh == 0:
            self._explicit_auto = False
            self._explicit_off = True
            self._explicit_threshold = 0
        elif 0 < expthresh <= HLL.MAXIMUM_EXPTHRESH_PARAM:
            self._explicit_auto = False
            self._explicit_off = False
            self._explicit_threshold = BitUtil.left_shift_int(1, (expthresh - 1))
        else:
            raise Exception("'expthresh' must be at least " + str(HLL.MINIMUM_EXPTHRESH_PARAM) + " and at most " + str(HLL.MAXIMUM_EXPTHRESH_PARAM) + " (was: " + str(expthresh) + ")")

        self._short_word_length = regwidth + log2m
        self._sparse_off = not sparseon
        if self._sparse_off:
            self._sparse_threshold = 0
        else:
            # TODO improve this cutoff to include the cost overhead of members/objects
            largest_pow_2_less_than_cutoff = int(NumberUtil.log2((self._m * self._regwidth) / self._short_word_length))
            self._sparse_threshold = BitUtil.left_shift_int(1, largest_pow_2_less_than_cutoff)

        self._initialize_storage(type)

    @classmethod
    def create_for_testing(cls, log2m, regwidth, explicit_threshold, sparse_threshold, type):
        """
        Convenience constructor for testing. Assumes that both ``HLLType.EXPLICIT``
        and ``HLLType.SPARSE`` representations should be enabled.

        :param log2m: log-base-2 of the number of registers used in the HyperLogLog
               algorithm. Must be at least 4 and at most 30.
        :type log2m: int
        :param regwidth: number of bits used per register in the HyperLogLog
               algorithm. Must be at least 1 and at most 8.
        :type regwidth: int
        :param explicit_threshold: cardinality threshold at which the ``HLLType.EXPLICIT``
               representation should be promoted to ``HLLType.SPARSE``.
               This must be greater than zero and less than or equal to ``MAXIMUM_EXPLICIT_THRESHOLD``.
        :type explicit_threshold: int
        :param sparse_threshold: register count threshold at which the ``HLLType.SPARSE``
               representation should be promoted to ``HLLType.FULL``.
               This must be greater than zero.
        :type sparse_threshold: int
        :param type: the type in the promotion hierarchy which this instance should
               start at. This cannot be ``None``.
        :type type: HLLType
        :rtype: HLL
        """
        hll = HLL(log2m=log2m, regwidth=regwidth, expthresh=-1, sparseon=True, type=type)
        hll._explicit_auto = False
        hll._explicit_off = False
        hll._explicit_threshold = explicit_threshold
        if explicit_threshold < 1 or explicit_threshold > cls.MAXIMUM_EXPLICIT_THRESHOLD:
            raise Exception("'explicit_threshold' must be at least 1 and at most " + str(cls.MAXIMUM_EXPLICIT_THRESHOLD) + " (was: " + str(explicit_threshold) + ")")
        hll._sparse_off = False
        hll._sparse_threshold = sparse_threshold
        return hll

    def get_type(self):
        """
        Returns the type in the promotion hierarchy of this instance. This will
        never be ``None``.

        :rtype: HLLType
        """
        return self._type

    def add_raw(self, raw_value):
        """
        Adds ``rawValue`` directly to the HLL.

        :param long raw_value: the value to be added. It is very important that this
               value already be hashed with a strong (but not
               necessarily cryptographic) hash function. For instance, the
               `MurmurHash3 implementation <https://pypi.org/project/mmh3/>`_
               is an excellent hash function for this purpose.
        :rtype: void
        """

        if self._type == HLLType.EMPTY:
            # Note: EMPTY type is always promoted on add_raw()
            if self._explicit_threshold > 0:
                self._initialize_storage(HLLType.EXPLICIT)
                self._explicit_storage.add(raw_value)
            elif not self._sparse_off:
                self._initialize_storage(HLLType.SPARSE)
                self._add_raw_sparse_probabilistic(raw_value)
            else:
                self._initialize_storage(HLLType.FULL)
                self._add_raw_probabilistic(raw_value)
            return

        elif self._type == HLLType.EXPLICIT:
            self._explicit_storage.add(raw_value)

            # promotion, if necessary
            if len(self._explicit_storage) > self._explicit_threshold:
                if not self._sparse_off:
                    self._initialize_storage(HLLType.SPARSE)
                    for value in self._explicit_storage:
                        self._add_raw_sparse_probabilistic(value)
                else:
                    self._initialize_storage(HLLType.FULL)
                    for value in self._explicit_storage:
                        self._add_raw_probabilistic(value)
                self._explicit_storage = None
            return

        elif self._type == HLLType.SPARSE:
            self._add_raw_sparse_probabilistic(raw_value)

            # promotion, if necessary
            if len(self._sparse_probabilistic_storage) > self._sparse_threshold:
                self._initialize_storage(HLLType.FULL)
                for register_index in self._sparse_probabilistic_storage.keys():
                    register_value = self._sparse_probabilistic_storage.get(register_index, 0)
                    self._probabilistic_storage.set_max_register(register_index, register_value)
                self._sparse_probabilistic_storage = None
            return

        elif self._type == HLLType.FULL:
            self._add_raw_probabilistic(raw_value)
            return

        else:
            raise Exception("Unsupported HLL type: {}".format(self._type))

    def _add_raw_sparse_probabilistic(self, raw_value):
        """
        Adds the raw value to the ``sparseProbabilisticStorage``.
        ``type`` ``HLLType.SPARSE``.

        :param long raw_value: the raw value to add to the sparse storage.
        :rtype: void
        """

        # p(w): position of the least significant set bit (one-indexed)
        # By contract: p(w) <= 2^(register_value_in_bits) - 1 (the max register value)
        #
        # By construction of pw_max_mask (see constructor),
        #      lsb(pw_max_mask) = 2^(register_value_in_bits) - 2,
        # thus lsb(any_long | pw_max_mask) <= 2^(register_value_in_bits) - 2,
        # thus 1 + lsb(any_long | pw_max_mask) <= 2^(register_value_in_bits) -1.
        sub_stream_value = BitUtil.unsigned_right_shift_long(raw_value, self._log2m)
        p_w = None

        if sub_stream_value == 0:
            # The paper does not cover p(0x0), so the special value 0 is used.
            # 0 is the original initialization value of the registers, so by
            # doing this the multiset simply ignores it. This is acceptable
            # because the probability is 1/(2^(2^register_size_in_bits)).
            p_w = 0
        else:
            p_w = BitUtil.to_signed_byte(1 + BitUtil.least_significant_bit(sub_stream_value | self._pw_max_mask))

        # Short-circuit if the register is being set to zero, since algorithmically
        # this corresponds to an "unset" register, and "unset" registers aren't
        # stored to save memory. (The very reason this sparse implementation
        # exists.) If a register is set to zero it will break the algorithm_cardinality
        # code.
        if p_w == 0:
            return

        # NOTE:  no +1 as in paper since 0-based indexing
        j = int(raw_value & self._m_bits_mask)

        current_value = self._sparse_probabilistic_storage.get(j, 0)
        if p_w > current_value:
            self._sparse_probabilistic_storage[j] = p_w

    def _add_raw_probabilistic(self, raw_value):
        """
        Adds the raw value to the ``probabilisticStorage``.
        ``type`` must be ``HLLType.FULL``.

        :param long raw_value: the raw value to add to the full probabilistic storage.
        :rtype: void
        """
        # p(w): position of the least significant set bit (one-indexed)
        # By contract: p(w) <= 2^(register_value_in_bits) - 1 (the max register value)
        #
        # By construction of pw_max_mask (see constructor),
        #      lsb(pw_max_mask) = 2^(register_value_in_bits) - 2,
        # thus lsb(any_long | pw_max_mask) <= 2^(register_value_in_bits) - 2,
        # thus 1 + lsb(any_long | pw_max_mask) <= 2^(register_value_in_bits) -1.
        sub_stream_value = BitUtil.unsigned_right_shift_long(raw_value, self._log2m)
        p_w = None

        if sub_stream_value == 0:
            # The paper does not cover p(0x0), so the special value 0 is used.
            # 0 is the original initialization value of the registers, so by
            # doing this the multiset simply ignores it. This is acceptable
            # because the probability is 1/(2^(2^register_size_in_bits)).
            p_w = 0
        else:
            p_w = BitUtil.to_signed_byte(1 + BitUtil.least_significant_bit(sub_stream_value | self._pw_max_mask))

        # Short-circuit if the register is being set to zero, since algorithmically
        # this corresponds to an "unset" register, and "unset" registers aren't
        # stored to save memory. (The very reason this sparse implementation
        # exists.) If a register is set to zero it will break the algorithm_cardinality
        # code.
        if p_w == 0:
            return

        # NOTE:  no +1 as in paper since 0-based indexing
        j = int(raw_value & self._m_bits_mask)

        self._probabilistic_storage.set_max_register(j, p_w)

    def _initialize_storage(self, type):
        """
        Initializes storage for the specified ``HLLType`` and changes the
        instance's ``type``.

        :param HLLType type: the ``HLLType`` to initialize storage for. This cannot be
               ``None`` and must be an instantiable type. (For instance,
               it cannot be ``HLLType.UNDEFINED``.)
        :rtype: void
        """
        self._type = type
        if type == HLLType.EMPTY:
            # nothing to be done
            pass
        elif type == HLLType.EXPLICIT:
            self._explicit_storage = set()
        elif type == HLLType.SPARSE:
            self._sparse_probabilistic_storage = dict()
        elif type == HLLType.FULL:
            self._probabilistic_storage = BitVector(self._regwidth, self._m)
        else:
            raise Exception("Unsupported HLL type: {}".format(self._type))

    def cardinality(self):
        """
        Computes the cardinality of the HLL.

        :returns: the cardinality of HLL. This will never be negative.
        :rtype: long
        """
        if self._type == HLLType.EMPTY:
            return 0  # by definition
        elif self._type == HLLType.EXPLICIT:
            return len(self._explicit_storage)
        elif self._type == HLLType.SPARSE:
            return ceil(self._sparse_probabilistic_algorithm_cardinality())
        elif self._type == HLLType.FULL:
            return ceil(self._full_probabilistic_algorithm_cardinality())
        else:
            raise Exception("Unsupported HLL type: {}".format(self._type))

    def _sparse_probabilistic_algorithm_cardinality(self):
        """
        Computes the exact cardinality value returned by the HLL algorithm when
        represented as a ``HLLType.SPARSE`` HLL. Kept
        separate from ``cardinality()`` for testing purposes. ``type``
        must be ``HLLType.SPARSE``.

        :returns: the exact, unrounded cardinality given by the HLL algorithm
        :rtype: float
        """
        from python_hll.hllutil import HLLUtil
        m = self._m

        # compute the "indicator function" -- sum(2^(-M[j])) where M[j] is the
        # 'j'th register value
        indicator_function = 0.0
        number_of_zeroes = 0  # "V" in the paper
        for j in range(m):
            register = self._sparse_probabilistic_storage.get(j, 0)

            indicator_function += 1.0 / BitUtil.left_shift_long(1, register)
            if register == 0:
                number_of_zeroes += 1

        # apply the estimate and correction to the indicator function
        estimator = self._alpha_m_squared / indicator_function
        if number_of_zeroes != 0 and estimator < self._small_estimator_cutoff:
            return HLLUtil.small_estimator(m, number_of_zeroes)
        elif estimator <= self._large_estimator_cutoff:
            return estimator
        else:
            return HLLUtil.large_estimator(self._log2m, self._regwidth, estimator)

    def _full_probabilistic_algorithm_cardinality(self):
        """
        Computes the exact cardinality value returned by the HLL algorithm when
        represented as a ``HLLType.FULL`` HLL. Kept separate from ``cardinality()`` for testing purposes.
        type must be ``HLLType.FULL``.

        :rtype: float
        """
        from python_hll.hllutil import HLLUtil
        # for performance
        m = self._m
        # compute the "indicator function" -- sum(2^(-M[j])) where M[j] is the
        # 'j'th register value
        sum = 0
        number_of_zeroes = 0  # "V" in the paper
        iterator = self._probabilistic_storage.register_iterator()
        for register in iterator:
            sum += 1.0 / BitUtil.left_shift_long(1, register)
            if register == 0:
                number_of_zeroes += 1
        # apply the estimate and correction to the indicator function
        estimator = self._alpha_m_squared / sum
        if number_of_zeroes != 0 and (estimator < self._small_estimator_cutoff):
            return HLLUtil.small_estimator(m, number_of_zeroes)
        elif estimator <= self._large_estimator_cutoff:
            return estimator
        else:
            return HLLUtil.large_estimator(self._log2m, self._regwidth, estimator)

    def clear(self):
        """
        Clears the HLL. The HLL will have cardinality zero and will act as if no
        elements have been added.

        NOTE: Unlike ``addRaw(long)``, ``clear`` does NOT handle
        transitions between ``HLLType``'s - a probabilistic type will remain
        probabilistic after being cleared.

        :rtype: void
        """
        if self._type == HLLType.EMPTY:
            return  # do nothing
        elif self._type == HLLType.EXPLICIT:
            return self._explicit_storage.clear()
        elif self._type == HLLType.SPARSE:
            return self._sparse_probabilistic_storage.clear()
        elif self._type == HLLType.FULL:
            self._probabilistic_storage.fill(0)
            return
        else:
            raise Exception('Unsupported HLL type: {}'.format(self._type))

    def union(self, other):
        """
        Computes the union of HLLs and stores the result in this instance.

        :param HLL other: the other ``HLL`` instance to union into this one. This
               cannot be ``None``.
        :rtype: void
        """
        # TODO: verify HLL compatibility
        other_type = other.get_type()

        if self._type == other_type:
            self._homogeneous_union(other)
        else:
            self._heterogenous_union(other)

    def _heterogeneous_union_for_empty_hll(self, other):
        # The union of empty with non-empty HLL is just a clone of the non-empty.

        if other.get_type() == HLLType.EXPLICIT:
            # src: EXPLICIT
            # dest: EMPTY

            if len(other._explicit_storage) <= self._explicit_threshold:
                self._type = HLLType.EXPLICIT
                self._explicit_storage = deepcopy(other._explicit_storage)
            else:
                if not self._sparse_off:
                    self._initialize_storage(HLLType.SPARSE)
                else:
                    self._initialize_storage(HLLType.FULL)

                for value in other._explicit_storage:
                    self.add_raw(value)

        elif other.get_type() == HLLType.SPARSE:
            # src: SPARSE
            # dest: EMPTY

            if not self._sparse_off:
                self._type = HLLType.SPARSE
                self._sparse_probabilistic_storage = deepcopy(other._sparse_probabilistic_storage)
            else:
                self._initialize_storage(HLLType.FULL)
                for register_index in other._sparse_probabilistic_storage.keys():
                    register_value = other._sparse_probabilistic_storage.get(register_index)
                    self._probabilistic_storage.set_max_register(register_index, register_value)
            return

        else:  # case FULL
            # src: FULL
            # dest: EMPTY
            self._type = HLLType.FULL
            self._probabilistic_storage = deepcopy(other._probabilistic_storage)
            return

    def _heterogeneous_union_for_non_empty_hll(self, other):
        if self._type == HLLType.EXPLICIT:
            # src:  FULL/SPARSE
            # dest: EXPLICIT
            # "Storing into destination" cannot be done (since destination
            # is by definition of smaller capacity than source), so a clone
            # of source is made and values from destination are inserted
            # into that.

            # Determine source and destination storage.
            # NOTE:  destination storage may change through promotion if
            #        source is SPARSE.

            if other.get_type() == HLLType.SPARSE:
                if not self._sparse_off:
                    self._type = HLLType.SPARSE
                    self._sparse_probabilistic_storage = deepcopy(other._sparse_probabilistic_storage)
                else:
                    self._initialize_storage(HLLType.FULL)
                    for register_index in other._sparse_probabilistic_storage.keys():
                        register_value = other._sparse_probabilistic_storage.get(register_index)
                        self._probabilistic_storage.set_max_register(register_index, register_value)

            else:  # source is HLLType.FULL
                self._type = HLLType.FULL
                self._probabilistic_storage = deepcopy(other._probabilistic_storage)

            for value in self._explicit_storage:
                self.add_raw(value)
            self._explicit_storage = None
            return

        elif self._type == HLLType.SPARSE:
            if other.get_type() == HLLType.EXPLICIT:
                # src: EXPLICIT
                # dest: SPARSE
                # Add the raw values from the source to the destination.

                for value in other._explicit_storage:
                    # NOTE: add_raw will handle promotion cleanup
                    self.add_raw(value)

            else:  # source is HLLType.FULL
                # src:  FULL
                # dest: SPARSE
                # "Storing into destination" cannot be done (since destination
                # is by definition of smaller capacity than source), so a
                # clone of source is made and registers from the destination
                # are merged into the clone.

                self._type = HLLType.FULL
                self._probabilistic_storage = deepcopy(other._probabilistic_storage)
                for register_index in self._sparse_probabilistic_storage.keys():
                    register_value = self._sparse_probabilistic_storage.get(register_index, 0)
                    self._probabilistic_storage.set_max_register(register_index, register_value)
                self._sparse_probabilistic_storage = None

        else:  # destination is HLLType.FULL
            if other._type == HLLType.EXPLICIT:
                # src: EXPLICIT
                # dest: FULL
                # Add the raw values from the source to the destination.
                # Promotion is not possible, so don't bother checking.

                for value in other._explicit_storage:
                    self.add_raw(value)

            else:  # source is HLLType.SPARSE
                # src: SPARSE
                # dest: FULL
                # Merge the registers from the source into the destination.
                # Promotion is not possible, so don't bother checking.

                for register_index in other._sparse_probabilistic_storage.keys():
                    register_value = other._sparse_probabilistic_storage.get(register_index)
                    self._probabilistic_storage.set_max_register(register_index, register_value)

    def _heterogenous_union(self, other):
        """
        The logic here is divided into two sections: unions with an EMPTY
        HLL, and unions between EXPLICIT/SPARSE/FULL HLL.

        Between those two sections, all possible heterogeneous unions are
        covered. Should another type be added to HLLType whose unions
        are not easily reduced (say, as EMPTY's are below) this may be more
        easily implemented as Strategies. However, that is unnecessary as it
        stands.
        :type other: HLL
        :rtype: void
        """

        # Union with an EMPTY
        if self._type == HLLType.EMPTY:
            self._heterogeneous_union_for_empty_hll(other)
            return
        elif other.get_type() == HLLType.EMPTY:
            # source is empty, so just return destination since it is unchanged
            return

        # else -- both of the sets are not empty
        self._heterogeneous_union_for_non_empty_hll(other)

    def _homogeneous_union(self, other):
        """
        Computes the union of two HLLs of the same type, and stores the
        result in this instance.

        :param HLL other: the other ``HLL`` instance to union into this one. This
               cannot be ``None``.
        :rtype: void
        """
        if self._type == HLLType.EMPTY:
            # union of empty and empty is empty
            return

        elif self._type == HLLType.EXPLICIT:
            for value in other._explicit_storage:
                # Note: add_raw() will handle promotion, if necessary
                self.add_raw(value)

        elif self._type == HLLType.SPARSE:

            for register_index in other._sparse_probabilistic_storage.keys():
                register_value = other._sparse_probabilistic_storage.get(register_index)
                current_register_value = self._sparse_probabilistic_storage.get(register_index, 0)
                if register_value > current_register_value:
                    self._sparse_probabilistic_storage[register_index] = register_value

            # promotion, if necessary
            if len(self._sparse_probabilistic_storage) > self._sparse_threshold:
                self._initialize_storage(HLLType.FULL)
                for register_index in self._sparse_probabilistic_storage.keys():
                    register_value = self._sparse_probabilistic_storage.get(register_index, 0)
                    self._probabilistic_storage.set_max_register(register_index, register_value)

                self._sparse_probabilistic_storage = None

        elif self._type == HLLType.FULL:
            for i in range(self._m):
                register_value = other._probabilistic_storage.get_register(i)
                self._probabilistic_storage.set_max_register(i, register_value)
            return

        else:
            raise Exception('Unsupported HLL type: {}'.format(self._type))

    def to_bytes(self, schema_version=SerializationUtil.DEFAULT_SCHEMA_VERSION):
        """
        Serializes the HLL to an array of bytes in correspondence with the format
        of the default schema version, ``SerializationUtil.DEFAULT_SCHEMA_VERSION``.

        :param SchemaVersion schema_version: the schema version dictating the serialization format
        :returns: the array of bytes representing the HLL. This will never be
                  ``None`` or empty.
        :rtype: list
        """
        from python_hll.hllutil import HLLUtil
        if self._type == HLLType.EMPTY:
            byte_array_length = schema_version.padding_bytes(self._type)
            byte_array = [0] * byte_array_length

        elif self._type == HLLType.EXPLICIT:
            serializer = schema_version.get_serializer(
                self._type,
                HLLUtil.LONG_BIT_LENGTH,
                len(self._explicit_storage)
            )

            values = list(self._explicit_storage)
            values = sorted(values)
            for value in values:
                serializer.write_word(value)

            byte_array = serializer.get_bytes()

        elif self._type == HLLType.SPARSE:
            serializer = schema_version.get_serializer(
                self._type,
                self._short_word_length,
                len(self._sparse_probabilistic_storage)
            )

            indices = self._sparse_probabilistic_storage.keys()
            indices = sorted(indices)

            for register_index in indices:
                register_value = self._sparse_probabilistic_storage.get(register_index, 0)

                # pack index and value into "short word"
                short_word = BitUtil.left_shift_int(register_index, self._regwidth) | register_value
                serializer.write_word(short_word)

            byte_array = serializer.get_bytes()

        elif self._type == HLLType.FULL:
            serializer = schema_version.get_serializer(self._type, self._regwidth, self._m)
            self._probabilistic_storage.get_register_contents(serializer)

            byte_array = serializer.get_bytes()

        else:
            raise Exception('Unsupported HLL type: {}'.format(self._type))

        # no use of it if any _explicit_off or _explicit_auto is true
        log2_explicit_threshold = 0
        if not self._explicit_auto | self._explicit_off:
            log2_explicit_threshold = int(NumberUtil.log2(self._explicit_threshold))

        metadata = HLLMetadata(
            schema_version.schema_version_number(),
            self._type,
            self._log2m,
            self._regwidth,
            log2_explicit_threshold,
            self._explicit_off,
            self._explicit_auto,
            not self._sparse_off
        )
        schema_version.write_metadata(byte_array, metadata)

        return byte_array

    @classmethod
    def from_bytes(cls, bytes):
        """
        Deserializes the HLL (in ``toBytes()`` format) serialized
        into ``bytes``.

        :param list bytes: the serialized bytes of new HLL
        :returns: the deserialized HLL. This will never be ``None``.
        :rtype: HLL
        """
        from python_hll.hllutil import HLLUtil
        schema_version = SerializationUtil.get_schema_version(bytes)
        metadata = schema_version.read_metadata(bytes)

        type = metadata.hll_type()
        reg_width = metadata.register_width()
        log_2m = metadata.register_count_log2()
        sparseon = metadata.sparse_enabled()

        expthresh = 0
        if metadata.explicit_auto():
            expthresh = -1
        elif metadata.explicit_off():
            expthresh = 0
        else:
            # NOTE: take into account that the postgres-compatible constructor
            # subtracts one before taking a power of two.
            expthresh = metadata.log2_explicit_cutoff() + 1

        hll = HLL(log_2m, reg_width, expthresh, sparseon, type)

        # Short-circuit on empty, which needs no other deserialization.
        if type == HLLType.EMPTY:
            return hll

        word_length = 0
        if type == HLLType.EXPLICIT:
            word_length = HLLUtil.LONG_BIT_LENGTH  # 64 for both java and python

        elif type == HLLType.SPARSE:
            word_length = hll._short_word_length

        elif type == HLLType.FULL:
            word_length = hll._regwidth

        else:
            raise Exception('Unsupported HLL type: {}'.format(type))

        deserializer = schema_version.get_deserializer(type, word_length, bytes)
        if type == HLLType.EXPLICIT:
            # NOTE:  This should not exceed expthresh and this will always
            #        be exactly the number of words that were encoded,
            #        because the word length is at least a byte wide.
            # SEE:   BigEndianAscendingWordDeserializer.total_word_count()
            for i in range(deserializer.total_word_count()):
                hll._explicit_storage.add(deserializer.read_word())

        elif type == HLLType.SPARSE:
            # NOTE:  If the short_word_length were smaller than 8 bits
            #        (1 byte) there would be a possibility (because of
            #        padding arithmetic) of having one or more extra
            #        registers read. However, this is not relevant as the
            #        extra registers will be all zeroes, which are ignored
            #        in the sparse representation.
            for i in range(deserializer.total_word_count()):
                short_word = deserializer.read_word()

                register_value = BitUtil.to_signed_byte(short_word & hll._value_mask)
                # Only set non-zero registers.
                if register_value != 0:
                    register_key = int(BitUtil.unsigned_right_shift_long(short_word, hll._regwidth))
                    hll._sparse_probabilistic_storage[register_key] = register_value

        elif type == HLLType.FULL:
            # NOTE:  Iteration is done using m (register count) and NOT
            #        deserializer.total_word_count() because regwidth may be
            #        less than 8 and as such the padding on the 'last' byte
            #        may be larger than regwidth, causing an extra register
            #        to be read.
            # SEE: BigEndianAscendingWordDeserializer.total_word_count()
            for i in range(hll._m):
                hll._probabilistic_storage.set_register(i, deserializer.read_word())

        else:
            raise Exception('Unsupported HLL type: {}'.format(type))

        return hll
