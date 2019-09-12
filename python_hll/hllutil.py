# -*- coding: utf-8 -*-
from math import log
from python_hll.hll import HLL
from python_hll.util import NumberUtil
from python_hll.util import BitUtil


class HLLUtil:
    """
    Static functions for computing constants and parameters used in the HLL
    algorithm.
    """

    # The number of bits used to represent a long value in two's complement binary form
    LONG_BIT_LENGTH = 64

    # Precomputed ``pw_max_mask`` values indexed by ``register_size_in_bits``.
    # Calculated with this formula::
    #
    #     int max_register_value = (1 << register_size_in_bits) - 1;
    #     // Mask with all bits set except for (max_register_value - 1) least significant bits (see add_raw())
    #     return ~((1L << (max_register_value - 1)) - 1);
    #
    # See ``pw_max_mask()``

    PW_MASK = [
        -9223372036854775808,  # ~((1 << (((1 << 0) - 1) - 1)) - 1)
        -1,                    # ~((1 << (((1 << 1) - 1) - 1)) - 1)
        -4,                    # ~((1 << (((1 << 2) - 1) - 1)) - 1)
        -64,                   # ~((1 << (((1 << 3) - 1) - 1)) - 1)
        -16384,                # ~((1 << (((1 << 4) - 1) - 1)) - 1)
        -1073741824,           # ~((1 << (((1 << 5) - 1) - 1)) - 1)
        -4611686018427387904,  # ~((1 << (((1 << 6) - 1) - 1)) - 1)
        -4611686018427387904,  # ~((1 << (((1 << 7) - 1) - 1)) - 1)
        -4611686018427387904,  # ~((1 << (((1 << 8) - 1) - 1)) - 1)
    ]

    # Spacing constant used to compute offsets into ``TWO_TO_L``.
    REG_WIDTH_INDEX_MULTIPLIER = HLL.MAXIMUM_LOG2M_PARAM + 1

    @classmethod
    def register_bit_size(cls, expected_unique_elements):
        """
        Computes the bit-width of HLL registers necessary to estimate a set of
        the specified cardinality.

        :param long expected_unique_elements: an upper bound on the number of unique
               elements that are expected.  This must be greater than zero.
        :returns: a register size in bits (i.e. ``log2(log2(n))``)
        :rtype: int
        """
        return max(
            HLL.MINIMUM_REGWIDTH_PARAM,
            NumberUtil.log2(NumberUtil.log2(expected_unique_elements))
        )

    @classmethod
    def alpha_m_squared(cls, m):
        """
        Computes the 'alpha-m-squared' constant used by the HyperLogLog algorithm.

        :param int m: this must be a power of two, cannot be less than
               16 (2:sup:`4`), and cannot be greater than 65536 (2:sup:`16`).
        :returns: gamma times ``registerCount`` squared where gamma is
                  based on the value of ``registerCount``.
        :rtype: float
        """

        if m < 16:
            raise Exception("'m' cannot be less than 16 ({m} < 16).".format(m=m))

        elif m == 16:
            return 0.673 * m * m

        elif m == 32:
            return 0.673 * m * m

        elif m == 64:
            return 0.709 * m * m

        else:
            return (0.7213 / (1.0 + 1.079 / m)) * m * m

    @classmethod
    def pw_max_mask(cls, register_size_in_bits):
        """
        Computes a mask that prevents overflow of HyperLogLog registers.

        :param int register_size_in_bits: the size of the HLL registers, in bits.
        :returns: mask a ``long`` mask to prevent overflow of the registers
        :rtype: long
        """
        return cls.PW_MASK[register_size_in_bits]

    @classmethod
    def small_estimator_cutoff(cls, m):
        """
        The cutoff for using the "small range correction" formula, in the
        HyperLogLog algorithm.

        :param int m: the number of registers in the HLL. <em>m<em> in the paper.
        :returns: the cutoff for the small range correction.
        :rtype: float
        """
        return (float(m) * 5) / 2

    @classmethod
    def small_estimator(cls, m, number_of_zeroes):
        """
        The "small range correction" formula from the HyperLogLog algorithm. Only
        appropriate if both the estimator is smaller than ``(5/2) * m`` and
        there are still registers that have the zero value.

        :param int m: the number of registers in the HLL. <em>m<em> in the paper.
        :param int number_of_zeroes: the number of registers with value zero. ``V``
               in the paper.
        :returns: a corrected cardinality estimate.
        :rtype: float
        """
        return m * log(float(m) / number_of_zeroes)

    @classmethod
    def large_estimator_cutoff(cls, log2m, register_size_in_bits):
        """
        The cutoff for using the "large range correction" formula, from the
        HyperLogLog algorithm, adapted for 64 bit hashes.

        See `Blog post with section on 64 bit hashes and "large range correction" cutoff<http://research.neustar.biz/2013/01/24/hyperloglog-googles-take-on-engineering-hll/>`_.

        :param int log2m: log-base-2 of the number of registers in the HLL. <em>b<em> in the paper.
        :param int register_size_in_bits: the size of the HLL registers, in bits.
        :returns: the cutoff for the large range correction.
        :rtype: float
        """
        return TWO_TO_L[
            (cls.REG_WIDTH_INDEX_MULTIPLIER * register_size_in_bits) + log2m
        ] / 30.0

    @classmethod
    def large_estimator(cls, log2m, register_size_in_bits, estimator):
        """
        The "large range correction" formula from the HyperLogLog algorithm, adapted
        for 64 bit hashes. Only appropriate for estimators whose value exceeds
        the return of ``largeEstimatorCutoff()``.

        See `Blog post with section on 64 bit hashes and "large range correction" cutoff<http://research.neustar.biz/2013/01/24/hyperloglog-googles-take-on-engineering-hll/>`_.

        :param int log2m: log-base-2 of the number of registers in the HLL. <em>b<em> in the paper.
        :param int register_size_in_bits: the size of the HLL registers, in bits.
        :param float estimator: the original estimator ("E" in the paper).
        :returns: a corrected cardinality estimate.
        :rtype: float
        """
        two_to_l = TWO_TO_L[(cls.REG_WIDTH_INDEX_MULTIPLIER * register_size_in_bits) + log2m]
        try:
            return -1 * two_to_l * log(1.0 - (estimator/two_to_l))
        except ValueError:
            return 0


# Precomputed ``twoToL`` values indexed by a linear combination of
# ``regwidth`` and ``log2m``.
#
# The array is one-dimensional and can be accessed by using index
# ``(REG_WIDTH_INDEX_MULTIPLIER * regwidth) + log2m``
# for ``regwidth`` and ``log2m`` between the specified
# ``HLL.{MINIMUM,MAXIMUM}_{REGWIDTH,LOG2M}_PARAM`` constants.
#
# See ``large_estimator()``.
# See ``large_estimator_cutoff()``.
# See `Blog post with section on 2^L
# <http://research.neustar.biz/2013/01/24/hyperloglog-googles-take-on-engineering-hll/>`_
TWO_TO_L = [0.0] * (HLL.MAXIMUM_REGWIDTH_PARAM + 1) * (HLL.MAXIMUM_LOG2M_PARAM + 1)
for reg_width in range(HLL.MINIMUM_REGWIDTH_PARAM, HLL.MAXIMUM_REGWIDTH_PARAM+1):
    for log2m in range(HLL.MINIMUM_LOG2M_PARAM, HLL.MAXIMUM_LOG2M_PARAM+1):
        max_register_value = BitUtil.left_shift_int(1, reg_width) - 1

        # Since 1 is added to p(w) in the insertion algorithm, only
        # (maxRegisterValue - 1) bits are inspected hence the hash
        # space is one power of two smaller.
        pw_bits = max_register_value - 1
        total_bits = pw_bits + log2m
        two_to_l = 2**total_bits
        TWO_TO_L[(HLLUtil.REG_WIDTH_INDEX_MULTIPLIER * reg_width) + log2m] = two_to_l
