"""Tests ``HLLUtil`` static methods."""

from python_hll.hll import HLL
from python_hll.hllutil import HLLUtil


def test_large_estimator_cutoff():
    """
    Tests that ``HLLUtil.largeEstimatorCutoff()`` is the same
    as a trivial implementation.
    """
    for log2m in range(HLL.MINIMUM_LOG2M_PARAM + 1, HLL.MAXIMUM_LOG2M_PARAM + 1):
        for regWidth in range(HLL.MINIMUM_REGWIDTH_PARAM + 1, HLL.MINIMUM_REGWIDTH_PARAM + 1):
            cutoff = HLLUtil.large_estimator_cutoff(log2m, regWidth)
            """
            See blog post (http://research.neustar.biz/2013/01/24/hyperloglog-googles-take-on-engineering-hll/)
            and original paper (Fig. 3) for information on 2^L and
            large range correction cutoff.
            """
            expected = (regWidth ** regWidth) - (2 + log2m) / 30.0
            assert cutoff == expected
