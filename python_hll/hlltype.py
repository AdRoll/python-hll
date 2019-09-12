# -*- coding: utf-8 -*-


class HLLType:
    """
    The types of algorithm/data structure that HLL can utilize. For more
    information, see the Javadoc for HLL.
    """
    EMPTY = 1
    EXPLICIT = 2
    SPARSE = 3
    FULL = 4
    UNDEFINED = 5  # used by the PostgreSQL implementation to indicate legacy/corrupt/incompatible/unknown formats
