# This file is here to add the project root to the sys.path to prevent
# import errors when running a single test. See https://stackoverflow.com/a/50610630/378457

# It also defines the --fast-only command-line option below.

import pytest


def pytest_addoption(parser):
    parser.addoption("--fast-only", action="store_true", help="Run fast tests only")


@pytest.fixture
def fastonly(request):
    return request.config.getoption("--fast-only")
