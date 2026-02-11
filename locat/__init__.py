"""
    The ``locat`` package provides the implementation of Locat.

    The main class is found under locat_condensed.LOCAT
"""

# Get the version from _version.py (added when building using scm)
try:
    from .version import __version__ # noqa
except ModuleNotFoundError as e:
    __version__ = '0.0.0-dev'