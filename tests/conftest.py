"""Pytest fixtures + import shims for the benchmark harness.

Both the dumb client (`lib/dumb_gateway_client.py`, CAC-152) and the legacy
proxy retained for one release (`lib/lookup_proxy_legacy.py`) hard-import
the inference-cache gRPC stubs at module load time and `sys.exit(2)` when
they're missing. Generating them needs the sibling main repo (`make proto`)
— too heavy a dependency for a unit test. The shim below registers
placeholder modules in `sys.modules` BEFORE pytest collects test files, so
the `from inferencecache.v1alpha1 import ...` succeeds against fakes.

Adds `lib/` to `sys.path` so tests can `from dumb_gateway_client import ...`
matching the way `python3 lib/dumb_gateway_client.py` runs it in production.
"""

from __future__ import annotations

import os
import sys
import types


def _shim_inferencecache_proto() -> None:
    if "inferencecache.v1alpha1.inferencecache_pb2" in sys.modules:
        return

    pkg = types.ModuleType("inferencecache")
    pkg.__path__ = []  # mark as a package
    sub = types.ModuleType("inferencecache.v1alpha1")
    sub.__path__ = []

    pb2 = types.ModuleType("inferencecache.v1alpha1.inferencecache_pb2")

    class _LookupRouteRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    pb2.LookupRouteRequest = _LookupRouteRequest

    pb2_grpc = types.ModuleType("inferencecache.v1alpha1.inferencecache_pb2_grpc")

    class _Stub:
        def __init__(self, *_args, **_kwargs):
            pass

    pb2_grpc.InferenceCacheStub = _Stub

    sys.modules["inferencecache"] = pkg
    sys.modules["inferencecache.v1alpha1"] = sub
    sys.modules["inferencecache.v1alpha1.inferencecache_pb2"] = pb2
    sys.modules["inferencecache.v1alpha1.inferencecache_pb2_grpc"] = pb2_grpc


_shim_inferencecache_proto()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LIB_DIR = os.path.join(_REPO_ROOT, "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)
