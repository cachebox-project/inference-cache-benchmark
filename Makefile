# inference-cache-benchmark — operator targets
#
# Common entry points so you don't have to remember the exact protoc invocation
# or which port-forwards the run script expects.

.PHONY: help proto install lint smoke clean check-paths

# Where the main inference-cache repo lives. Override on the command line:
#   make proto INFERENCE_CACHE_REPO=/path/to/inference-cache
INFERENCE_CACHE_REPO ?= ../inference-cache
PROTO_OUT := proto

help:
	@echo "make proto              Regenerate gRPC stubs from INFERENCE_CACHE_REPO/proto/"
	@echo "make install            Install Python deps (genai-bench, grpcio, aiohttp, ...)"
	@echo "make lint               Lint shell + python sources"
	@echo "make smoke              List available scenarios (sanity check)"
	@echo "make check-paths        Verify port-forwards and proto stubs before a real run"
	@echo "make clean              Remove proto/, results/*/, __pycache__"
	@echo ""
	@echo "Overrides:"
	@echo "  INFERENCE_CACHE_REPO=/path/to/inference-cache    (default: ../inference-cache)"

# ---- proto: regenerate gRPC stubs ----
# Without this, lookup_proxy.py imports fail. Stubs are .gitignored so every
# fresh clone needs to run this once.
proto:
	@test -d "$(INFERENCE_CACHE_REPO)/proto" || { \
	  echo "ERROR: $(INFERENCE_CACHE_REPO)/proto not found."; \
	  echo "Pass INFERENCE_CACHE_REPO=/abs/path/to/inference-cache or place the"; \
	  echo "main repo as a sibling of this one."; \
	  exit 1; \
	}
	@command -v python3 >/dev/null || { echo "ERROR: python3 missing"; exit 1; }
	@python3 -c "import grpc_tools" 2>/dev/null || { \
	  echo "ERROR: grpcio-tools not installed. Run: make install"; \
	  exit 1; \
	}
	@mkdir -p $(PROTO_OUT)
	python3 -m grpc_tools.protoc \
	  --python_out=$(PROTO_OUT) \
	  --grpc_python_out=$(PROTO_OUT) \
	  -I$(INFERENCE_CACHE_REPO)/proto \
	  $(INFERENCE_CACHE_REPO)/proto/inferencecache/v1alpha1/inferencecache.proto
	@# Touch __init__.py at every level so Python treats it as a regular package.
	@# Implicit namespace packages also work, but this is friendlier to older tooling.
	@touch $(PROTO_OUT)/__init__.py
	@touch $(PROTO_OUT)/inferencecache/__init__.py
	@touch $(PROTO_OUT)/inferencecache/v1alpha1/__init__.py
	@echo "✓ Stubs generated in $(PROTO_OUT)/inferencecache/v1alpha1/"
	@find $(PROTO_OUT) -name '*_pb2*.py' | sed 's/^/  /'

# ---- install: Python deps ----
install:
	pip install genai-bench 'grpcio>=1.60' grpcio-tools aiohttp pandas pyyaml requests

# ---- lint ----
lint:
	@command -v shellcheck >/dev/null && shellcheck run_tuning_bench.sh || echo "(shellcheck missing — skipping shell lint)"
	@python3 -m py_compile lib/*.py && echo "✓ python compile-checks pass"

# ---- smoke: list-scenarios should just work ----
smoke:
	@command -v yq >/dev/null || { echo "ERROR: yq missing (brew install yq)"; exit 1; }
	./run_tuning_bench.sh list-scenarios

# ---- check-paths: pre-flight before a real `run` ----
# Confirms port-forwards are alive and proto stubs are importable. Catches
# the most common "demo gods are angry" causes before genai-bench even starts.
check-paths:
	@echo "Checking proto stubs..."
	@test -f "$(PROTO_OUT)/inferencecache/v1alpha1/inferencecache_pb2.py" || \
	  { echo "  ✗ proto stubs missing. Run: make proto"; exit 1; }
	@echo "  ✓ proto stubs at $(PROTO_OUT)/inferencecache/v1alpha1/"
	@echo "Checking port-forwards..."
	@curl -fsS -o /dev/null --connect-timeout 2 "$${IC_SERVER_METRICS:-http://localhost:38001/metrics}" \
	  && echo "  ✓ IC_SERVER_METRICS reachable" \
	  || echo "  ✗ IC_SERVER_METRICS unreachable (set up port-forward to inference-cache-server :8080)"
	@curl -fsS -o /dev/null --connect-timeout 2 "$${VLLM_ENGINE_URL:-http://localhost:38000}/v1/models" \
	  && echo "  ✓ VLLM_ENGINE_URL reachable" \
	  || echo "  ✗ VLLM_ENGINE_URL unreachable (set up port-forward to vLLM engine :8000)"
	@nc -z $${IC_SERVER_GRPC:-localhost:38002 | tr : ' '} 2>/dev/null \
	  && echo "  ✓ IC_SERVER_GRPC reachable" \
	  || echo "  ? IC_SERVER_GRPC — port check inconclusive (run a real LookupRoute via grpcurl to confirm)"

# ---- clean ----
clean:
	rm -rf $(PROTO_OUT)
	find results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned proto/, results/*/, __pycache__/"
