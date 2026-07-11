from __future__ import annotations

import collect_vllm_metrics


class _Response:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_scrape_once_captures_t1_t2_counters(monkeypatch):
    body = """
# HELP vllm:prefix_cache_queries_total T1 queries
vllm:prefix_cache_queries_total{model_name="m"} 7
vllm:prefix_cache_hits_total{model_name="m"} 5
vllm:external_prefix_cache_queries_total{model_name="m"} 3
vllm:external_prefix_cache_hits_total{model_name="m"} 2
vllm:unrelated_total 999
"""

    def fake_get(url: str, timeout: int):
        assert url == "http://r0/metrics"
        assert timeout == 5
        return _Response(body)

    monkeypatch.setattr(collect_vllm_metrics.requests, "get", fake_get)

    got = collect_vllm_metrics.scrape_once("r0", "http://r0/metrics")

    def value_for(metric: str) -> float:
        for (name, labels), value in got.items():
            if name == metric:
                assert ("pod", "r0") in labels
                assert ("model_name", "m") in labels
                return value
        raise AssertionError(f"missing metric {metric}")

    assert value_for("vllm:prefix_cache_queries_total") == 7
    assert value_for("vllm:prefix_cache_hits_total") == 5
    assert value_for("vllm:external_prefix_cache_queries_total") == 3
    assert value_for("vllm:external_prefix_cache_hits_total") == 2
    assert all(name != "vllm:unrelated_total" for name, _labels in got)
