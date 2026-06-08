"""Capture cluster state around a benchmark run.

Writes ``results/<label>-<timestamp>/cluster-state.yaml`` so post-mortem
reviewers can answer "did a pod restart mid-run?" without live cluster
access. See CAC-161 for the motivating incident.

Two subcommands:

  snapshot   one-shot dump of pod + node state for the listed targets.
             Called twice from run_tuning_bench.sh — once before genai-bench
             starts, once after it finishes.

  finalize   reads the two snapshots, pulls events that fell inside the run
             window from the listed namespaces, and writes the merged YAML.

Targets are passed as ``<namespace>:<name-prefix>``; every pod in the
namespace whose name starts with the prefix is captured. We use prefix
match (not label selectors) because the deployments we care about — the
vllm-engine StatefulSet, the lm-smoke lm-cache pods, the gpu-baseline
vLLM pod — don't share a single label convention across the repo, but
their names are stable.

Failure modes are deliberately soft. kubectl errors degrade to a stub
``cluster-state.yaml`` with an ``error:`` field so the rest of the run
artifacts still ship.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


def _kubectl_json(args: list[str]) -> dict[str, Any] | None:
    """Run kubectl, return parsed JSON, or None on any failure."""
    try:
        proc = subprocess.run(
            ["kubectl", *args, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        sys.stderr.write(f"[cluster-state] kubectl {' '.join(args)} failed: {e}\n")
        return None
    if proc.returncode != 0:
        sys.stderr.write(
            f"[cluster-state] kubectl {' '.join(args)} rc={proc.returncode}: "
            f"{proc.stderr.strip()[:200]}\n"
        )
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[cluster-state] bad json from kubectl: {e}\n")
        return None


def _age(start_iso: str | None) -> str:
    """Pod age as a short string ('10h12m' / '37s'), or '' if start is missing."""
    if not start_iso:
        return ""
    try:
        started = datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return ""
    delta = datetime.now(timezone.utc) - started
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h{mins:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _summarize_pod(pod: dict[str, Any]) -> dict[str, Any]:
    meta = pod.get("metadata", {})
    spec = pod.get("spec", {})
    status = pod.get("status", {})
    restart = 0
    for cs in status.get("containerStatuses") or []:
        restart += int(cs.get("restartCount") or 0)
    return {
        "name": meta.get("name"),
        "namespace": meta.get("namespace"),
        "pod_uid": meta.get("uid"),
        "node": spec.get("nodeName"),
        "phase": status.get("phase"),
        "restart_count": restart,
        "started_at": status.get("startTime"),
        "age": _age(status.get("startTime")),
    }


def _collect_pods(targets: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """For each (namespace, prefix) target, gather matching pod summaries."""
    out: list[dict[str, Any]] = []
    pods_by_ns: dict[str, list[dict[str, Any]]] = {}
    for ns, _prefix in targets:
        if ns in pods_by_ns:
            continue
        listing = _kubectl_json(["-n", ns, "get", "pods"])
        pods_by_ns[ns] = (listing or {}).get("items", []) or []
    for ns, prefix in targets:
        for pod in pods_by_ns[ns]:
            name = (pod.get("metadata") or {}).get("name") or ""
            if name.startswith(prefix):
                out.append(_summarize_pod(pod))
    # Stable order: namespace, then name.
    out.sort(key=lambda p: (p.get("namespace") or "", p.get("name") or ""))
    return out


def _collect_nodes(node_names: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(n for n in node_names if n):
        node = _kubectl_json(["get", "node", name])
        if not node:
            out.append({"name": name, "error": "kubectl get node failed"})
            continue
        status = node.get("status") or {}
        cap = status.get("capacity") or {}
        alloc = status.get("allocatable") or {}
        conditions = {c.get("type"): c.get("status") for c in status.get("conditions") or []}
        out.append(
            {
                "name": name,
                "memory_capacity": cap.get("memory"),
                "memory_allocatable": alloc.get("memory"),
                "cpu_capacity": cap.get("cpu"),
                "cpu_allocatable": alloc.get("cpu"),
                "ready": conditions.get("Ready") == "True",
                "memory_pressure": conditions.get("MemoryPressure") == "True",
                "disk_pressure": conditions.get("DiskPressure") == "True",
                "pid_pressure": conditions.get("PIDPressure") == "True",
                "kernel_version": (status.get("nodeInfo") or {}).get("kernelVersion"),
                "kubelet_version": (status.get("nodeInfo") or {}).get("kubeletVersion"),
            }
        )
    return out


def _collect_events(namespaces: list[str], since_epoch: float) -> list[dict[str, Any]]:
    since_iso_min = (
        datetime.fromtimestamp(since_epoch, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    events: list[dict[str, Any]] = []
    for ns in namespaces:
        listing = _kubectl_json(
            [
                "get",
                "events",
                "-A",
                f"--field-selector=involvedObject.namespace={ns}",
            ]
        )
        for ev in (listing or {}).get("items", []) or []:
            ts = (
                ev.get("eventTime")
                or ev.get("lastTimestamp")
                or ev.get("firstTimestamp")
                or (ev.get("metadata") or {}).get("creationTimestamp")
            )
            if not ts or ts < since_iso_min:
                continue
            obj = ev.get("involvedObject") or {}
            events.append(
                {
                    "timestamp": ts,
                    "type": ev.get("type"),
                    "reason": ev.get("reason"),
                    "object": f"{obj.get('kind','')}/{obj.get('name','')}".strip("/"),
                    "namespace": obj.get("namespace"),
                    "count": ev.get("count"),
                    "message": (ev.get("message") or "").strip(),
                }
            )
    events.sort(key=lambda e: e.get("timestamp") or "")
    return events


def _parse_targets(raw: list[str]) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for entry in raw:
        if ":" not in entry:
            raise SystemExit(f"target must be NAMESPACE:PREFIX, got {entry!r}")
        ns, prefix = entry.split(":", 1)
        if not ns or not prefix:
            raise SystemExit(f"target must be NAMESPACE:PREFIX, got {entry!r}")
        targets.append((ns, prefix))
    return targets


def cmd_snapshot(args: argparse.Namespace) -> int:
    targets = _parse_targets(args.target)
    pods = _collect_pods(targets)
    snapshot = {
        "captured_at_epoch": time.time(),
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": [f"{ns}:{p}" for ns, p in targets],
        "pods": pods,
    }
    with open(args.output, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    sys.stderr.write(
        f"[cluster-state] snapshot: {len(pods)} pods → {args.output}\n"
    )
    return 0


def _diff_pods(
    start: list[dict[str, Any]], end: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key_start = {(p.get("namespace"), p.get("name")): p for p in start}
    by_key_end = {(p.get("namespace"), p.get("name")): p for p in end}
    notes: list[dict[str, Any]] = []
    for key, e in by_key_end.items():
        s = by_key_start.get(key)
        if s is None:
            notes.append(
                {
                    "name": e.get("name"),
                    "namespace": e.get("namespace"),
                    "change": "created_during_run",
                }
            )
            continue
        if s.get("pod_uid") and e.get("pod_uid") and s["pod_uid"] != e["pod_uid"]:
            notes.append(
                {
                    "name": e.get("name"),
                    "namespace": e.get("namespace"),
                    "change": "pod_uid_changed",
                    "uid_start": s.get("pod_uid"),
                    "uid_end": e.get("pod_uid"),
                }
            )
        if (e.get("restart_count") or 0) > (s.get("restart_count") or 0):
            notes.append(
                {
                    "name": e.get("name"),
                    "namespace": e.get("namespace"),
                    "change": "container_restarted",
                    "restart_count_start": s.get("restart_count"),
                    "restart_count_end": e.get("restart_count"),
                }
            )
    for key, s in by_key_start.items():
        if key not in by_key_end:
            notes.append(
                {
                    "name": s.get("name"),
                    "namespace": s.get("namespace"),
                    "change": "deleted_during_run",
                }
            )
    return notes


def cmd_finalize(args: argparse.Namespace) -> int:
    try:
        import yaml  # PyYAML — installed via `make install`.
    except ImportError:
        sys.stderr.write("[cluster-state] PyYAML missing; install with `pip install pyyaml`\n")
        return 1

    try:
        with open(args.start) as f:
            start = json.load(f)
    except FileNotFoundError:
        start = {"pods": [], "error": f"missing start snapshot: {args.start}"}
    try:
        with open(args.end) as f:
            end = json.load(f)
    except FileNotFoundError:
        end = {"pods": [], "error": f"missing end snapshot: {args.end}"}

    nodes_of_interest = {
        p.get("node") for p in (start.get("pods") or []) + (end.get("pods") or [])
    }
    nodes = _collect_nodes(nodes_of_interest)

    namespaces = args.events_namespace or []
    events = _collect_events(namespaces, args.run_start_epoch)

    diff = _diff_pods(start.get("pods") or [], end.get("pods") or [])

    doc = {
        "schema_version": 1,
        "captured_by": "lib/cluster_state.py (CAC-161)",
        "run_start": datetime.fromtimestamp(
            args.run_start_epoch, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets_start": start.get("targets") or [],
        "targets_end": end.get("targets") or [],
        "pods_start": start.get("pods") or [],
        "pods_end": end.get("pods") or [],
        "pod_changes": diff,
        "events_namespaces": namespaces,
        "events": events,
        "nodes": nodes,
    }
    if start.get("error"):
        doc["start_error"] = start["error"]
    if end.get("error"):
        doc["end_error"] = end["error"]

    with open(args.output, "w") as f:
        f.write("# cluster-state.yaml — written by lib/cluster_state.py (CAC-161).\n")
        f.write("# Pod + node + event snapshot for the run window. See README §cluster-state.\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    summary = (
        f"[cluster-state] finalize: {len(doc['pods_start'])} pods@start, "
        f"{len(doc['pods_end'])} pods@end, {len(diff)} change(s), "
        f"{len(events)} event(s), {len(nodes)} node(s) → {args.output}\n"
    )
    sys.stderr.write(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="dump current pod/node state to JSON")
    sp.add_argument(
        "--target",
        action="append",
        default=[],
        required=True,
        metavar="NAMESPACE:PREFIX",
        help="Capture pods in NAMESPACE whose name starts with PREFIX. Repeatable.",
    )
    sp.add_argument("--output", required=True, help="Path to write the snapshot JSON.")
    sp.set_defaults(func=cmd_snapshot)

    fp = sub.add_parser("finalize", help="merge two snapshots into cluster-state.yaml")
    fp.add_argument("--start", required=True, help="Path to the start snapshot JSON.")
    fp.add_argument("--end", required=True, help="Path to the end snapshot JSON.")
    fp.add_argument(
        "--run-start-epoch",
        required=True,
        type=float,
        help="Unix epoch seconds when the run started; events older than this are discarded.",
    )
    fp.add_argument(
        "--events-namespace",
        action="append",
        default=[],
        help="Namespace to pull events from. Repeatable.",
    )
    fp.add_argument("--output", required=True, help="Path to write cluster-state.yaml.")
    fp.set_defaults(func=cmd_finalize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
