# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Measure the blast radius of a fault from per-service request success.

A service is in the blast radius of a fault when its rate of successful
server spans during the fault falls by at least a threshold fraction of its
rate before the fault. Callers of a failed service lose successful requests
because their calls fail; callees lose them because no traffic reaches them.

The counts come from the span metrics that the OpenTelemetry Collector's
spanmetrics connector exports to Prometheus. They count every span, so trace
sampling and trace-query limits do not affect the measurement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import requests

SPAN_METRIC = "traces_span_metrics_calls_total"
SERVER_SPAN_KIND = "SPAN_KIND_SERVER"
CONSUMER_SPAN_KIND = "SPAN_KIND_CONSUMER"
ERROR_STATUS = "STATUS_CODE_ERROR"


def successful_rates(samples: Iterable[tuple[str, str, float]], seconds: float) -> dict[str, float]:
    """Per-service rate of successful requests handled (server and consumer spans).

    Args:
        samples: (service, status_code, span count) tuples for one window.
        seconds: length of the window in seconds.

    Returns:
        dict mapping each service to successful spans per second. A service
        that only produced error spans maps to 0.0.
    """
    if seconds <= 0:
        raise ValueError("window length must be positive")
    rates: dict[str, float] = {}
    for service, status, count in samples:
        ok = 0.0 if status == ERROR_STATUS else float(count)
        rates[service] = rates.get(service, 0.0) + ok / seconds
    return rates


def rate_drops(
    before: Mapping[str, float], during: Mapping[str, float], min_rate: float = 0.05
) -> dict[str, float]:
    """Fractional drop in successful request rate for each service.

    Services whose baseline rate is below `min_rate` requests per second are
    left out, since a drop in a near-silent service is mostly noise. Callers
    usually set `min_rate` from the fault window, for example 20 expected
    requests divided by its length in seconds. A service that disappears
    during the fault has a drop of 1.0.
    """
    drops = {}
    for service, base in before.items():
        if base < min_rate:
            continue
        now = during.get(service, 0.0)
        drops[service] = min(1.0, max(0.0, 1.0 - now / base))
    return drops


def blast_radius(drops: Mapping[str, float], failed: str, threshold: float = 0.2) -> set[str]:
    """Services other than `failed` whose successful rate fell by `threshold` or more."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    return {s for s, d in drops.items() if s != failed and d >= threshold}


def _normalise(names: Iterable[str]) -> set[str]:
    return {str(n).strip().lower() for n in names if str(n).strip()}


def score_submission(submitted: Iterable[str], truth: Iterable[str], failed: str) -> dict:
    """Compare an agent's list of affected services with the measured set.

    The failed service itself is scored separately (`names_failed_service`)
    and removed from the submission before the set comparison, because the
    measured set leaves it out.
    """
    failed_name = failed.strip().lower()
    sub = _normalise(submitted)
    names_failed = failed_name in sub
    sub.discard(failed_name)
    true = _normalise(truth)
    true.discard(failed_name)

    hit = len(sub & true)
    if not sub and not true:
        precision = recall = f1 = jaccard = 1.0
    else:
        precision = hit / len(sub) if sub else 0.0
        recall = hit / len(true) if true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        jaccard = hit / len(sub | true)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "names_failed_service": names_failed,
        "exact_match": sub == true,
    }


class SpanMetricsQuery:
    """Read per-service span counts for a time window from Prometheus."""

    def __init__(self, base_url: str, metric: str = SPAN_METRIC, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.metric = metric
        self.timeout = timeout

    def window_counts(self, start: float, end: float) -> list[tuple[str, str, float]]:
        """(service, status_code, count) of server and consumer spans between `start` and `end` (Unix seconds).

        Consumer spans count as handled requests, so services that only read
        from a queue are measured too.
        """
        seconds = int(round(end - start))
        if seconds <= 0:
            raise ValueError("end must be after start")
        query = (
            f'sum by (service_name, status_code) '
            f'(increase({self.metric}{{span_kind=~"{SERVER_SPAN_KIND}|{CONSUMER_SPAN_KIND}"}}[{seconds}s]))'
        )
        resp = requests.get(
            f"{self.base_url}/api/v1/query",
            params={"query": query, "time": end},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body}")
        out = []
        for row in body["data"]["result"]:
            labels = row["metric"]
            out.append(
                (labels.get("service_name", ""), labels.get("status_code", ""), float(row["value"][1]))
            )
        return out

    def success_rates(self, start: float, end: float) -> dict[str, float]:
        """Successful requests handled per second for each service in the window."""
        return successful_rates(self.window_counts(start, end), end - start)
