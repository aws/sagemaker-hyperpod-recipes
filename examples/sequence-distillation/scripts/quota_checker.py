"""SageMaker endpoint-usage service-quota validation with cached quota-code lookups.

Adapted from the training-job quota checker for real-time inference endpoints.

A real-time endpoint provisions instances over an instance preference list (ProductionVariant.instance_pools)
across a ranked list of instance types; the relevant Service Quota per type is "<type> for endpoint usage"
(e.g. "ml.g6e.2xlarge for endpoint usage"). SageMaker falls back across the list, so the account can host the
run if ANY type in the chain has enough quota (not necessarily the first).

Fetch strategy (same as the training checker): the AWS DEFAULT quota catalog (list_aws_default_service_quotas)
is static and AWS-cached, so we build the quota name->code map from it once, persist it to a local json, and
then read only the *applied* values for the requested types with targeted get_service_quota(QuotaCode) calls.
This avoids the heavy, throttle-prone enumeration of per-account list_service_quotas.

Usage:
    from quota_checker import EndpointQuotaPreflight
    EndpointQuotaPreflight(region=REGION).check(
        instance_types=INSTANCE_TYPES, n_new_endpoints=N_NEW)   # prints a table; raises on FAIL
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config

QUOTA_SUFFIX = " for endpoint usage"  # exact Service Quotas naming: "<instance_type> for endpoint usage"

# Anchor the cache one directory up from scripts/ (the project root), NOT the current working directory, so the
# whole project shares one cache regardless of which notebook/cwd invokes the check. This module lives at
# <project-root>/scripts/quota_checker.py, so its parent's parent is the project root. (gitignored.)
_CACHE_DIR = Path(__file__).resolve().parent.parent


def _get_quotas_client(region):
    """Service Quotas client with adaptive retry (handles the occasional throttle in place)."""
    return boto3.client(
        "service-quotas", region_name=region, config=Config(retries={"max_attempts": 5, "mode": "adaptive"})
    )


@dataclass
class PreflightResult:
    """Structured outcome of a quota check (logic separated from presentation)."""

    instance_types: List[str]
    n_new: int
    quotas: Dict[str, Optional[int]]  # instance_type -> applied quota value (None if not found)
    source: str  # "cache" or "api" (for the code map)
    verdict: str  # "pass" | "marginal" | "fail" | "skip"
    message: str
    viable: List[str] = field(default_factory=list)  # types whose quota >= n_new


class EndpointQuotaPreflight:
    def __init__(self, region: str, cache_ttl_s: int = 24 * 3600):
        self.region = region
        # Cache the quota NAME->CODE map (not values): codes are stable, applied values are read live per run.
        self.cache = _CACHE_DIR / f".quota_codes_endpoint_usage_{region}.json"
        self.cache_ttl_s = cache_ttl_s

    # ---------- name->code map (cached) ----------
    def load_or_build_quota_cache(self, force_refresh: bool = False) -> Dict[str, str]:
        """Load the cached instance_type->quota_code map, or build it from the AWS defaults catalog.

        Only caches '<type> for endpoint usage' quotas to keep it small. Uses list_aws_default_service_quotas
        (cached on AWS side, less throttle-prone than list_service_quotas)."""
        if not force_refresh and self.cache.exists():
            try:
                blob = json.loads(self.cache.read_text())
                if blob.get("region") == self.region and (time.time() - blob.get("fetched_at", 0)) < self.cache_ttl_s:
                    print(f"  Loaded quota cache ({len(blob['codes'])} endpoint-usage quotas)")
                    return blob["codes"]
            except Exception:
                pass  # unreadable/stale cache -> rebuild

        print("  Building endpoint-usage quota-code cache (first run only)...")
        quotas_client = _get_quotas_client(self.region)
        codes: Dict[str, str] = {}
        paginator = quotas_client.get_paginator("list_aws_default_service_quotas")
        for page in paginator.paginate(ServiceCode="sagemaker", PaginationConfig={"PageSize": 100}):
            for quota in page["Quotas"]:
                name = quota["QuotaName"]
                if name.endswith(QUOTA_SUFFIX):
                    codes[name[: -len(QUOTA_SUFFIX)]] = quota["QuotaCode"]  # strip suffix -> instance type
        try:
            self.cache.write_text(json.dumps({"region": self.region, "fetched_at": time.time(), "codes": codes}))
            print(f"  Cached {len(codes)} endpoint-usage quota codes to {self.cache}")
        except Exception as e:
            print(f"  (warning: could not write quota-code cache {self.cache}: {e})")
        return codes

    @staticmethod
    def _get_applied_quota(quotas_client, quota_code_map, instance_type) -> Optional[int]:
        """Applied quota value for '<instance_type> for endpoint usage' via a direct get_service_quota call."""
        code = quota_code_map.get(instance_type)
        if not code:
            return None
        response = quotas_client.get_service_quota(ServiceCode="sagemaker", QuotaCode=code)
        return int(response["Quota"]["Value"])

    def load_quotas(self, instance_types: List[str], force_refresh: bool = False):
        """Return (quota_map, source): {instance_type: applied_quota or None} for ONLY the requested types.
        Resolves each type's code from the cached map, then reads its applied value with a targeted call. If a
        requested type's code is missing (e.g. a newer type absent from the cached catalog), rebuild once."""
        source = "cache" if (self.cache.exists() and not force_refresh) else "api"
        codes = self.load_or_build_quota_cache(force_refresh=force_refresh)
        if any(t not in codes for t in instance_types) and source == "cache":
            codes = self.load_or_build_quota_cache(force_refresh=True)  # a new instance type -> refresh codes
            source = "api"
        quotas_client = _get_quotas_client(self.region)
        quotas = {t: self._get_applied_quota(quotas_client, codes, t) for t in instance_types}
        return quotas, source

    # ---------- logic ----------
    def evaluate(self, instance_types: List[str], n_new_endpoints: int, force_refresh: bool = False) -> PreflightResult:
        """Compute the preflight verdict without printing. n_new_endpoints = instances needed (1 per NEW endpoint;
        0 if reused). SageMaker falls back across the preference list, so the account passes if any single type
        can host all n_new, with a MARGINAL note when only spread-across-types capacity exists."""
        chain = list(instance_types)
        if n_new_endpoints == 0:
            return PreflightResult(
                chain, 0, {}, "n/a", "skip", "Nothing to deploy — endpoint reused. Skipping quota check."
            )
        quotas, source = self.load_quotas(chain, force_refresh=force_refresh)
        qmap = {t: quotas.get(t) for t in chain}
        viable = [t for t in chain if (qmap[t] is not None and qmap[t] >= n_new_endpoints)]
        preferred = chain[0]
        pq = qmap[preferred]
        if pq is not None and pq >= n_new_endpoints:
            verdict, msg = "pass", (
                f"preferred '{preferred}' quota {int(pq)} >= {n_new_endpoints} "
                f"new endpoint(s): can deploy on the first-choice type."
            )
        elif viable:
            verdict, msg = "pass", (
                f"preferred '{preferred}' can't host all {n_new_endpoints}, but fallback "
                f"pool(s) {viable} can (instance_pools will fall through to them)."
            )
        elif sum(1 for t in chain if (qmap[t] or 0) >= 1) >= n_new_endpoints:
            verdict, msg = "marginal", (
                f"no single type can host all {n_new_endpoints} endpoint(s), but there is 1 instance across "
                f"different types — only works if endpoints land on different pools. Consider an increase on "
                f"'{preferred}' (quota name: '{preferred}{QUOTA_SUFFIX}')."
            )
        else:
            total = sum(int(qmap[t] or 0) for t in chain)
            verdict, msg = "fail", (
                f"insufficient endpoint-usage quota: need {n_new_endpoints} instance(s) across "
                f"{chain}, but total available is {total}. Request an increase in the Service Quotas console "
                f"(SageMaker, '<type>{QUOTA_SUFFIX}') for one of these types, then re-run "
                f"(delete {self.cache.name} to refresh the cache)."
            )
        return PreflightResult(chain, n_new_endpoints, qmap, source, verdict, msg, viable)

    # ---------- presentation ----------
    def report(self, result: PreflightResult) -> None:
        """Print the per-type quota table and the verdict line for a PreflightResult."""
        if result.verdict == "skip":
            print(result.message)
            return
        print(
            f"endpoint-usage quotas loaded from {result.source} "
            f"(cache: {self.cache.name}; delete it or pass force_refresh=True to refresh)\n"
        )
        n = result.n_new
        print(f"{'instance type':<20} {'quota':>6}  {'can host ' + str(n) + '?':>12}")
        print("-" * 44)
        for itype in result.instance_types:
            q = result.quotas.get(itype)
            if q is None:
                mark, note = "  UNKNOWN", "(no 'for endpoint usage' quota found; may be a newer type)"
            elif q >= n:
                mark, note = "       yes", ""
            elif q >= 1:
                mark, note = " partial", f"(only {int(q)} instance(s); fine if endpoints land on different types)"
            else:
                mark, note = "       NO", "(0 instances allowed)"
            qstr = "   n/a" if q is None else f"{q:6.0f}"
            print(f"{itype:<20} {qstr}  {mark:>12}  {note}")
        icon = {"pass": "✅ PASS", "marginal": "⚠️  MARGINAL", "fail": "❌ FAIL"}[result.verdict]
        print(f"\n{icon}  {result.message}")

    def check(self, instance_types: List[str], n_new_endpoints: int, force_refresh: bool = False) -> PreflightResult:
        """Evaluate, print the table + verdict, and raise ValueError if the verdict is FAIL. Returns the
        PreflightResult (verdict 'pass' / 'marginal' / 'skip') otherwise."""
        print(
            f"Checking SageMaker endpoint-usage quotas: "
            f"{n_new_endpoints} new endpoint(s) over {list(instance_types)}\n"
        )
        result = self.evaluate(instance_types, n_new_endpoints, force_refresh=force_refresh)
        self.report(result)
        if result.verdict == "fail":
            raise ValueError(result.message)
        return result
