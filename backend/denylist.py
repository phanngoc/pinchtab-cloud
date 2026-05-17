"""Domain denylist with eTLD+1 normalization (CEO review D4 + outside voice fix).

The naive approach — regex against the URL string — fails on:
  - subdomains: m.shopee.vn, shopee.com.vn
  - redirect hops: l.facebook.com → facebook.com
  - IDN/punycode: xn--shp-2na.vn rendering as shopée.vn
  - IP literals: 31.13.66.35 (Facebook)
  - iframe contexts: parent OK, iframe loads denied origin

This module normalizes a URL to its registrable domain (eTLD+1) using the
Public Suffix List (via tldextract), and supports allow/deny rules at that
level. IP-literal hosts are blocked unless explicitly allowlisted.

The enforcement is layered: this library is the policy oracle. The worker's
CDP request interceptor (worker/cdp_interceptor.py) calls is_blocked() on
EVERY network request — main frame, sub-frame, fetch, XHR, image. That is
how click-redirect bypass is prevented.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import tldextract

# Default denylist — ToS-violation hot spots flagged in CEO review Gray-Zone Gate.
# Stored as registrable domains (eTLD+1). Subdomains are caught by normalization.
DEFAULT_DENY: frozenset[str] = frozenset(
    {
        "shopee.vn",
        "shopee.com.vn",
        "shopee.com",
        "facebook.com",
        "fb.com",
        "instagram.com",
        "tiktok.com",
        "lazada.vn",
        "lazada.com",
    }
)


_IDN_PATTERN = re.compile(r"^xn--", re.IGNORECASE)


@dataclass(frozen=True)
class DenylistPolicy:
    """Per-tenant policy. Allow set takes precedence over deny set."""

    deny: frozenset[str] = field(default_factory=lambda: DEFAULT_DENY)
    allow: frozenset[str] = field(default_factory=frozenset)
    block_ip_literals: bool = True
    block_idn_unicode: bool = False  # block idn unicode hostnames that decoded to denied

    def with_user_overrides(
        self, extra_deny: set[str] | None = None, extra_allow: set[str] | None = None
    ) -> "DenylistPolicy":
        return DenylistPolicy(
            deny=self.deny | frozenset(extra_deny or set()),
            allow=self.allow | frozenset(extra_allow or set()),
            block_ip_literals=self.block_ip_literals,
            block_idn_unicode=self.block_idn_unicode,
        )


# tldextract caches the public suffix list to ~/.cache/python-tldextract by default.
_extractor = tldextract.TLDExtract(suffix_list_urls=())  # offline mode; bundled list


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def registrable_domain(url_or_host: str) -> str | None:
    """Return the eTLD+1 of a URL or host string, normalized lowercase.

    Returns None for invalid input or hosts without a registrable suffix.
    Punycode IDN labels are decoded so denylist entries can be expressed in unicode.
    """
    if not url_or_host:
        return None

    parsed = urlparse(url_or_host if "://" in url_or_host else f"https://{url_or_host}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None

    # IP literals don't have eTLD+1; surface them as-is so callers can decide.
    if _is_ip_literal(host):
        return host

    try:
        host_decoded = host.encode("ascii").decode("idna") if "xn--" in host else host
    except UnicodeError:
        host_decoded = host

    ext = _extractor(host_decoded)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}".lower()


@dataclass(frozen=True)
class DenyDecision:
    blocked: bool
    reason: str = ""
    matched_rule: str = ""


def evaluate(url: str, policy: DenylistPolicy = DenylistPolicy()) -> DenyDecision:
    """Decide whether a URL should be blocked under the policy.

    Order of checks:
      1. Allow set wins outright.
      2. IP-literal block.
      3. eTLD+1 against deny set.
      4. Otherwise allow.
    """
    rd = registrable_domain(url)
    if rd is None:
        return DenyDecision(blocked=True, reason="unparseable_host", matched_rule=url)

    if rd in policy.allow:
        return DenyDecision(blocked=False, reason="allowlisted", matched_rule=rd)

    if _is_ip_literal(rd):
        if policy.block_ip_literals and rd not in policy.allow:
            return DenyDecision(blocked=True, reason="ip_literal", matched_rule=rd)
        return DenyDecision(blocked=False)

    if rd in policy.deny:
        return DenyDecision(blocked=True, reason="domain_denied", matched_rule=rd)

    return DenyDecision(blocked=False)


def is_blocked(url: str, policy: DenylistPolicy = DenylistPolicy()) -> bool:
    return evaluate(url, policy).blocked
