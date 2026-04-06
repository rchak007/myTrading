#!/usr/bin/env python3
"""
test_aftermath_api.py — standalone probe of the Aftermath REST API

Run:
  python3 test_aftermath_api.py

This will:
  1. GET /router/trade-route  (find DEEP→USDC route)
  2. Print the FULL route response (so we can see exact shape)
  3. POST /router/transactions/trade (try to build tx)
  4. Print the FULL response/error

No wallet needed — just tests the API.
"""

import json
import re
import requests

API_BASE = "https://aftermath.finance/api"

DEEP_TYPE = "0xdeeb7a4662eec9f2f3def03fb937a663dddaa2e215b8078a284d026b7946c270::deep::DEEP"
USDC_TYPE = "0xdba34672e30cb065b1f93e3ab55318768fd6fef66c15942c9f7cb846e2f900e7::usdc::USDC"
WALLET    = "0x246c7037d5fd8c424e45631b930c2f3acacbee27a07a6863e797b8700a6f331d"

AMOUNT_RAW = 180_000_000_000   # ~180 DEEP (9 decimals)


def strip_bigint_n(obj):
    """Recursively strip trailing 'n' from BigInt strings."""
    if isinstance(obj, dict):
        return {k: strip_bigint_n(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_bigint_n(v) for v in obj]
    elif isinstance(obj, str) and re.fullmatch(r'-?\d+n', obj):
        return obj[:-1]
    return obj


def parse_response(resp):
    """Parse JSON, handling bare BigInt notation."""
    raw = resp.text
    cleaned = re.sub(r'(?<=[\s:,\[])(\d+)n(?=[\s,}\]])', r'\1', raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return resp.json()


print("=" * 60)
print("STEP 1: Get trade route")
print("=" * 60)

route_url = f"{API_BASE}/router/trade-route"
route_payload = {
    "coinInType":   DEEP_TYPE,
    "coinOutType":  USDC_TYPE,
    "coinInAmount": str(AMOUNT_RAW),
}

print(f"POST {route_url}")
print(f"Payload: {json.dumps(route_payload, indent=2)}")
print()

r1 = requests.post(route_url, json=route_payload, timeout=30)
print(f"Status: {r1.status_code}")

if r1.status_code != 200:
    print(f"ERROR body: {r1.text[:1000]}")
    exit(1)

route = parse_response(r1)
route = strip_bigint_n(route)

print(f"Route response (pretty):")
print(json.dumps(route, indent=2, default=str)[:3000])
print("...")
print()


print("=" * 60)
print("STEP 2: Build transaction — attempt A (completeRoute + walletAddress)")
print("=" * 60)

tx_url = f"{API_BASE}/router/transactions/trade"

# Attempt A: what the TS SDK docs suggest
payload_a = {
    "completeRoute":  route,
    "walletAddress":  WALLET,
    "slippage":       0.03,
}

print(f"POST {tx_url}")
print(f"Payload keys: {list(payload_a.keys())}")
print()

r2a = requests.post(tx_url, json=payload_a, timeout=30)
print(f"Status: {r2a.status_code}")
print(f"Response: {r2a.text[:1000]}")
print()

if r2a.status_code == 200:
    print("SUCCESS with attempt A!")
    tx_data = r2a.json()
    print(f"Response keys: {list(tx_data.keys())}")
    exit(0)


print("=" * 60)
print("STEP 3: Build transaction — attempt B (route + walletAddress)")
print("=" * 60)

payload_b = {
    "route":          route,
    "walletAddress":  WALLET,
    "slippage":       0.03,
}

r2b = requests.post(tx_url, json=payload_b, timeout=30)
print(f"Status: {r2b.status_code}")
print(f"Response: {r2b.text[:1000]}")
print()


print("=" * 60)
print("STEP 4: Build transaction — attempt C (just the route fields directly)")
print("=" * 60)

# Maybe it expects the route fields at the top level + wallet + slippage
payload_c = {**route, "walletAddress": WALLET, "slippage": 0.03}

r2c = requests.post(tx_url, json=payload_c, timeout=30)
print(f"Status: {r2c.status_code}")
print(f"Response: {r2c.text[:1000]}")
print()


print("=" * 60)
print("STEP 5: Build transaction — attempt D (sender + slippage)")
print("=" * 60)

payload_d = {
    "completeRoute":  route,
    "sender":         WALLET,
    "slippage":       0.03,
}

r2d = requests.post(tx_url, json=payload_d, timeout=30)
print(f"Status: {r2d.status_code}")
print(f"Response: {r2d.text[:1000]}")
print()

# Also try just GET with query params
print("=" * 60)
print("STEP 6: Try GET instead of POST")
print("=" * 60)

r2e = requests.get(tx_url, params={
    "walletAddress": WALLET,
    "slippage": 0.03,
}, timeout=30)
print(f"GET Status: {r2e.status_code}")
print(f"Response: {r2e.text[:500]}")
print()


# Try the OpenAPI spec
print("=" * 60)
print("STEP 7: Fetch OpenAPI spec for router endpoints")
print("=" * 60)

for spec_url in [
    "https://aftermath.finance/api/openapi.json",
    "https://aftermath.finance/api/docs",
    "https://aftermath.finance/docs/openapi.json",
    "https://openapi.aftermath.finance/spec.json",
]:
    try:
        r = requests.get(spec_url, timeout=10)
        print(f"{spec_url} → {r.status_code} (len={len(r.text)})")
        if r.status_code == 200 and "router" in r.text.lower():
            # Find router-related endpoints
            for line in r.text.split("\n"):
                if "router" in line.lower() and ("trade" in line.lower() or "transaction" in line.lower()):
                    print(f"  → {line.strip()[:200]}")
    except Exception as e:
        print(f"{spec_url} → FAILED: {e}")

print()
print("Done! Share the output above and I'll fix the payload format.")