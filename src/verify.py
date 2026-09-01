"""Task 1: verify the Nextstrain forecast archive before building anything on it.

The brief claims `nextstrain/forecasts-ncov` publishes a dated probabilistic forecast every day
since 2022-12-23 (732 open-clade / 607 open-Pango / 680 GISAID-Pango snapshots, ~3.6 GB), that
files are gzipped despite a `.json` extension, and that the GISAID-branch *outputs* are readable
with no credential.

Every one of those is an assumption until checked. The pattern from projects 01 and 04 is that
roughly half the recorded data facts turn out wrong, and finding out in week 3 is expensive.

Checks:
  1. Does the S3 bucket list, and under what prefixes?
  2. How many dated snapshots actually exist, and what is the real date range?
  3. Are the files really gzipped-despite-.json?
  4. Does a snapshot parse, and does it contain the fields the audit needs
     (freq_forecast, HDIs, growth advantages)?
  5. Is the GISAID-derived branch really downloadable without credentials?
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import gzip  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import requests  # noqa: E402

BUCKET = "https://nextstrain-data.s3.amazonaws.com/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
# VERIFIED 2026-08-29: the brief said "forecasts-ncov/", which returns ZERO objects.
# The real prefix is nested three levels deeper.
PREFIX = "files/workflows/forecasts-ncov/"


def s3_list(prefix, delimiter=None, max_keys=1000, token=None):
    url = f"{BUCKET}?list-type=2&prefix={prefix}&max-keys={max_keys}"
    if delimiter:
        url += f"&delimiter={delimiter}"
    if token:
        url += f"&continuation-token={requests.utils.quote(token, safe='')}"
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    keys = [(c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text))
            for c in root.findall("s3:Contents", NS)]
    prefixes = [p.find("s3:Prefix", NS).text for p in root.findall("s3:CommonPrefixes", NS)]
    nxt = root.find("s3:NextContinuationToken", NS)
    return keys, prefixes, (nxt.text if nxt is not None else None)


def list_all(prefix, cap=20000):
    out, token = [], None
    while True:
        keys, _, token = s3_list(prefix, token=token)
        out.extend(keys)
        if not token or len(out) >= cap:
            break
    return out


def main():
    print("=" * 72)
    print("1. Bucket reachable? top-level prefixes under forecasts-ncov/")
    try:
        _, prefixes, _ = s3_list(PREFIX, delimiter="/")
        print(f"   HTTP 200. {len(prefixes)} prefixes:")
        for p in prefixes:
            print(f"     {p}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        return 1

    print("\n" + "=" * 72)
    print("2. Enumerate dated snapshots")
    keys = list_all(PREFIX)
    print(f"   {len(keys):,} objects, {sum(s for _, s in keys)/1e9:.2f} GB total")

    dated = {}
    for k, s in keys:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", k)
        if m:
            rest = k[len(PREFIX):]
            branch = rest.split("/")[0] if "/" in rest else "(root)"
            dated.setdefault(branch, []).append((m.group(1), k, s))
    for branch, items in sorted(dated.items()):
        ds = sorted({d for d, _, _ in items})
        print(f"   {branch:22s} {len(items):5,} dated objects | "
              f"{len(ds):4d} distinct dates | {ds[0]} .. {ds[-1]}")

    print("\n" + "=" * 72)
    print("3/4. Fetch one snapshot: is it gzipped despite .json, and what fields?")
    cand = [k for k, _ in keys if k.endswith(".json")]
    print(f"   {len(cand):,} .json keys")
    if not cand:
        print("   no .json keys found — the brief's assumption is wrong")
        return 1
    target = sorted(cand)[len(cand) // 2]
    r = requests.get(BUCKET + target, timeout=180)
    raw = r.content
    print(f"   GET {target}")
    print(f"   HTTP {r.status_code}, {len(raw):,} bytes, "
          f"content-encoding={r.headers.get('Content-Encoding')!r}")
    is_gz = raw[:2] == b"\x1f\x8b"
    print(f"   gzip magic bytes present: {is_gz}  <- brief says YES")
    try:
        text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode() if is_gz else raw.decode()
        obj = json.loads(text)
        print(f"   parsed OK. top-level keys: {list(obj)[:10]}")
        meta = obj.get("metadata", {})
        if meta:
            print(f"   metadata keys: {list(meta)[:12]}")
            for f in ("forecast_date", "location", "variant", "dates"):
                if f in meta:
                    v = meta[f]
                    print(f"     {f}: {str(v)[:90]}")
        data = obj.get("data")
        if isinstance(data, list) and data:
            print(f"   data: {len(data):,} records; first record keys: {list(data[0])}")
            sites = {d.get("site") for d in data[:5000] if isinstance(d, dict)}
            print(f"   distinct 'site' values (first 5k): {sorted(x for x in sites if x)[:10]}")
    except Exception as e:
        print(f"   PARSE FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    print("5. GISAID-derived branch without credentials?")
    gis = [k for k, _ in keys if "gisaid" in k.lower()]
    print(f"   {len(gis):,} keys mention gisaid")
    if gis:
        t = sorted(gis)[len(gis) // 2]
        h = requests.head(BUCKET + t, timeout=90)
        print(f"   HEAD {t}\n   -> HTTP {h.status_code}, "
              f"{int(h.headers.get('Content-Length', 0)):,} bytes  (no credential sent)")
    else:
        print("   none found under this prefix — brief's claim needs re-checking")

    os.makedirs("results", exist_ok=True)
    with open("results/verify_inventory.json", "w") as fh:
        json.dump({b: dict(n_objects=len(v),
                           n_dates=len({d for d, _, _ in v}),
                           first=min(d for d, _, _ in v),
                           last=max(d for d, _, _ in v),
                           bytes=sum(s for _, _, s in v))
                   for b, v in dated.items()}, fh, indent=2)
    print("\nwrote results/verify_inventory.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
