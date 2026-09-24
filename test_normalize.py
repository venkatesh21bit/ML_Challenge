"""
Quick smoke test for normalization + deterministic blocking.
Uses only stdlib + pandas — no sklearn/faiss needed.
Run: python test_normalize.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.data.normalize import (
    normalize_name, normalize_address, build_blocking_keys, build_combined_text
)

# Sample records mimicking the real dataset patterns
samples = [
    {"entity_id": "S1-001", "business_name": "Sri Lakshmi Electronics Pvt Ltd", "business_address": "12 MG Road, Coimbatore", "country": "India"},
    {"entity_id": "S2-001", "business_name": "Sri Laxmi Electronics P Ltd",     "business_address": "12 M.G. Rd Coimbatore",   "country": "India"},
    {"entity_id": "S2-002", "business_name": "Sri Lakshmi Electricals",          "business_address": "12 Mahatma Gandhi Road",   "country": "India"},
    {"entity_id": "S1-002", "business_name": "Apple Inc",                        "business_address": "1 Infinite Loop Cupertino CA", "country": "US"},
    {"entity_id": "S2-003", "business_name": "Apple Incorporated",               "business_address": "1 Infinite Loop Cupertino",    "country": "US"},
    {"entity_id": "S1-003", "business_name": "Samsung Electronics Co",           "business_address": "Seoul South Korea",             "country": "unknown"},
    {"entity_id": "S2-004", "business_name": "Samzung Electronics",              "business_address": "Seoul Korea",                   "country": "unknown"},
]

print("=" * 70)
print("NORMALIZE TEST")
print("=" * 70)
for s in samples:
    name = normalize_name(s["business_name"])
    addr = normalize_address(s["business_address"])
    keys = build_blocking_keys(s)
    txt  = build_combined_text(s)
    print(f"\n{s['entity_id']}: {s['business_name']}")
    print(f"  norm_name : '{name}'")
    print(f"  norm_addr : '{addr}'")
    print(f"  block_keys: {keys}")
    print(f"  combined  : '{txt[:80]}'")

print("\n" + "=" * 70)
print("BLOCKING KEY INTERSECTION TEST")
print("=" * 70)

# Check if S1-001 and S2-001 share any blocking key (they should!)
keys_s1 = set(build_blocking_keys(samples[0]))
keys_s2 = set(build_blocking_keys(samples[1]))
overlap  = keys_s1 & keys_s2
print(f"\nS1-001 keys: {sorted(keys_s1)}")
print(f"S2-001 keys: {sorted(keys_s2)}")
print(f"OVERLAP    : {sorted(overlap)}")
if overlap:
    print("✅ True pair (Sri Lakshmi / Sri Laxmi) shares blocking key!")
else:
    print("❌ No overlap — these would be MISSED by deterministic blocking!")

# Check Apple Inc vs Apple Incorporated
keys_apple1 = set(build_blocking_keys(samples[3]))
keys_apple2 = set(build_blocking_keys(samples[4]))
overlap2 = keys_apple1 & keys_apple2
print(f"\nApple Inc keys       : {sorted(keys_apple1)}")
print(f"Apple Incorporated   : {sorted(keys_apple2)}")
print(f"OVERLAP              : {sorted(overlap2)}")
if overlap2:
    print("✅ Apple pair shares blocking key!")
else:
    print("⚠️  Apple pair missed by deterministic — TF-IDF layer needed")

print("\n✅ Smoke test done.")
