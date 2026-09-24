"""
Deep diagnostic: look at missed pairs after deterministic blocking
to understand WHY keys don't match.
"""
import pandas as pd, sys, re
from collections import defaultdict

sys.path.insert(0, '.')

TRAIN = 'datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train'
s1 = pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t')
s2 = pd.read_csv(f'{TRAIN}/train_source2.tsv', sep='\t')
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t')
print(f"Loaded S1={len(s1):,} S2={len(s2):,}")

# Use tiny sample for speed
import random; random.seed(42)
s1_sample = s1.sample(500, random_state=42)

from src.blocking.blocking_deterministic import build_inverted_index_fast, _compute_keys

# Build index from a small S2 sample
s2_sample_ids = set(s2.sample(50000, random_state=42)['entity_id'])

# Build full GT dict
eids_gt = gt['source1_entity_id'].tolist()
raws_gt = gt['matched_entity_ids'].fillna('').astype(str).tolist()
gt_dict = {}
for eid, raw in zip(eids_gt, raws_gt):
    raw = raw.strip()
    gt_dict[eid] = set(x.strip() for x in raw.split(',') if x.strip()) if raw and raw != 'nan' else set()

# For 20 random S1 entities with S2 matches, compare keys
print("\n=== MISSED PAIR KEY ANALYSIS ===")
count = 0
s2_map = s2.set_index('entity_id').to_dict('index')
s1_map = s1.set_index('entity_id').to_dict('index')

LEGAL = re.compile(
    r"\b(pvt\.?\s*ltd\.?|private\s+limited|private\s+ltd\.?|p\.?\s*ltd\.?|"
    r"llp|llc|inc\.?|corp\.?|corporation|limited|ltd\.?|co\.?|company|"
    r"enterprises?|industries|industry|group|holdings?|trading|traders?|"
    r"distributors?|solutions?|technologies|technology|tech|services?|"
    r"international|intl\.?|s\.a\.s|s\.a\.|sarl|sas|eurl|srl|"
    r"proprietorship|proprietor|prop\.?|& sons|and sons|brothers|bros\.?|"
    r"pllc|plc|associates?|association|foundation|trust|school|college|"
    r"hospital|clinic|center|centre)\b", re.IGNORECASE
)

def norm_name(n):
    n = str(n).lower()
    n = LEGAL.sub(' ', n)
    n = re.sub(r'[^\w\s]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n

def get_keys(name, addr, country):
    nc = norm_name(name).replace(' ', '')
    na = re.sub(r'[^\w\s]', ' ', str(addr).lower()).replace(' ', '')
    ct = str(country).lower().strip()
    nm = norm_name(name)
    tokens = sorted(nm.split())[:5]
    kd = ' '.join(tokens)
    return {
        'A': f"{ct}|{nc[:8]}",
        'B': f"{ct}|{na[:8]}",
        'C': nc[:10],
        'D': kd,
        'E': f"{nc[:6]}|{na[:6]}",
        'F': f"{nc[:5]}|{ct}",
    }

rows = gt.sample(200, random_state=7)
for _, gtr in rows.iterrows():
    s1_eid = gtr['source1_entity_id']
    matches = [m.strip() for m in str(gtr.get('matched_entity_ids','')).split(',')
               if m.strip().startswith('S2-')][:1]
    if not matches or s1_eid not in s1_map: continue
    m_eid = matches[0]
    if m_eid not in s2_map: continue

    r1, r2 = s1_map[s1_eid], s2_map[m_eid]
    k1 = get_keys(r1['business_name'], r1['business_address'], r1['country'])
    k2 = get_keys(r2['business_name'], r2['business_address'], r2['country'])

    shared = [k for k in k1 if k1[k] == k2[k] and len(k1[k]) > 3]
    if not shared:
        print(f"MISS — no shared key:")
        print(f"  S1: {r1['business_name']!r}")
        print(f"  S2: {r2['business_name']!r}")
        print(f"  S1 keys: {k1}")
        print(f"  S2 keys: {k2}")
        print()
        count += 1
        if count >= 20: break

print(f"\nTotal missed (no shared key): {count}/200 sampled pairs")
