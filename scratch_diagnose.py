import pandas as pd, random

TRAIN = 'datasets/6ab10eb3b23ba_student_resource/student_resource/dataset/train'
s1 = pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t')
s2 = pd.read_csv(f'{TRAIN}/train_source2.tsv', sep='\t')
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t')
print(f"Loaded: S1={len(s1):,} S2={len(s2):,} GT={len(gt):,}")

s1m = s1.set_index('entity_id').to_dict('index')
s2m = s2.set_index('entity_id').to_dict('index')

# Sample 20 true pairs and show name/address side by side
count = 0
rows = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')].sample(50, random_state=42)
for _, gtr in rows.iterrows():
    s1_eid = gtr['source1_entity_id']
    matches = str(gtr['matched_entity_ids']).split(',')
    s2_matches = [m.strip() for m in matches if m.strip().startswith('S2-')][:1]
    if not s2_matches or s1_eid not in s1m:
        continue
    m_eid = s2_matches[0]
    if m_eid not in s2m:
        continue
    r1, r2 = s1m[s1_eid], s2m[m_eid]
    n1, n2 = str(r1['business_name']), str(r2['business_name'])
    a1, a2 = str(r1['business_address']), str(r2['business_address'])
    print(f"S1 name: {n1}")
    print(f"S2 name: {n2}")
    print(f"S1 addr: {a1[:60]}")
    print(f"S2 addr: {a2[:60]}")
    # Common prefix length on normalized names
    n1l, n2l = n1.lower().replace(' ',''), n2.lower().replace(' ','')
    cp = sum(1 for a,b in zip(n1l, n2l) if a == b and all(x == y for x,y in zip(n1l[:_+1], n2l[:_+1]) for _ in range(len(n1l[:len(n2l)]))))
    pref = 0
    for a, b in zip(n1l, n2l):
        if a == b: pref += 1
        else: break
    print(f"Name compact prefix match: {pref} chars  |  n1={n1l[:10]!r} n2={n2l[:10]!r}")
    print()
    count += 1
    if count >= 15:
        break
