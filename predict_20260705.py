# -*- coding: utf-8 -*-
"""Quick prediction for 2026-07-05 21:08 (第2026076期) v3.2"""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from lottery_model import (
    LotteryModelV3, extract_features_enhanced, BlueBallARDual,
    TIAN_GAN, DI_ZHI, TG_WX, DZ_WX, TG_YY,
    WX_ORDER, DZ_CANG_GAN, NAYIN_MAP, get_shishen,
    get_nayin_wuxing, number_features_v32, _compute_multi_period_freqs,
    ZONE1, ZONE2, ZONE3, _number_tiangan
)
from sklearn.ensemble import RandomForestClassifier

# ======= Load data =======
df = pd.read_excel(r'C:\Users\landy\ai\luspace\双色球200期号码.xlsx')
df.columns = ['qi', 'date', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'blue']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f'Loaded {len(df)} draws')
print(f'Date range: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')

# Append latest draws up to 2026075
latest = [
    ('2026069','2026-06-18',12,14,16,17,18,32,8),
    ('2026070','2026-06-21',3,6,8,14,26,27,8),
    ('2026071','2026-06-23',3,8,19,25,31,33,5),
    ('2026072','2026-06-25',7,8,12,15,17,21,1),
    ('2026073','2026-06-28',9,10,13,16,19,21,8),
    ('2026074','2026-06-30',2,23,24,26,28,32,4),
    ('2026075','2026-07-02',8,12,18,21,24,30,1),
]
existing = set(df['qi'].astype(str).values)
for qi, date, r1, r2, r3, r4, r5, r6, blue in latest:
    if str(qi) not in existing:
        new_row = pd.DataFrame([{'qi':qi,'date':pd.Timestamp(date),'r1':r1,'r2':r2,'r3':r3,'r4':r4,'r5':r5,'r6':r6,'blue':blue}])
        df = pd.concat([df, new_row], ignore_index=True)
df = df.sort_values('date').reset_index(drop=True)
print(f'After append: {len(df)} draws')
print(f'Last draw: {df["qi"].iloc[-1]} ({df["date"].iloc[-1].date()}), last blue: {df["blue"].iloc[-1]}')

# ======= Extract features for all rows =======
print('Extracting features...')
all_feats = []
for _, row in df.iterrows():
    f, b, w = extract_features_enhanced(row['date'])
    all_feats.append(f)
print(f'Extracted features for {len(all_feats)} periods')

# ======= Build pointwise ranking data =======
print('Building rank data...')
base_cols = [
    'year','month','day','weekday','day_of_year','solar_term_dist',
    'nianzhu_gan','nianzhu_zhi','yuezhu_gan','yuezhu_zhi',
    'rizhu_gan','rizhu_zhi','shizhu_gan','shizhu_zhi',
    'mu','huo','tu','jin','shui',
    'cg_mu','cg_huo','cg_tu','cg_jin','cg_shui',
    'nayin_nian','nayin_yue','nayin_ri','nayin_shi',
    'shishen_nian','shishen_yue','shishen_shi',
    'nianzhu_yy','rizhu_yy',
]
num_rank_cols = base_cols + [
    'num_gan','num_zhi','num_wx','num_parity',
    'interact_same','interact_isheng','interact_woke','interact_kewo','interact_shengwo',
    'freq_last3','freq_last5','freq_last10','days_cold',
]

rows = []
labels = []
for i in range(len(df)):
    feat = all_feats[i]
    ri_wx_idx = WX_ORDER[TG_WX[feat['rizhu_gan']]]
    freq_features = _compute_multi_period_freqs(df, i) if i > 0 else None
    drawn = set([int(df.iloc[i]['r1']),int(df.iloc[i]['r2']),int(df.iloc[i]['r3']),
                 int(df.iloc[i]['r4']),int(df.iloc[i]['r5']),int(df.iloc[i]['r6'])])
    for n in range(1, 34):
        nf = number_features_v32(n, ri_wx_idx, freq_features)
        x = [feat[c] for c in base_cols] + nf
        rows.append(x)
        labels.append(1 if n in drawn else 0)

X_rank = np.array(rows)
y_rank = np.array(labels)
print(f'Rank data: {X_rank.shape}, positive: {y_rank.sum()} ({y_rank.sum()/len(y_rank)*100:.1f}%)')

# ======= Train red model =======
print('Training red model (RF 200 trees, n_jobs=1)...')
model_red = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=1
)
model_red.fit(X_rank, y_rank)
print('Red model trained')

# ======= Train blue classifier =======
print('Training blue classifier...')
blue_records = []
for i, row in df.iterrows():
    f = all_feats[i].copy()
    f['blue'] = int(row['blue'])
    blue_records.append(f)
df_blue = pd.DataFrame(blue_records)
Xb = df_blue[base_cols].values
yb = df_blue['blue'].values - 1

blue_clf = RandomForestClassifier(
    n_estimators=100, max_depth=6, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=1
)
blue_clf.fit(Xb, yb)
print('Blue classifier trained')

# ======= Blue AR Dual =======
print('Fitting blue AR Dual...')
blue_series = df['blue'].values
blue_ar = BlueBallARDual(short_window=10)
blue_ar.fit(blue_series)
print(f'AR Dual: mu_long={blue_ar.mu_long:.2f}, mu_short={blue_ar.mu_short:.2f}, w_short={blue_ar.w_short:.2f}')

# ======= Compute trend weights =======
print('Computing trend weights...')
n_total = len(df)
weights = {}
for n in range(1, 34):
    count5 = sum(1 for j in range(max(0,n_total-5), n_total)
                if n in set([int(df.iloc[j]['r1']),int(df.iloc[j]['r2']),int(df.iloc[j]['r3']),
                            int(df.iloc[j]['r4']),int(df.iloc[j]['r5']),int(df.iloc[j]['r6'])]))
    count20 = sum(1 for j in range(max(0,n_total-20), n_total)
                 if n in set([int(df.iloc[j]['r1']),int(df.iloc[j]['r2']),int(df.iloc[j]['r3']),
                             int(df.iloc[j]['r4']),int(df.iloc[j]['r5']),int(df.iloc[j]['r6'])]))
    if count5 >= 3: weights[n] = 1.20
    elif count5 >= 2: weights[n] = 1.10
    elif count20 == 0: weights[n] = 1.05
    elif count5 == 0: weights[n] = 0.90
    else: weights[n] = 1.00

hot = [n for n,w in weights.items() if w>=1.15]
warm = [n for n,w in weights.items() if 1.05<=w<1.15]
cool = [n for n,w in weights.items() if w<=0.90]
print(f'Trend HOT (w>=1.15): {hot}')
print(f'Trend warm (w=1.05-1.15): {warm}')
print(f'Trend cool (w<=0.90): {cool}')

# ======= FEATURE IMPORTANCE =======
print()
print('Feature Importance Top-15:')
imp = model_red.feature_importances_
idx = np.argsort(imp)[::-1]
for i in idx[:15]:
    print(f'  {num_rank_cols[i]:25s}: {imp[i]:.4f}')

# ======= Predict for 2026-07-05 =======
print()
print('='*60)
print('=== PREDICTION for 2026-07-05 21:08 (第2026076期) ===')
print('='*60)

target = datetime(2026, 7, 5)
feat_target, bazi_target, wx_target = extract_features_enhanced(target, hour=21)

# Bazi
print()
print('=== 八字分析 ===')
print(f'年柱: {TIAN_GAN[bazi_target["nianzhu_gan"]]}{DI_ZHI[bazi_target["nianzhu_zhi"]]} ({TG_WX[bazi_target["nianzhu_gan"]]}/{DZ_WX[bazi_target["nianzhu_zhi"]]})')
print(f'月柱: {TIAN_GAN[bazi_target["yuezhu_gan"]]}{DI_ZHI[bazi_target["yuezhu_zhi"]]} ({TG_WX[bazi_target["yuezhu_gan"]]}/{DZ_WX[bazi_target["yuezhu_zhi"]]})')
print(f'日柱: {TIAN_GAN[bazi_target["rizhu_gan"]]}{DI_ZHI[bazi_target["rizhu_zhi"]]} ({TG_WX[bazi_target["rizhu_gan"]]}/{DZ_WX[bazi_target["rizhu_zhi"]]}) — 日主{TG_WX[bazi_target["rizhu_gan"]]}')
print(f'时柱: {TIAN_GAN[bazi_target["shizhu_gan"]]}{DI_ZHI[bazi_target["shizhu_zhi"]]} ({TG_WX[bazi_target["shizhu_gan"]]}/{DZ_WX[bazi_target["shizhu_zhi"]]})')

# Wuxing
main_wx = wx_target
cg_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
    for gan in DZ_CANG_GAN[bazi_target[k]]:
        cg_wx[TG_WX[gan]] += 1
print(f'\n=== 五行分布 ===')
print(f'主气: {main_wx}')
print(f'藏干: {cg_wx}')

# 五鼠遁
wushu_map = {0:0, 5:0, 1:2, 6:2, 2:4, 7:4, 3:6, 8:6, 4:8, 9:8}
day_gan_idx = bazi_target['rizhu_gan']
hour_zhi_idx = bazi_target['shizhu_zhi']
correct_g_hour = (wushu_map[day_gan_idx] + hour_zhi_idx) % 10
print(f'五鼠遁修正时柱: {TIAN_GAN[correct_g_hour]}{DI_ZHI[hour_zhi_idx]}')

# Nayin
nayin_names = ['木','火','土','金','水']
print(f'\n纳音:')
for label, g, z in [('年柱', bazi_target['nianzhu_gan'], bazi_target['nianzhu_zhi']),
                     ('月柱', bazi_target['yuezhu_gan'], bazi_target['yuezhu_zhi']),
                     ('日柱', bazi_target['rizhu_gan'], bazi_target['rizhu_zhi']),
                     ('时柱', correct_g_hour, hour_zhi_idx)]:
    ni = get_nayin_wuxing(g, z)
    print(f'  {TIAN_GAN[g]}{DI_ZHI[z]}: {nayin_names[ni]}')

# 十神
ss_names = ['比肩','劫财','食神','伤官','偏财','正财','七杀','正官','偏印','正印']
print(f'\n十神 (日主={TG_WX[day_gan_idx]}):')
for label, g_val in [('年干', bazi_target['nianzhu_gan']),
                      ('月干', bazi_target['yuezhu_gan']),
                      ('时干', correct_g_hour)]:
    ss_idx = get_shishen(day_gan_idx, g_val)
    print(f'  {label} {TIAN_GAN[g_val]} → {ss_names[ss_idx]}')

# Red scores
print(f'\n=== 红球排序 Top-12 得分 ===')
base_data = [feat_target[c] for c in base_cols]
ri_wx_idx_target = WX_ORDER[TG_WX[feat_target['rizhu_gan']]]
freq_features_target = _compute_multi_period_freqs(df, len(df))
scores = []
for n in range(1, 34):
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    prob = model_red.predict_proba(x)[0][1]
    prob *= weights.get(n, 1.0)
    scores.append((n, prob))
scores.sort(key=lambda t: t[1], reverse=True)

for n, s in scores[:12]:
    bar = '█' * int(s * 200)
    print(f'  {n:2d}: {s:.4f} {bar}')

# Blue prediction
print(f'\n=== 蓝球分析 (v3.2) ===')
Xb_target = np.array([[feat_target[c] for c in base_cols]])
blue_proba_raw = blue_clf.predict_proba(Xb_target)[0]
blue_classes = blue_clf.classes_
blue_proba_full = np.zeros(16)
for idx, cls in enumerate(blue_classes):
    if idx < len(blue_proba_raw):
        blue_proba_full[int(cls)] = blue_proba_raw[idx]
blue_clf_argmax = int(np.argmax(blue_proba_full)) + 1
blue_clf_expected = sum((i+1) * blue_proba_full[i] for i in range(16))
blue_ar_val = blue_ar.predict(blue_series[-1])
blue_ens = int(round(np.clip(0.5 * blue_clf_expected + 0.5 * blue_ar_val, 1, 16)))

print(f'  CLF argmax: {blue_clf_argmax}')
print(f'  CLF expected: {blue_clf_expected:.2f}')
print(f'  AR Dual: {blue_ar_val:.2f} (mu_long={blue_ar.mu_long:.2f}, mu_short={blue_ar.mu_short:.2f}, w_short={blue_ar.w_short:.2f})')
print(f'  Ensemble (0.5*exp + 0.5*AR): {blue_ens}')
print(f'  Top-5 prob: ', end='')
top5_blue = [(int(i)+1, float(blue_proba_full[i])) for i in np.argsort(blue_proba_full)[::-1][:5]]
for bn, bp in top5_blue:
    print(f'{bn}({bp:.3f}) ', end='')
print()

# Balanced Top-6 (adaptive zone constraint)
zone1_nums = list(range(1, 12))
zone2_nums = list(range(12, 23))
zone3_nums = list(range(23, 34))
scores_dict = dict(scores)

# Adaptive zone constraint: skip zone if best score < 60% of 12th global score
threshold_12th = scores[11][1] if len(scores) > 11 else 0
zone_threshold = threshold_12th * 0.60

top6 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, scores_dict[n]) for n in z_nums]
    z_list.sort(key=lambda t: t[1], reverse=True)
    best_in_zone = z_list[0][1] if z_list else 0
    if best_in_zone < zone_threshold:
        print(f'  Zone {z} best={best_in_zone:.4f} < threshold={zone_threshold:.4f}, skipping hard constraint')
        continue
    for n, s in z_list:
        if n not in top6:
            top6.add(n)
            break

for n, s in scores:
    if len(top6) >= 6:
        break
    if n not in top6:
        top6.add(n)

red_top6 = sorted(top6)

# === SIX GROUPS ===
print(f'\n=== 六组预测 (v3.2) ===')

# Group 1
print(f'组1 [基准(自适应区域平衡)]: 红球 {red_top6} | 蓝球 {blue_ens:02d}')

# Group 2 - Cold chase
cold_raw = []
for n in range(1, 34):
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    prob = model_red.predict_proba(x)[0][1]
    prob *= weights.get(n, 1.0)
    freq_bonus = sum(freq_features_target[n][:3])/3.0 if freq_features_target else 0
    prob *= (1.0 + (0.3 - 1.0) * freq_bonus)
    cold_raw.append((n, prob))
cold_raw.sort(key=lambda t: t[1], reverse=True)

cold_top6 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, dict(cold_raw)[n]) for n in z_nums]
    z_list.sort(key=lambda t: t[1], reverse=True)
    best_in_z2 = z_list[0][1] if z_list else 0
    if best_in_z2 < zone_threshold:
        continue
    for n, s in z_list:
        if n not in cold_top6:
            cold_top6.add(n)
            break
for n, s in cold_raw:
    if len(cold_top6) >= 6: break
    if n not in cold_top6: cold_top6.add(n)
blue2 = top5_blue[1][0] if len(top5_blue) > 1 else blue_ens
print(f'组2 [追冷号(freqx0.3)]:   红球 {sorted(cold_top6)} | 蓝球 {blue2:02d}')

# Group 3 - Hot chase
hot_raw = []
for n in range(1, 34):
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    prob = model_red.predict_proba(x)[0][1]
    prob *= weights.get(n, 1.0)
    freq_bonus = sum(freq_features_target[n][:3])/3.0 if freq_features_target else 0
    prob *= (1.0 + (3.0 - 1.0) * freq_bonus)
    hot_raw.append((n, prob))
hot_raw.sort(key=lambda t: t[1], reverse=True)

hot_top6 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, dict(hot_raw)[n]) for n in z_nums]
    z_list.sort(key=lambda t: t[1], reverse=True)
    best_in_z3 = z_list[0][1] if z_list else 0
    if best_in_z3 < zone_threshold:
        continue
    for n, s in z_list:
        if n not in hot_top6:
            hot_top6.add(n)
            break
for n, s in hot_raw:
    if len(hot_top6) >= 6: break
    if n not in hot_top6: hot_top6.add(n)
print(f'组3 [追热号(freqx3.0)]:   红球 {sorted(hot_top6)} | 蓝球 {blue_clf_argmax:02d}')

# Group 4 - Balanced 2-2-2
selected4 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, scores_dict[n]) for n in z_nums if n not in selected4]
    z_list.sort(key=lambda t: t[1], reverse=True)
    for n, s in z_list[:2]:
        selected4.add(n)
blue4 = blue_ar.predict_discrete(blue_series[-1])
print(f'组4 [均衡分散(2-2-2)]:    红球 {sorted(selected4)} | 蓝球 {blue4:02d}')

# Group 5 - Wuxing bias
rizhu_wx = TG_WX[feat_target['rizhu_gan']]
wx_boost = {'木':['水','木'],'火':['木','火'],'土':['火','土'],'金':['土','金'],'水':['金','水']}
boost_wx = wx_boost.get(rizhu_wx, ['木','火'])
wx_adj = []
for n in range(1, 34):
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    prob = model_red.predict_proba(x)[0][1]
    prob *= weights.get(n, 1.0)
    n_wx = TG_WX[_number_tiangan(n)]
    if n_wx in boost_wx:
        prob *= 1.08
    wx_adj.append((n, prob))
wx_adj.sort(key=lambda t: t[1], reverse=True)
wx_top6 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, dict(wx_adj)[n]) for n in z_nums]
    z_list.sort(key=lambda t: t[1], reverse=True)
    best_in_z5 = z_list[0][1] if z_list else 0
    if best_in_z5 < zone_threshold:
        continue
    for n, s in z_list:
        if n not in wx_top6:
            wx_top6.add(n)
            break
for n, s in wx_adj:
    if len(wx_top6) >= 6: break
    if n not in wx_top6: wx_top6.add(n)
print(f'组5 [五行偏重(喜{"/".join(boost_wx)})]: 红球 {sorted(wx_top6)} | 蓝球 {int(round(blue_clf_expected)):02d}')

# Group 6 - Weighted random
np.random.seed(20260705 + 5)
top15_nums = [n for n, _ in scores[:15]]
top15_dict = {n: scores_dict[n] for n in top15_nums}
sampled6 = set()
for z_nums, z_want in [(zone1_nums,2),(zone2_nums,2),(zone3_nums,2)]:
    z_c = [n for n in top15_nums if n in z_nums and n not in sampled6]
    if z_c:
        z_p = np.array([top15_dict[n] for n in z_c])
        z_p = z_p / z_p.sum()
        picks = np.random.choice(z_c, size=min(z_want, len(z_c)), replace=False, p=z_p)
        sampled6.update(picks)
for n, _ in scores:
    if len(sampled6) >= 6: break
    if n not in sampled6: sampled6.add(n)

b_nums = [b[0] for b in top5_blue]
b_probs = np.array([b[1] for b in top5_blue])
b_probs = b_probs / b_probs.sum()
blue6 = int(np.random.choice(b_nums, p=b_probs))
print(f'组6 [加权随机采样]:        红球 {sorted(sampled6)} | 蓝球 {blue6:02d}')

# Zone distribution
print(f'\n=== 各预测组分区分布 ===')
all_groups = [
    ('组1', red_top6),
    ('组2', sorted(cold_top6)),
    ('组3', sorted(hot_top6)),
    ('组4', sorted(selected4)),
    ('组5', sorted(wx_top6)),
    ('组6', sorted(sampled6)),
]
for label, reds in all_groups:
    z1 = [n for n in reds if n in zone1_nums]
    z2 = [n for n in reds if n in zone2_nums]
    z3 = [n for n in reds if n in zone3_nums]
    print(f'  {label}: 一区{z1} 二区{z2} 三区{z3}')

# Diversity check
print(f'\n=== 多样性检查 ===')
for i, (label_i, reds_i) in enumerate(all_groups):
    overlaps = []
    for j, (label_j, reds_j) in enumerate(all_groups):
        if i != j:
            overlaps.append(f'{label_j}:{len(set(reds_i) & set(reds_j))}')
    print(f'  {label_i} vs others: {", ".join(overlaps)}')

# ======= Quick Backtest =======
print(f'\n=== 快速回测 (最近30期) ===')
results_bt = []
for i in range(len(df)-30, len(df)):
    train_df = df.iloc[:i]
    test_row = df.iloc[i]

    # Train red
    Xr, yr = [], []
    for j in range(len(train_df)):
        rf = all_feats[j]
        r_wx = WX_ORDER[TG_WX[rf['rizhu_gan']]]
        r_freq = _compute_multi_period_freqs(train_df, j) if j > 0 else None
        r_drawn = set([int(train_df.iloc[j]['r1']),int(train_df.iloc[j]['r2']),int(train_df.iloc[j]['r3']),
                      int(train_df.iloc[j]['r4']),int(train_df.iloc[j]['r5']),int(train_df.iloc[j]['r6'])])
        for n in range(1, 34):
            nf = number_features_v32(n, r_wx, r_freq)
            x = [rf[c] for c in base_cols] + nf
            Xr.append(x)
            yr.append(1 if n in r_drawn else 0)
    Xr, yr = np.array(Xr), np.array(yr)
    rfc = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=1)
    rfc.fit(Xr, yr)

    # Train blue
    br = []
    for j in range(len(train_df)):
        bf = all_feats[j].copy()
        bf['blue'] = int(train_df.iloc[j]['blue'])
        br.append(bf)
    brf = pd.DataFrame(br)
    bfc = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42, n_jobs=1)
    bfc.fit(brf[base_cols].values, brf['blue'].values.astype(int)-1)

    # Blue AR
    bar_model = BlueBallARDual(short_window=10)
    bar_model.fit(train_df['blue'].values)

    # Predict
    tfeat = all_feats[i]
    tbase = [tfeat[c] for c in base_cols]
    t_ri_wx = WX_ORDER[TG_WX[tfeat['rizhu_gan']]]
    t_freq = _compute_multi_period_freqs(train_df, len(train_df))

    tscores = []
    for n in range(1, 34):
        nf = number_features_v32(n, t_ri_wx, t_freq)
        x = np.array([tbase + nf])
        prob = rfc.predict_proba(x)[0][1]
        tscores.append((n, prob))
    tscores.sort(key=lambda t: t[1], reverse=True)
    ts_dict_local = dict(tscores)

    ttop6 = set()
    t_threshold_12th = tscores[11][1] if len(tscores) > 11 else 0
    t_zone_threshold = t_threshold_12th * 0.60
    for z in [1,2,3]:
        z_nums = zone1_nums if z==1 else zone2_nums if z==2 else zone3_nums
        z_list = [(n, ts_dict_local[n]) for n in z_nums]
        z_list.sort(key=lambda t: t[1], reverse=True)
        best_z = z_list[0][1] if z_list else 0
        if best_z < t_zone_threshold:
            continue
        for n,s in z_list:
            if n not in ttop6:
                ttop6.add(n)
                break
    for n,s in tscores:
        if len(ttop6)>=6: break
        if n not in ttop6: ttop6.add(n)

    xb_test = np.array([tbase])
    bp_raw = bfc.predict_proba(xb_test)[0]
    bp_classes = bfc.classes_
    bp_full = np.zeros(16)
    for idx, cls in enumerate(bp_classes):
        if idx < len(bp_raw):
            bp_full[int(cls)] = bp_raw[idx]
    bexp = sum((k+1)*bp_full[k] for k in range(16))
    bar_val = bar_model.predict(train_df['blue'].iloc[-1])
    tblue = int(round(np.clip(0.5*bexp + 0.5*bar_val, 1, 16)))

    actual_red = set([int(test_row['r1']),int(test_row['r2']),int(test_row['r3']),
                      int(test_row['r4']),int(test_row['r5']),int(test_row['r6'])])
    actual_blue = int(test_row['blue'])
    hit_red = len(ttop6 & actual_red)
    hit_blue = 1 if tblue == actual_blue else 0
    results_bt.append({'hit_red':hit_red, 'hit_blue':hit_blue})

red_hits = [r['hit_red'] for r in results_bt]
blue_hits = [r['hit_blue'] for r in results_bt]
print(f'  最近30期红球均值: {np.mean(red_hits):.3f} (随机期望 1.091)')
print(f'  最近30期 >=3命中: {sum(1 for h in red_hits if h>=3)}/30 ({sum(1 for h in red_hits if h>=3)/30*100:.1f}%)')
print(f'  最近30期蓝球命中: {sum(blue_hits)}/30 ({sum(blue_hits)/30*100:.1f}%)')

print()
print('='*60)
print(f'训练样本: {len(df)}期 | 特征: {len(num_rank_cols)}维')
print(f'上期蓝球: {blue_series[-2]} (第{df["qi"].iloc[-2]}期)')
print(f'最新蓝球: {blue_series[-1]} (第{df["qi"].iloc[-1]}期)')
print('⚠️ 纯娱乐预测，请理性购彩！')
