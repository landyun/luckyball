# -*- coding: utf-8 -*-
"""逆向策略分析：对抗冷号偏好，采用热号动量 + 温和逆向"""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from lottery_model import (
    extract_features_enhanced, BlueBallARDual,
    TIAN_GAN, DI_ZHI, TG_WX, DZ_WX, TG_YY,
    WX_ORDER, DZ_CANG_GAN, NAYIN_MAP, get_shishen,
    get_nayin_wuxing, number_features_v32, _compute_multi_period_freqs,
    ZONE1, ZONE2, ZONE3, _number_tiangan, interaction_features
)
from sklearn.ensemble import RandomForestClassifier

# ======= Load data =======
df = pd.read_excel(r'C:\Users\landy\ai\luspace\双色球200期号码.xlsx')
df.columns = ['qi', 'date', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'blue']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

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

n_total = len(df)
print(f'Total draws: {n_total}')
print(f'Last: {df["qi"].iloc[-1]} ({df["date"].iloc[-1].date()}), blue={df["blue"].iloc[-1]}')
print()

# ======= Extract features =======
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

all_feats = []
for _, row in df.iterrows():
    f, b, w = extract_features_enhanced(row['date'])
    all_feats.append(f)

# ======= Train full model =======
print('Training models...')
X_rank, y_rank = [], []
for i in range(n_total):
    feat = all_feats[i]
    ri_wx_idx = WX_ORDER[TG_WX[feat['rizhu_gan']]]
    freq_features = _compute_multi_period_freqs(df, i) if i > 0 else None
    drawn = set([int(df.iloc[i]['r1']),int(df.iloc[i]['r2']),int(df.iloc[i]['r3']),
                 int(df.iloc[i]['r4']),int(df.iloc[i]['r5']),int(df.iloc[i]['r6'])])
    for n in range(1, 34):
        nf = number_features_v32(n, ri_wx_idx, freq_features)
        x = [feat[c] for c in base_cols] + nf
        X_rank.append(x)
        y_rank.append(1 if n in drawn else 0)

X_rank = np.array(X_rank)
y_rank = np.array(y_rank)

model_red = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=1)
model_red.fit(X_rank, y_rank)

# Blue
blue_records = []
for i, row in df.iterrows():
    f = all_feats[i].copy()
    f['blue'] = int(row['blue'])
    blue_records.append(f)
df_blue = pd.DataFrame(blue_records)
blue_clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=1)
blue_clf.fit(df_blue[base_cols].values, df_blue['blue'].values - 1)

blue_series = df['blue'].values
blue_ar = BlueBallARDual(short_window=10)
blue_ar.fit(blue_series)

print('Models trained.\n')

# ======= Target date =======
target = datetime(2026, 7, 5)
feat_target, bazi_target, wx_target = extract_features_enhanced(target, hour=21)
base_data = [feat_target[c] for c in base_cols]
ri_wx_idx_target = WX_ORDER[TG_WX[feat_target['rizhu_gan']]]
freq_features_target = _compute_multi_period_freqs(df, n_total)

# ======= Stats for each number =======
print('=' * 80)
print('号码冷热 & 频率分析（最近10期/5期）')
print('=' * 80)
zone1_nums = list(range(1, 12))
zone2_nums = list(range(12, 23))
zone3_nums = list(range(23, 34))

num_stats = {}
for n in range(1, 34):
    # Count appearances
    count5 = sum(1 for j in range(max(0, n_total - 5), n_total)
                if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']), int(df.iloc[j]['r3']),
                            int(df.iloc[j]['r4']), int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]))
    count10 = sum(1 for j in range(max(0, n_total - 10), n_total)
                 if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']), int(df.iloc[j]['r3']),
                             int(df.iloc[j]['r4']), int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]))
    count3 = sum(1 for j in range(max(0, n_total - 3), n_total)
                if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']), int(df.iloc[j]['r3']),
                            int(df.iloc[j]['r4']), int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]))

    # Days since last appearance
    days_since = None
    for j in range(n_total - 1, -1, -1):
        if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']), int(df.iloc[j]['r3']),
                     int(df.iloc[j]['r4']), int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]):
            days_since = n_total - 1 - j
            break

    # Raw RF score
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    raw_score = model_red.predict_proba(x)[0][1]

    num_stats[n] = {
        'count5': count5, 'count10': count10, 'count3': count3,
        'days_since': days_since if days_since is not None else 999,
        'raw_rf': raw_score,
    }

# Print stats
for zone_name, zone_nums in [('一区(1-11)', zone1_nums), ('二区(12-22)', zone2_nums), ('三区(23-33)', zone3_nums)]:
    print(f'\n  {zone_name}:')
    for n in zone_nums:
        s = num_stats[n]
        hot_label = '🔥' if s['count3'] >= 2 else ('🔶' if s['count3'] >= 1 else ('🔵' if s['days_since'] > 10 else '  '))
        print(f'  {hot_label} {n:2d}: 近5期{s["count5"]}次 近10期{s["count10"]}次 距上次{s["days_since"]:3d}期 RF原始分={s["raw_rf"]:.4f}')

# =====================================================================
# 逆向策略：不同强度的"反冷"权重
# =====================================================================

print()
print('=' * 80)
print('逆向策略预测（反冷号偏好）')
print('=' * 80)

def apply_contrarian_weight(raw_score, stats, strength='moderate'):
    """
    逆向权重：惩罚冷号，奖励热号

    strength levels:
    - 'mild': 轻量逆向
    - 'moderate': 中等逆向
    - 'aggressive': 激进逆向
    """
    days = stats['days_since']
    count3 = stats['count3']
    count5 = stats['count5']

    if strength == 'mild':
        # 冷号降温10-20%，热号加温10-15%
        if days > 20:
            multiplier = 0.80
        elif days > 10:
            multiplier = 0.90
        elif count3 >= 2:
            multiplier = 1.15
        elif count3 >= 1:
            multiplier = 1.08
        else:
            multiplier = 1.00
    elif strength == 'moderate':
        # 冷号降温20-35%，热号加温15-25%
        if days > 20:
            multiplier = 0.65
        elif days > 10:
            multiplier = 0.80
        elif days > 5:
            multiplier = 0.90
        elif count3 >= 2:
            multiplier = 1.25
        elif count3 >= 1:
            multiplier = 1.15
        else:
            multiplier = 1.00
    elif strength == 'aggressive':
        # 激进：只考虑近5期至少出1次的号码
        if days > 15:
            multiplier = 0.40  # 严重惩罚
        elif days > 8:
            multiplier = 0.65
        elif count5 == 0 and days > 5:
            multiplier = 0.75
        elif count3 >= 2:
            multiplier = 1.40  # 强奖励
        elif count3 >= 1:
            multiplier = 1.25
        else:
            multiplier = 1.00

    return raw_score * multiplier


def select_top6(scores_list, zone_constraint='adaptive'):
    """从分数列表中选择 Top-6"""
    scores_dict = dict(scores_list)
    selected = set()

    if zone_constraint == 'adaptive':
        threshold_12th = scores_list[11][1] if len(scores_list) > 11 else 0
        zone_threshold = threshold_12th * 0.50

        for z in [1, 2, 3]:
            z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
            z_list = [(n, scores_dict[n]) for n in z_nums]
            z_list.sort(key=lambda t: t[1], reverse=True)
            if z_list and z_list[0][1] >= zone_threshold:
                for n, s in z_list:
                    if n not in selected:
                        selected.add(n)
                        break

    for n, s in scores_list:
        if len(selected) >= 6:
            break
        if n not in selected:
            selected.add(n)

    return sorted(selected)


# ======= Blue contrarian =======
# 逆向蓝球：CLF argmax 如果是上期蓝球(01)则跳过，选第二选择
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

print(f'\n蓝球分析:')
print(f'  CLF argmax: {blue_clf_argmax} (概率 {blue_proba_full[blue_clf_argmax-1]:.1%})')
print(f'  CLF expected: {blue_clf_expected:.2f}')
print(f'  AR Dual: {blue_ar_val:.2f}')

# 逆向蓝球策略
top5_blue = [(int(i)+1, float(blue_proba_full[i])) for i in np.argsort(blue_proba_full)[::-1][:5]]
print(f'  CLF Top-5: {top5_blue}')

# 排除CLF argmax（如果是01重号预测），或取AR离散值
blue_contrarian_mild = int(round(blue_ar_val))  # 纯AR
blue_contrarian_mod = top5_blue[1][0] if top5_blue[0][0] == 1 else top5_blue[0][0]  # 跳过01
blue_contrarian_agg = int(round(blue_clf_expected))  # 期望值
print(f'  逆向蓝球: mild(纯AR)={blue_contrarian_mild}, mod(跳过01)={blue_contrarian_mod}, agg(期望值)={blue_contrarian_agg}')

# ======= Generate contrarian predictions =======
print()
print('-' * 80)

strategies = [
    ('逆向-轻度', 'mild', '适度惩罚冷号(-20%)，适度奖励热号(+15%)'),
    ('逆向-中度', 'moderate', '较重惩罚冷号(-35%)，较重奖励热号(+25%)'),
    ('逆向-激进', 'aggressive', '严厉惩罚冷号(-60%)，强力奖励热号(+40%)'),
]

# Get original ranking for comparison
orig_scores = []
for n in range(1, 34):
    nf = number_features_v32(n, ri_wx_idx_target, freq_features_target)
    x = np.array([base_data + nf])
    prob = model_red.predict_proba(x)[0][1]
    orig_scores.append((n, prob))
orig_scores.sort(key=lambda t: t[1], reverse=True)

# Also compute hot momentum scores (independent of RF)
# Based purely on recent frequency
print()
print('=== 纯热度排名（无模型，仅统计近10期频率） ===')
hot_scores = []
for n in range(1, 34):
    s = num_stats[n]
    # Score = weighted recent appearances
    hot_score = s['count3'] * 3.0 + s['count5'] * 1.5 + s['count10'] * 0.5
    # Penalize very long absences
    if s['days_since'] > 30:
        hot_score *= 0.5
    hot_scores.append((n, hot_score))
hot_scores.sort(key=lambda t: t[1], reverse=True)
for n, s in hot_scores[:15]:
    bar = '█' * int(s * 5)
    print(f'  {n:2d}: {s:.1f} {bar}')

# Pure hot Top-6
hot_top6 = set()
for z in [1, 2, 3]:
    z_nums = zone1_nums if z == 1 else zone2_nums if z == 2 else zone3_nums
    z_list = [(n, s) for n, s in hot_scores if n in z_nums]
    for n, s in z_list[:2]:
        if n not in hot_top6:
            hot_top6.add(n)
            if len([x for x in hot_top6 if x in z_nums]) >= 2:
                break
hot_pure = sorted(hot_top6)

print(f'\n  纯热度 Top-6 (2-2-2): {hot_pure}')

# ======= All strategies =======
print()
print('=' * 80)
print('=== 最终推荐组合 ===')
print('=' * 80)

all_predictions = []

# Strategy A: 逆向-轻度
print(f'\n【组A】逆向轻度 — 惩罚冷号-20%，奖励热号+15%')
scores_a = [(n, apply_contrarian_weight(num_stats[n]['raw_rf'], num_stats[n], 'mild')) for n in range(1, 34)]
scores_a.sort(key=lambda t: t[1], reverse=True)
top6_a = select_top6(scores_a)
print(f'  红球: {top6_a}')
print(f'  蓝球: {blue_contrarian_mild:02d} (纯AR，抵抗CLF重号偏见)')
all_predictions.append(('A', '逆向轻度', top6_a, blue_contrarian_mild))

# Strategy B: 逆向-中度
print(f'\n【组B】逆向中度 — 惩罚冷号-35%，奖励热号+25%')
scores_b = [(n, apply_contrarian_weight(num_stats[n]['raw_rf'], num_stats[n], 'moderate')) for n in range(1, 34)]
scores_b.sort(key=lambda t: t[1], reverse=True)
top6_b = select_top6(scores_b)
print(f'  红球: {top6_b}')
print(f'  蓝球: {blue_contrarian_mod:02d} (跳过CLF argmax 01，取第二选择)')
all_predictions.append(('B', '逆向中度', top6_b, blue_contrarian_mod))

# Strategy C: 逆向-激进
print(f'\n【组C】逆向激进 — 只考虑近5期有出现的号码')
scores_c = [(n, apply_contrarian_weight(num_stats[n]['raw_rf'], num_stats[n], 'aggressive')) for n in range(1, 34)]
scores_c.sort(key=lambda t: t[1], reverse=True)
top6_c = select_top6(scores_c)
print(f'  红球: {top6_c}')
print(f'  蓝球: {blue_contrarian_agg:02d} (CLF期望值，抵抗重号+抵抗极端AR)')
all_predictions.append(('C', '逆向激进', top6_c, blue_contrarian_agg))

# Strategy D: 纯热度动量（不用RF，只看频率）
print(f'\n【组D】纯热度动量 — 完全不用ML，只看近期频率统计')
top6_d = hot_pure
# Blue: most frequent in recent 10 draws
blue_counts = {}
for j in range(max(0, n_total - 10), n_total):
    b = int(df.iloc[j]['blue'])
    blue_counts[b] = blue_counts.get(b, 0) + 1
blue_d = max(blue_counts, key=blue_counts.get)
print(f'  红球: {top6_d}')
print(f'  蓝球: {blue_d:02d} (近10期最频繁蓝球)')
all_predictions.append(('D', '纯热度动量', top6_d, blue_d))

# Strategy E: 混合 — 原始RF分 + 热度分 加权
print(f'\n【组E】混合加权 — RF原始分(30%) + 热度分(70%)')
scores_e = []
max_rf = max(s[1] for s in orig_scores)
max_hot = max(s[1] for s in hot_scores)
for n in range(1, 34):
    rf_norm = num_stats[n]['raw_rf'] / max_rf
    hot_norm = dict(hot_scores)[n] / max_hot if max_hot > 0 else 0
    mixed = 0.30 * rf_norm + 0.70 * hot_norm
    scores_e.append((n, mixed))
scores_e.sort(key=lambda t: t[1], reverse=True)
top6_e = select_top6(scores_e)
# Blue: ensemble of CLF expected with heavier AR
blue_e = int(round(0.3 * blue_clf_expected + 0.7 * blue_ar_val))
print(f'  红球: {top6_e}')
print(f'  蓝球: {blue_e:02d} (30%CLF + 70%AR，严重偏向均值回归)')
all_predictions.append(('E', '混合加权(RF30%+热度70%)', top6_e, blue_e))

# ======= Summary table =======
print()
print('=' * 80)
print('=== 六组预测汇总（含原始基准组对比）===')
print('=' * 80)
print()
print('| 组别 | 策略 | 红球 | 蓝球 | 设计理念 |')
print('|:--:|:--|------|:--:|:--|')
print(f'| 组1 | 原始基准(v3.2) | {sorted([3,11,14,25,31,33])} | 09 | 追冷号（对照） |')
for label, strategy, reds, blue in all_predictions:
    reds_str = '、'.join(f'{n:02d}' for n in reds)
    print(f'| 组{label} | {strategy} | {reds_str} | {blue:02d} | — |')

# ======= Zone distribution =======
print()
print('=== 各策略分区分布 ===')
print(f'  原始基准: 一区{[3,11]} 二区{[14]} 三区{[25,31,33]}')
for label, strategy, reds, blue in all_predictions:
    z1 = [n for n in reds if n in zone1_nums]
    z2 = [n for n in reds if n in zone2_nums]
    z3 = [n for n in reds if n in zone3_nums]
    print(f'  组{label}: 一区{z1} 二区{z2} 三区{z3}')

# ======= Key differences analysis =======
print()
print('=== 逆向策略与原始基准的关键差异 ===')
orig_set = set([3, 11, 14, 25, 31, 33])
for label, strategy, reds, blue in all_predictions:
    new_set = set(reds)
    added = new_set - orig_set
    removed = orig_set - new_set
    kept = new_set & orig_set
    if added or removed:
        print(f'  组{label}: +{sorted(added)} -{sorted(removed)} 保留{sorted(kept)}')

# ======= Recent pattern analysis =======
print()
print('=== 近期走势特征 ===')
# Check consecutive numbers in recent draws
recent_reds = []
for j in range(n_total - 10, n_total):
    reds = sorted([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']), int(df.iloc[j]['r3']),
                   int(df.iloc[j]['r4']), int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])])
    recent_reds.append(reds)

# Consecutive pairs in last 10 draws
all_consec = []
for reds in recent_reds:
    for i in range(len(reds)-1):
        if reds[i+1] - reds[i] == 1:
            all_consec.append((reds[i], reds[i+1]))
print(f'  近10期连号: {all_consec}')
print(f'  连号频率: {len(all_consec)}/10期')

# Zone distribution recently
zone_counts = {1: 0, 2: 0, 3: 0}
for reds in recent_reds:
    for r in reds:
        if r <= 11: zone_counts[1] += 1
        elif r <= 22: zone_counts[2] += 1
        else: zone_counts[3] += 1
print(f'  近10期分区分布: 一区{zone_counts[1]}个({zone_counts[1]/60*100:.0f}%) 二区{zone_counts[2]}个({zone_counts[2]/60*100:.0f}%) 三区{zone_counts[3]}个({zone_counts[3]/60*100:.0f}%)')
print(f'  理论分布: 一区20个(33%) 二区20个(33%) 三区20个(33%)')

# Odd/even recently
odd_count = sum(1 for reds in recent_reds for r in reds if r % 2 == 1)
even_count = sum(1 for reds in recent_reds for r in reds if r % 2 == 0)
print(f'  近10期奇偶比: {odd_count}:{even_count} (奇{odd_count/60*100:.0f}% 偶{even_count/60*100:.0f}%)')

# Repeat numbers (numbers that appeared in consecutive draws)
repeats = []
for j in range(len(recent_reds)-1):
    rp = set(recent_reds[j]) & set(recent_reds[j+1])
    if rp:
        repeats.extend(rp)
print(f'  近10期重号: {sorted(set(repeats))} (共{len(repeats)}次重号事件)')

print()
print('⚠️ 纯娱乐分析，请理性购彩！')
