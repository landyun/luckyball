# -*- coding: utf-8 -*-
"""v3.3 双策略预测模板 — 冷号追踪 + 热号动量
Usage: 修改 target_date 和 latest_draws, 然后运行
"""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from lottery_model import (
    LotteryModelV3, extract_features_enhanced,
    TIAN_GAN, DI_ZHI, TG_WX, DZ_WX, TG_YY,
    WX_ORDER, DZ_CANG_GAN, get_shishen, get_nayin_wuxing,
    _compute_number_frequency_counts, _compute_hot_momentum_scores,
    ZONE1, ZONE2, ZONE3,
)

# =====================================================================
# CONFIG: 修改这里
# =====================================================================
TARGET_DATE = datetime(2026, 7, 5)
TARGET_HOUR = 21
PERIOD_LABEL = '2026076'
DATA_FILE = r'C:\Users\landy\ai\luspace\双色球200期号码.xlsx'

# 最近未在 Excel 中的开奖结果 (qi, date_str, r1-r6, blue)
LATEST_DRAWS = [
    ('2026069','2026-06-18',12,14,16,17,18,32,8),
    ('2026070','2026-06-21',3,6,8,14,26,27,8),
    ('2026071','2026-06-23',3,8,19,25,31,33,5),
    ('2026072','2026-06-25',7,8,12,15,17,21,1),
    ('2026073','2026-06-28',9,10,13,16,19,21,8),
    ('2026074','2026-06-30',2,23,24,26,28,32,4),
    ('2026075','2026-07-02',8,12,18,21,24,30,1),
]

# =====================================================================
# 数据加载
# =====================================================================
print('=' * 70)
print(f'双色球预测 v3.3 — 冷热双策略')
print(f'预测日期: {TARGET_DATE.strftime("%Y-%m-%d")} {TARGET_HOUR}:00+ (第{PERIOD_LABEL}期)')
print('=' * 70)

df = pd.read_excel(DATA_FILE)
df.columns = ['qi', 'date', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'blue']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f'\nLoaded {len(df)} draws from Excel')

existing = set(df['qi'].astype(str).values)
for qi, date, r1, r2, r3, r4, r5, r6, blue in LATEST_DRAWS:
    if str(qi) not in existing:
        new_row = pd.DataFrame([{'qi':qi,'date':pd.Timestamp(date),
                                 'r1':r1,'r2':r2,'r3':r3,'r4':r4,'r5':r5,'r6':r6,'blue':blue}])
        df = pd.concat([df, new_row], ignore_index=True)
df = df.sort_values('date').reset_index(drop=True)
n_total = len(df)
print(f'After append: {n_total} draws, latest: {df["qi"].iloc[-1]} ({df["date"].iloc[-1].date()})')

# =====================================================================
# 八字分析
# =====================================================================
feat_target, bazi_target, wx_target = extract_features_enhanced(TARGET_DATE, hour=TARGET_HOUR)
day_gan_idx = bazi_target['rizhu_gan']
rizhu_wx = TG_WX[day_gan_idx]

# 五鼠遁
wushu_map = {0:0, 5:0, 1:2, 6:2, 2:4, 7:4, 3:6, 8:6, 4:8, 9:8}
hour_zhi_idx = bazi_target['shizhu_zhi']
correct_g_hour = (wushu_map[day_gan_idx] + hour_zhi_idx) % 10

print(f'\n=== 八字分析 ===')
print(f'年柱: {TIAN_GAN[bazi_target["nianzhu_gan"]]}{DI_ZHI[bazi_target["nianzhu_zhi"]]} '
      f'({TG_WX[bazi_target["nianzhu_gan"]]}/{DZ_WX[bazi_target["nianzhu_zhi"]]})')
print(f'月柱: {TIAN_GAN[bazi_target["yuezhu_gan"]]}{DI_ZHI[bazi_target["yuezhu_zhi"]]} '
      f'({TG_WX[bazi_target["yuezhu_gan"]]}/{DZ_WX[bazi_target["yuezhu_zhi"]]})')
print(f'日柱: {TIAN_GAN[day_gan_idx]}{DI_ZHI[bazi_target["rizhu_zhi"]]} '
      f'({rizhu_wx}/{DZ_WX[bazi_target["rizhu_zhi"]]}) — 日主{rizhu_wx}')
print(f'时柱: {TIAN_GAN[correct_g_hour]}{DI_ZHI[hour_zhi_idx]} '
      f'({TG_WX[correct_g_hour]}/{DZ_WX[hour_zhi_idx]}) [五鼠遁修正]')

# 五行
main_wx = wx_target
cg_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
    for gan in DZ_CANG_GAN[bazi_target[k]]:
        cg_wx[TG_WX[gan]] += 1
print(f'\n五行 主气: {main_wx}')
print(f'五行 藏干: {cg_wx}')

# 纳音
nayin_names = ['木','火','土','金','水']
print(f'\n纳音:')
for label, g, z in [
    ('年柱', bazi_target['nianzhu_gan'], bazi_target['nianzhu_zhi']),
    ('月柱', bazi_target['yuezhu_gan'], bazi_target['yuezhu_zhi']),
    ('日柱', day_gan_idx, bazi_target['rizhu_zhi']),
    ('时柱', correct_g_hour, hour_zhi_idx)]:
    ni = get_nayin_wuxing(g, z)
    print(f'  {TIAN_GAN[g]}{DI_ZHI[z]}: {nayin_names[ni]}')

# 十神
ss_names = ['比肩','劫财','食神','伤官','偏财','正财','七杀','正官','偏印','正印']
print(f'\n十神 (日主={rizhu_wx}):')
for label, g_val in [('年干', bazi_target['nianzhu_gan']),
                      ('月干', bazi_target['yuezhu_gan']),
                      ('时干', correct_g_hour)]:
    ss_idx = get_shishen(day_gan_idx, g_val)
    print(f'  {label} {TIAN_GAN[g_val]} → {ss_names[ss_idx]}')

# =====================================================================
# 训练 & 预测
# =====================================================================
print(f'\n{"="*70}')
print('训练模型 & 双策略预测...')
print('=' * 70)

model = LotteryModelV3()
model.df_history = df

# 手动训练 (精简版, 避免调用 train() 中的 Excel 路径)
from sklearn.ensemble import RandomForestClassifier
from lottery_model import (
    number_features_v32, _compute_multi_period_freqs, BlueBallARDual,
)

# Feature column definitions (matching LotteryModelV3.feature_cols_base)
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

# Build rank data
all_feats = []
for _, row in df.iterrows():
    f, b, w = extract_features_enhanced(row['date'])
    all_feats.append(f)

X_rank, y_rank = [], []
for i in range(len(df)):
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

model.model_red = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=1)
model.model_red.fit(np.array(X_rank), np.array(y_rank))

# Blue classifier
blue_records = []
for i, row in df.iterrows():
    f = all_feats[i].copy()
    f['blue'] = int(row['blue'])
    blue_records.append(f)
df_blue = pd.DataFrame(blue_records)
model.model_blue_clf = RandomForestClassifier(
    n_estimators=100, max_depth=6, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=1)
model.model_blue_clf.fit(df_blue[base_cols].values, df_blue['blue'].values.astype(int) - 1)

# Blue AR
blue_series = df['blue'].values
model.model_blue_ar = BlueBallARDual(short_window=10)
model.model_blue_ar.fit(blue_series)

# Trend weights
model._compute_trend_weights()

print(f'训练完成: {len(df)} 期, 蓝球AR mu_long={model.model_blue_ar.mu_long:.2f}, '
      f'mu_short={model.model_blue_ar.mu_short:.2f}, w_short={model.model_blue_ar.w_short:.2f}')

# =====================================================================
# 双策略预测
# =====================================================================
cold, hot = model.predict_multi(TARGET_DATE, hour=TARGET_HOUR)

# =====================================================================
# 输出: 冷号策略
# =====================================================================
print(f'\n{"="*70}')
print(cold['label'])
print('=' * 70)
print(f'\n| 组别 | 策略 | 红球 | 蓝球 |')
print(f'|:--:|:--|------|:--:|')
for g in cold['groups']:
    reds = '、'.join(f'{n:02d}' for n in g['pred_red'])
    print(f'| {g["index"]} | {g["strategy"]} | {reds} | {g["pred_blue"]:02d} |')

# 分区分布
print(f'\n冷号策略分区:')
for g in cold['groups']:
    z1 = [n for n in g['pred_red'] if n in ZONE1]
    z2 = [n for n in g['pred_red'] if n in ZONE2]
    z3 = [n for n in g['pred_red'] if n in ZONE3]
    print(f'  {g["index"]}: 一区{z1} 二区{z2} 三区{z3}')

# =====================================================================
# 输出: 热号策略
# =====================================================================
print(f'\n{"="*70}')
print(hot['label'])
print('=' * 70)
print(f'\n| 组别 | 策略 | 红球 | 蓝球 |')
print(f'|:--:|:--|------|:--:|')
for g in hot['groups']:
    reds = '、'.join(f'{n:02d}' for n in g['pred_red'])
    print(f'| {g["index"]} | {g["strategy"]} | {reds} | {g["pred_blue"]:02d} |')

# 分区分布
print(f'\n热号策略分区:')
for g in hot['groups']:
    z1 = [n for n in g['pred_red'] if n in ZONE1]
    z2 = [n for n in g['pred_red'] if n in ZONE2]
    z3 = [n for n in g['pred_red'] if n in ZONE3]
    print(f'  {g["index"]}: 一区{z1} 二区{z2} 三区{z3}')

# =====================================================================
# 策略对比
# =====================================================================
print(f'\n{"="*70}')
print('策略对比: 冷号 vs 热号')
print('=' * 70)

cold_best = cold['groups'][0]['pred_red']
hot_best = hot['groups'][0]['pred_red']
overlap = set(cold_best) & set(hot_best)
cold_only = set(cold_best) - set(hot_best)
hot_only = set(hot_best) - set(cold_best)

print(f'  冷号C1: {cold_best}')
print(f'  热号H1: {hot_best}')
print(f'  共同选择: {sorted(overlap) if overlap else "无"} ({len(overlap)}/6)')
print(f'  仅冷号选: {sorted(cold_only)}')
print(f'  仅热号选: {sorted(hot_only)}')
print(f'  差异化: {len(hot_only)}/6 — {"✅ 真差异化" if len(overlap) <= 3 else "⚠️ 重叠过多"}')

# =====================================================================
# 频率统计
# =====================================================================
print(f'\n{"="*70}')
print('近期热度排名 (纯频率, 不用ML)')
print('=' * 70)

freq_counts = hot['freq_stats']
hot_ranking = _compute_hot_momentum_scores(freq_counts, variant='default')
print(f'\nTop-12 热度号码:')
for n, s in hot_ranking[:12]:
    fc = freq_counts[n]
    bar = '█' * int(s * 3)
    status = '🔥' if fc['count3'] >= 2 else ('🔶' if fc['count3'] >= 1 else '  ')
    print(f'  {status} {n:2d}: {s:5.1f} (近3期{fc["count3"]}次 近5期{fc["count5"]}次 距上次{fc["days_since"]}期) {bar}')

# =====================================================================
# 蓝球分析
# =====================================================================
base = cold['base_prediction']
print(f'\n{"="*70}')
print('蓝球分析')
print('=' * 70)
print(f'  CLF argmax: {base["blue_clf_argmax"]} (概率 {base["blue_proba_top5"][0][1]:.1%})')
print(f'  CLF expected: {base["blue_clf_expected"]:.2f}')
print(f'  AR Dual: {base["blue_ar"]:.2f}')
print(f'  冷号Ensemble: {base["pred_blue"]}')
print(f'  Top-5: {base["blue_proba_top5"]}')
print(f'  上期蓝球: {df["blue"].iloc[-1]} (第{df["qi"].iloc[-1]}期)')

blue_freq = hot['blue_freq']
print(f'\n蓝球近10期频率:')
for b in sorted(range(1, 17), key=lambda x: blue_freq[x]['count10'], reverse=True)[:8]:
    print(f'  {b:2d}: 近5期{blue_freq[b]["count5"]}次 近10期{blue_freq[b]["count10"]}次')

# =====================================================================
# 模型状态
# =====================================================================
print(f'\n{"="*70}')
print(f'模型: v3.3 | 训练样本: {len(df)}期 | 特征: {len(model.feature_cols_rank)}维')
print(f'蓝球 AR: mu_long={model.model_blue_ar.mu_long:.2f}, '
      f'mu_short={model.model_blue_ar.mu_short:.2f}, '
      f'w_short={model.model_blue_ar.w_short:.2f}')
print(f'上期蓝球: {df["blue"].iloc[-1]} (第{df["qi"].iloc[-1]}期)')
print(f'\n⚠️ 纯娱乐预测，请理性购彩！')
