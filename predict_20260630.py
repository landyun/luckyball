# -*- coding: utf-8 -*-
"""预测 2026-06-30 第2026074期 双色球 (v3.2)"""
import sys
sys.path.insert(0, r'C:\Users\landy\ai\.claude\skills\luckyball')

import numpy as np
import pandas as pd
from datetime import datetime
from lottery_model import (
    LotteryModelV3, extract_features_enhanced,
    TIAN_GAN, DI_ZHI, TG_WX, DZ_WX, TG_YY,
    WX_ORDER, DZ_CANG_GAN, NAYIN_MAP, get_shishen,
    get_nayin_wuxing, number_features_v32, _compute_multi_period_freqs,
    ZONE1, ZONE2, ZONE3
)

# ============ 0. Load & prepare data (sorted ascending) ============
model = LotteryModelV3()
model.load_history(r'C:\Users\landy\ai\luspace\双色球200期号码.xlsx')
model.df_history['date'] = pd.to_datetime(model.df_history['date'])
model.df_history = model.df_history.sort_values('date').reset_index(drop=True)

# Append latest draws up to 2026073
latest_draws = [
    ('2026069', '2026-06-18', 12, 14, 16, 17, 18, 32, 8),
    ('2026070', '2026-06-21', 3, 6, 8, 14, 26, 27, 8),
    ('2026071', '2026-06-23', 3, 8, 19, 25, 31, 33, 5),
    ('2026072', '2026-06-25', 7, 8, 12, 15, 17, 21, 1),
    ('2026073', '2026-06-28', 9, 10, 13, 16, 19, 21, 8),
]

existing_qis = set(model.df_history['qi'].astype(str).values)
for qi, date, r1, r2, r3, r4, r5, r6, blue in latest_draws:
    if str(qi) not in existing_qis:
        new_row = pd.DataFrame([{
            'qi': qi, 'date': pd.Timestamp(date),
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'r5': r5, 'r6': r6,
            'blue': blue
        }])
        model.df_history = pd.concat([model.df_history, new_row], ignore_index=True)
        print(f"Appended draw {qi}: {date}")
    else:
        print(f"Draw {qi} already in data")

model.df_history = model.df_history.sort_values('date').reset_index(drop=True)
print(f"Total draws for training: {len(model.df_history)}")
print(f"Date range: {model.df_history['date'].iloc[0].date()} ~ {model.df_history['date'].iloc[-1].date()}")
print(f"Last draw: {model.df_history['qi'].iloc[-1]} ({model.df_history['date'].iloc[-1].date()})")
print(f"Last blue: {model.df_history['blue'].iloc[-1]}")

# ============ 1. Train v3.2 ============
print("\n" + "=" * 60)
print("Training v3.2 model...")
model.train()

# ============ 2. Feature Importance ============
print("\n【v3.2 特征重要性 Top-15】")
for feat, imp in model.get_feature_importance(15):
    bar = '█' * int(imp * 100)
    print(f"  {feat:25s}: {imp:.4f} {bar}")

# Frequency features distribution
imp = model.model_red.feature_importances_
freq_feats = ['freq_last3', 'freq_last5', 'freq_last10', 'days_cold']
freq_total = 0
for f in freq_feats:
    fi = imp[model.feature_cols_rank.index(f)]
    freq_total += fi
    print(f"  {f}: {fi:.4f}")
print(f"  → 4频率特征合计: {freq_total:.4f}")

# ============ 3. Backtest ============
print("\n【滚动回测 (start=120, 最近95期)】")
model.backtest(start_period=120)
bt_summary = model.backtest_summary()

recent_bt = [r['hit_red'] for r in model.backtest_results[-20:]]
print(f"  最近20期红球均值: {np.mean(recent_bt):.3f}")
print(f"  最近20期 >=3: {sum(1 for h in recent_bt if h>=3)}/20")

# Blue backtest detail
blue_hits = [r['hit_blue'] for r in model.backtest_results]
blue_hits_recent = blue_hits[-30:]
print(f"  最近30期蓝球命中: {sum(blue_hits_recent)}/30")

# ============ 4. Predict for 2026-06-30 21:08 ============
print("\n" + "=" * 60)
print("预测: 2026-06-30 21:08 (第2026074期)")
print("=" * 60)

target = datetime(2026, 6, 30)
results, base = model.predict_multi(target, hour=21)

# ============ 5. Bazi analysis ============
feat, bazi, wx = extract_features_enhanced(target, hour=21)

print("\n=== 八字分析 ===")
print(f"年柱: {TIAN_GAN[bazi['nianzhu_gan']]}{DI_ZHI[bazi['nianzhu_zhi']]}  ({TG_WX[bazi['nianzhu_gan']]}/{DZ_WX[bazi['nianzhu_zhi']]})")
print(f"月柱: {TIAN_GAN[bazi['yuezhu_gan']]}{DI_ZHI[bazi['yuezhu_zhi']]}  ({TG_WX[bazi['yuezhu_gan']]}/{DZ_WX[bazi['yuezhu_zhi']]})")
print(f"日柱: {TIAN_GAN[bazi['rizhu_gan']]}{DI_ZHI[bazi['rizhu_zhi']]}  ({TG_WX[bazi['rizhu_gan']]}/{DZ_WX[bazi['rizhu_zhi']]}) — 日主{TG_WX[bazi['rizhu_gan']]}")
print(f"时柱: {TIAN_GAN[bazi['shizhu_gan']]}{DI_ZHI[bazi['shizhu_zhi']]}  ({TG_WX[bazi['shizhu_gan']]}/{DZ_WX[bazi['shizhu_zhi']]})")

# 五鼠遁修正
wushu_map = {0:0, 5:0, 1:2, 6:2, 2:4, 7:4, 3:6, 8:6, 4:8, 9:8}
day_gan_idx = bazi['rizhu_gan']
hour_zhi_idx = bazi['shizhu_zhi']
correct_g_hour = (wushu_map[day_gan_idx] + hour_zhi_idx) % 10
print(f"五鼠遁修正时柱: {TIAN_GAN[correct_g_hour]}{DI_ZHI[hour_zhi_idx]}")

# 纳音
print("\n纳音:")
nayin_names = ['木','火','土','金','水']
for label, g, z in [('年柱', bazi['nianzhu_gan'], bazi['nianzhu_zhi']),
                     ('月柱', bazi['yuezhu_gan'], bazi['yuezhu_zhi']),
                     ('日柱', bazi['rizhu_gan'], bazi['rizhu_zhi']),
                     ('时柱', correct_g_hour, hour_zhi_idx)]:
    nayin_idx = get_nayin_wuxing(g, z)
    print(f"  {TIAN_GAN[g]}{DI_ZHI[z]}: 纳音五行={nayin_names[nayin_idx]}")

# 十神
print("\n十神 (日主={}):".format(TG_WX[bazi['rizhu_gan']]))
ss_names = ['比肩','劫财','食神','伤官','偏财','正财','七杀','正官','偏印','正印']
for label, g_val in [('年干', bazi['nianzhu_gan']),
                      ('月干', bazi['yuezhu_gan']),
                      ('时干', correct_g_hour)]:
    ss_idx = get_shishen(day_gan_idx, g_val)
    print(f"  {label} {TIAN_GAN[g_val]} → {ss_names[ss_idx]}")

# 五行分布
print("\n=== 五行分布 ===")
print("主气:", wx)
cg_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
    for gan in DZ_CANG_GAN[bazi[k]]:
        cg_wx[TG_WX[gan]] += 1
print("藏干:", cg_wx)

# ============ 6. Top-12 Red Scores ============
print("\n=== 红球排序 Top-12 得分 (v3.2) ===")
for n, s in base['red_scores']:
    bar = '█' * int(s * 200)
    print(f"  {n:2d}: {s:.4f} {bar}")

# Raw scores (before trend weight)
print("\n=== 红球原始分数 (去趋势权重) ===")
raw = sorted(base['red_raw_scores'].items(), key=lambda x: x[1], reverse=True)[:12]
for n, s in raw:
    print(f"  {n:2d}: {s:.4f} {'█' * int(s * 200)}")

# ============ 7. Blue analysis ============
print("\n=== 蓝球分析 (v3.2) ===")
print(f"  CLF argmax (分类器直接选择): {base['blue_clf_argmax']}")
print(f"  CLF expected (分类器期望值): {base['blue_clf_expected']:.2f}")
print(f"  AR Dual 预测: {base['blue_ar']:.2f}")
if base.get('ar_mu_short') is not None:
    print(f"    mu_long={base['ar_mu_long']:.2f}, mu_short={base['ar_mu_short']:.2f}, w_short={base['ar_w_short']:.2f}")
print(f"  Ensemble (0.5*exp + 0.5*AR): {base['pred_blue']}")
print(f"  Top-5 概率分布: {base['blue_proba_top5']}")

# ============ 8. Six Groups (v3.2) ============
print("\n=== 六组预测 (v3.2 差异化策略) ===")
for r in results:
    reds_str = '、'.join(f'{n:02d}' for n in r['pred_red'])
    print(f"  组{r['index']} [{r['strategy']}]: 红球 {reds_str} | 蓝球 {r['pred_blue']:02d}")

# ============ 9. Diversity check ============
print("\n=== 多样性检查 ===")
all_reds = [set(r['pred_red']) for r in results]
for i in range(len(results)):
    overlap_with_1 = len(all_reds[i] & all_reds[0])
    others = ', '.join(f'组{j+1}:{len(all_reds[i] & all_reds[j])}' for j in range(len(results)) if j != i)
    print(f"  组{i+1} vs others: {others}")

# ============ 10. Trend weights ============
if hasattr(model, '_trend_weights'):
    hot = [n for n, w in model._trend_weights.items() if w >= 1.15]
    warm = [n for n, w in model._trend_weights.items() if 1.05 <= w < 1.15]
    cool = [n for n, w in model._trend_weights.items() if w <= 0.90]
    if hot:
        print(f"\n趋势HOT (w>=1.15): {hot}")
    if warm:
        print(f"趋势warm (w=1.05-1.15): {warm}")
    if cool:
        print(f"趋势cool (w<=0.90): {cool}")

# ============ 11. Zone distribution ============
print("\n=== 各预测组分区分布 ===")
for r in results:
    z1 = [n for n in r['pred_red'] if n in ZONE1]
    z2 = [n for n in r['pred_red'] if n in ZONE2]
    z3 = [n for n in r['pred_red'] if n in ZONE3]
    print(f"  组{r['index']}: 一区{z1} 二区{z2} 三区{z3}")

# ============ 12. Summary ============
print("\n" + "=" * 60)
print("模型状态: v3.2")
print(f"训练样本: {len(model.df_history)}期 | 特征: 46维 (33基础 + 4号码 + 5交互 + 4频率)")
if model.model_blue_ar:
    if hasattr(model.model_blue_ar, 'phi_long'):
        print(f"蓝球 AR Dual: phi_long={model.model_blue_ar.phi_long:.4f}, mu_long={model.model_blue_ar.mu_long:.2f}, mu_short={model.model_blue_ar.mu_short:.2f}")
    else:
        print(f"蓝球 AR phi: {model.model_blue_ar.phi:.4f}")

print(f"\n回测 (n={bt_summary['n']}):")
print(f"  红球均值: {bt_summary['mean_red']:.3f} (v3.1: 1.386)")
print(f"  红球>=3: {bt_summary['red_ge3_pct']:.1f}% (v3.1: 10.5%)")
print(f"  蓝球命中: {bt_summary['mean_blue']*100:.1f}% (v3.1: 7.8%)")

print("\n⚠️ 纯娱乐预测，请理性购彩！")
