# -*- coding: utf-8 -*-
"""预测 2026-06-25 第2026072期 双色球"""
import sys
sys.path.insert(0, r'C:\Users\landy\ai\.claude\skills\luckyball')

import numpy as np
import pandas as pd
from datetime import datetime
from lottery_model import (
    LotteryModelV3, extract_features_enhanced,
    TIAN_GAN, DI_ZHI, TG_WX, DZ_WX, TG_YY,
    WX_ORDER, DZ_CANG_GAN, NAYIN_MAP, get_shishen,
    get_nayin_wuxing, number_features_v31,
    ZONE1, ZONE2, ZONE3
)

# ============ 0. Load & extend history with latest draws ============
model = LotteryModelV3()
model.load_history(r'C:\Users\landy\ai\luspace\双色球200期号码.xlsx')
print(f"Loaded {len(model.df_history)} draws from file")

# Append latest 3 draws if not already in the data
latest_draws = [
    # qi, date, r1,r2,r3,r4,r5,r6, blue
    ('2026069', '2026-06-18', 12, 14, 16, 17, 18, 32, 8),
    ('2026070', '2026-06-21', 3, 6, 8, 14, 26, 27, 8),
    ('2026071', '2026-06-23', 3, 8, 19, 25, 31, 33, 5),
]

existing_qis = set(model.df_history['qi'].astype(str).values)
for qi, date, r1, r2, r3, r4, r5, r6, blue in latest_draws:
    if str(qi) not in existing_qis:
        new_row = pd.DataFrame([{
            'qi': qi, 'date': date,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'r5': r5, 'r6': r6,
            'blue': blue
        }])
        model.df_history = pd.concat([model.df_history, new_row], ignore_index=True)
        print(f"Appended draw {qi}: {date}")
    else:
        print(f"Draw {qi} already in data")

model.df_history = model.df_history.reset_index(drop=True)
print(f"Total draws for training: {len(model.df_history)}")
print(f"Last draw: {model.df_history['qi'].iloc[-1]} ({model.df_history['date'].iloc[-1]})")
print(f"Last blue: {model.df_history['blue'].iloc[-1]}")

# ============ 1. Train ============
print("\n" + "=" * 60)
print("Training v3.1 model...")
model.train()

# ============ 2. Feature Importance ============
print("\n【特征重要性 Top-15】")
for feat, imp in model.get_feature_importance(15):
    print(f"  {feat}: {imp:.4f}")

# ============ 3. Backtest ============
print("\n【滚动回测】")
model.backtest(start_period=60)
bt_summary = model.backtest_summary()

# ============ 4. Predict for 2026-06-25 19:58 ============
print("\n" + "=" * 60)
print("预测: 2026-06-25 19:58 (第2026072期)")
print("=" * 60)

target = datetime(2026, 6, 25)
results, base = model.predict_multi(target, hour=19)

# ============ 5. Bazi analysis ============
feat, bazi, wx = extract_features_enhanced(target, hour=19)

print("\n=== 八字分析 ===")
print(f"年柱: {TIAN_GAN[bazi['nianzhu_gan']]}{DI_ZHI[bazi['nianzhu_zhi']]} "
      f"({TIAN_GAN[bazi['nianzhu_gan']]}{DI_ZHI[bazi['nianzhu_zhi']]} 纳音)")
print(f"月柱: {TIAN_GAN[bazi['yuezhu_gan']]}{DI_ZHI[bazi['yuezhu_zhi']]}")
print(f"日柱: {TIAN_GAN[bazi['rizhu_gan']]}{DI_ZHI[bazi['rizhu_zhi']]} — 日主{TG_WX[bazi['rizhu_gan']]}")
print(f"时柱: {TIAN_GAN[bazi['shizhu_gan']]}{DI_ZHI[bazi['shizhu_zhi']]}")

# 五鼠遁修正
wushu_map = {0:0, 5:0, 1:2, 6:2, 2:4, 7:4, 3:6, 8:6, 4:8, 9:8}
day_gan_idx = bazi['rizhu_gan']
hour_zhi_idx = bazi['shizhu_zhi']
correct_g_hour = (wushu_map[day_gan_idx] + hour_zhi_idx) % 10
print(f"五鼠遁修正时柱: {TIAN_GAN[correct_g_hour]}{DI_ZHI[hour_zhi_idx]}")

# 纳音
print("\n纳音:")
for label, g, z in [('年柱', bazi['nianzhu_gan'], bazi['nianzhu_zhi']),
                     ('月柱', bazi['yuezhu_gan'], bazi['yuezhu_zhi']),
                     ('日柱', bazi['rizhu_gan'], bazi['rizhu_zhi']),
                     ('时柱', correct_g_hour, hour_zhi_idx)]:
    nayin_idx = get_nayin_wuxing(g, z)
    nayin_names = ['木','火','土','金','水']
    print(f"  {TIAN_GAN[g]}{DI_ZHI[z]}: 纳音五行={nayin_names[nayin_idx]}")

# 十神
print("\n十神 (日主={}):".format(TG_WX[bazi['rizhu_gan']]))
for label, g_val in [('年干', bazi['nianzhu_gan']),
                      ('月干', bazi['yuezhu_gan']),
                      ('时干', correct_g_hour)]:
    ss_names = ['比肩','劫财','食神','伤官','偏财','正财','七杀','正官','偏印','正印']
    ss_idx = get_shishen(day_gan_idx, g_val)
    print(f"  {label} {TIAN_GAN[g_val]} → {ss_names[ss_idx]}")

# 五行分布
print("\n=== 五行分布 ===")
print("主气:", wx)
# 藏干
cg_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
    for gan in DZ_CANG_GAN[bazi[k]]:
        cg_wx[TG_WX[gan]] += 1
print("藏干:", cg_wx)

# ============ 6. Top-12 Red Scores ============
print("\n=== 红球排序 Top-12 得分 ===")
for n, s in base['red_scores']:
    bar = '█' * int(s * 200)
    print(f"  {n:2d}: {s:.4f} {bar}")

# ============ 7. Blue analysis ============
print("\n=== 蓝球分析 ===")
print(f"  CLF argmax (分类器直接选择): {base['blue_clf_argmax']}")
print(f"  CLF expected (分类器期望值): {base['blue_clf_expected']:.2f}")
print(f"  AR(1) 预测: {base['blue_ar']:.2f}")
print(f"  Ensemble (0.5*exp + 0.5*AR): {base['pred_blue']}")
print(f"  Top-5 概率分布: {base['blue_proba_top5']}")

# ============ 8. Five Groups ============
print("\n=== 五组预测 (v3.1 模板化) ===")
for r in results:
    reds_str = '、'.join(f'{n:02d}' for n in r['pred_red'])
    print(f"  组{r['index']} [{r['strategy']}]: 红球 {reds_str} | 蓝球 {r['pred_blue']:02d}")

# ============ 9. Debias info ============
if hasattr(model, '_debias_weights'):
    cooled = [n for n, w in model._debias_weights.items() if w < 0.95]
    warmed = [n for n, w in model._debias_weights.items() if w > 1.05]
    if cooled:
        print(f"\n降温号码: {cooled}")
    if warmed:
        print(f"加温号码: {warmed}")

# ============ 10. Summary ============
print("\n" + "=" * 60)
print("模型状态: v3.1")
print(f"训练样本: {len(model.df_history)}期")
print(f"特征: 43维 (33基础 + 4号码 + 5交互 + 1上期)")
if model.model_blue_ar:
    print(f"蓝球 AR φ: {model.model_blue_ar.phi:.4f}")
    print(f"蓝球 AR μ: {model.model_blue_ar.mu:.2f}")

print("\n⚠️ 纯娱乐预测，请理性购彩！")
