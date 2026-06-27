# -*- coding: utf-8 -*-
"""
双色球预测模型 v3.2
v3.0 基础:
  - 蓝球 AR(1) 均值回归模型 — 利用历史自相关 r≈-0.72
  - 红球 pointwise 排序 — 单分类器 + 号码特征, 对 33 个球打分取 Top-6
  - 增强特征: 十神、地支藏干、纳音、节气距离
  - 滚动回测框架 — walk-forward 评估真实命中率

v3.1 改进 (2026-06-18):
  1. 蓝球分类器 — RandomForestClassifier(16类) 替代回归
  2. 红球区域平衡约束 — 强制三区各至少选1个
  3. 五行生克交互特征 — 号码五行与日主五行的生克关系 (5维)
  4. 上期重号特征 — is_prev_red 标记上期红球
  5. 五组模板化多样性 — 基准/追冷/追热/分散/随机
  6. 过度预测号码降温 — 基于回测偏差施加指数衰减权重

v3.2 改进 (2026-06-26):
  1. 多周期频率特征 — is_prev_red 拆分为 freq3/freq5/freq10/days_cold (4维)
     → 打破单特征独大 (原重要性 0.4568), 释放 38 维被压抑信号
  2. 趋势权重替代降温 — 基于实际出现频率, 热号加温, 冷号微调
     → 修正方向性错误 (08 出现 4/5 却被降温)
  3. 蓝球双窗口 AR(1) — 短期(10期) + 长期(全部) 动态加权
     → 捕捉短期趋势转变 (近期均值 5.2 vs 长期 9.34)
  4. 自适应区域约束 — 当某区候选分过低时自动放宽
     → 避免在空区浪费选号
  5. 连号奖励 — 对高分号码的相邻号码加分
     → 利用 60% 连号概率
  6. 真多样性 — 五组使用差异化特征权重
     → 追热组提升频率特征权重, 追冷组抑制频率特征权重
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 基础常量
# =============================================================================

TIAN_GAN  = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
DI_ZHI    = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
TG_WX     = ['木', '木', '火', '火', '土', '土', '金', '金', '水', '水']  # 天干五行
TG_YY     = ['阳','阴','阳','阴','阳','阴','阳','阴','阳','阴']          # 天干阴阳
DZ_WX     = ['水','土','木','木','土','火','火','土','金','金','土','水'] # 地支五行
DZ_YY     = ['阳','阴','阳','阴','阳','阴','阳','阴','阳','阴','阳','阴']
WX_ORDER  = {'木':0, '火':1, '土':2, '金':3, '水':4}

# ---------------------------------------------------------------------------
# 地支藏干 (index 是 天干 0-9)
# ---------------------------------------------------------------------------
DZ_CANG_GAN = {
    0:  [9],           # 子: 癸
    1:  [5, 9, 7],     # 丑: 己 癸 辛
    2:  [0, 2, 4],     # 寅: 甲 丙 戊
    3:  [1],           # 卯: 乙
    4:  [4, 1, 9],     # 辰: 戊 乙 癸
    5:  [2, 6, 4],     # 巳: 丙 庚 戊
    6:  [3, 5],        # 午: 丁 己
    7:  [5, 3, 1],     # 未: 己 丁 乙
    8:  [6, 8, 4],     # 申: 庚 壬 戊
    9:  [7],           # 酉: 辛
    10: [4, 7, 3],     # 戌: 戊 辛 丁
    11: [8, 0],        # 亥: 壬 甲
}

# ---------------------------------------------------------------------------
# 纳音五行 (60 甲子 → 纳音五行 index: 0木1火2土3金4水)
# ---------------------------------------------------------------------------
_NAYIN_30 = [3,1,0,2, 3,1,4,2, 3,0,4,2, 1,0,4, 3,1,0,2, 3,1,4,2, 3,0,4,2, 1,0,4]

def _make_nayin_map():
    """生成 (gan, zhi) → na_yin_wuxing_idx 映射"""
    m = {}
    for i in range(60):
        gan = i % 10
        zhi = i % 12
        m[(gan, zhi)] = _NAYIN_30[i // 2]
    return m

NAYIN_MAP = _make_nayin_map()

# ---------------------------------------------------------------------------
# 节气 (day-of-year, 近似)
# ---------------------------------------------------------------------------
SOLAR_TERMS_DOY = [
    35, 50, 65, 80, 95, 110, 125, 141, 157, 172, 188, 204,
    219, 235, 251, 266, 281, 296, 311, 326, 341, 356, 370, 385
]

# =============================================================================
# 号码区域定义
# =============================================================================
ZONE1 = list(range(1, 12))   # 一区 1-11
ZONE2 = list(range(12, 23))  # 二区 12-22
ZONE3 = list(range(23, 34))  # 三区 23-33

# =============================================================================
# 八字 & 特征提取
# =============================================================================

def calc_jdn(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365*y + y//4 - y//100 + y//400 - 32045


def get_bazi(date_val, hour=21):
    """计算八字, 返回 0-indexed gan/zhi"""
    if hasattr(date_val, 'year'):
        year, month, day = date_val.year, date_val.month, date_val.day
    else:
        dt = datetime.strptime(str(date_val), '%Y-%m-%d')
        year, month, day = dt.year, dt.month, dt.day

    year_gan = (year - 3) % 10 or 10
    year_zhi = (year - 3) % 12 or 12
    month_gan = (year % 10 + month * 2) % 10 or 10
    month_zhi = (month + 2) % 12 or 12
    jdn = calc_jdn(year, month, day)
    day_gan = (jdn + 6) % 10 or 10
    day_zhi = (jdn + 6) % 12 or 12
    hour_zhi = (hour // 2 + 1) % 12 or 12
    hour_gan = (day_gan * 2 + hour_zhi - 1) % 10 or 10

    return {
        'nianzhu_gan': year_gan - 1,  'nianzhu_zhi': year_zhi - 1,
        'yuezhu_gan':  month_gan - 1, 'yuezhu_zhi':  month_zhi - 1,
        'rizhu_gan':   day_gan - 1,   'rizhu_zhi':   day_zhi - 1,
        'shizhu_gan':  hour_gan - 1,  'shizhu_zhi':  hour_zhi - 1,
    }


def calc_wuxing(bazi):
    """天干+地支主气五行计数"""
    wx = {'木':0, '火':0, '土':0, '金':0, '水':0}
    for k in ['nianzhu_gan','yuezhu_gan','rizhu_gan','shizhu_gan']:
        wx[TG_WX[bazi[k]]] += 1
    for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
        wx[DZ_WX[bazi[k]]] += 1
    return wx


def calc_canggan_wuxing(bazi):
    """地支藏干五行计数 (更深层的五行信息)"""
    wx = {'木':0, '火':0, '土':0, '金':0, '水':0}
    for k in ['nianzhu_zhi','yuezhu_zhi','rizhu_zhi','shizhu_zhi']:
        for gan in DZ_CANG_GAN[bazi[k]]:
            wx[TG_WX[gan]] += 1
    return wx


def get_shishen(day_gan_idx, other_gan_idx):
    """计算十神关系: 0比肩 1劫财 2食神 3伤官 4偏财 5正财 6七杀 7正官 8偏印 9正印"""
    de = TG_WX[day_gan_idx]
    oe = TG_WX[other_gan_idx]
    same_yy = TG_YY[day_gan_idx] == TG_YY[other_gan_idx]

    di = WX_ORDER[de]
    oi = WX_ORDER[oe]

    if di == oi:            # 同我
        return 0 if same_yy else 1        # 比肩 / 劫财
    if (di + 1) % 5 == oi:  # 我生
        return 2 if same_yy else 3        # 食神 / 伤官
    if (di + 2) % 5 == oi:  # 我克
        return 4 if same_yy else 5        # 偏财 / 正财
    if (di + 3) % 5 == oi:  # 克我
        return 6 if same_yy else 7        # 七杀 / 正官
    # 生我
    return 8 if same_yy else 9            # 偏印 / 正印


def get_nayin_wuxing(gan_idx, zhi_idx):
    """获取某个干支组合的纳音五行 index (0-4)"""
    return NAYIN_MAP.get((gan_idx, zhi_idx), 2)


def solar_term_distance(doy):
    """距离最近节气的天数 (正=已过, 负=未到), 归一化到 [-15, 15]"""
    best = 999
    for term_doy in SOLAR_TERMS_DOY:
        d = doy - term_doy
        if abs(d) < abs(best):
            best = d
        for offset in [-365, 365]:
            d2 = doy - (term_doy + offset)
            if abs(d2) < abs(best):
                best = d2
    return best


def extract_features_enhanced(date_val, hour=21):
    """提取增强版特征 (v3.0)"""
    bazi = get_bazi(date_val, hour)
    wx = calc_wuxing(bazi)
    cg_wx = calc_canggan_wuxing(bazi)

    if hasattr(date_val, 'year'):
        dt = date_val
    else:
        dt = datetime.strptime(str(date_val), '%Y-%m-%d')

    doy = dt.timetuple().tm_yday
    day_gan = bazi['rizhu_gan']

    feat = {
        # --- 日期 ---
        'year': dt.year, 'month': dt.month, 'day': dt.day,
        'weekday': dt.weekday() if hasattr(dt, 'weekday') else pd.Timestamp(dt).dayofweek,
        'day_of_year': doy,
        'solar_term_dist': solar_term_distance(doy),

        # --- 天干 ---
        'nianzhu_gan': bazi['nianzhu_gan'], 'yuezhu_gan': bazi['yuezhu_gan'],
        'rizhu_gan': day_gan, 'shizhu_gan': bazi['shizhu_gan'],

        # --- 地支 ---
        'nianzhu_zhi': bazi['nianzhu_zhi'], 'yuezhu_zhi': bazi['yuezhu_zhi'],
        'rizhu_zhi': bazi['rizhu_zhi'], 'shizhu_zhi': bazi['shizhu_zhi'],

        # --- 五行主气 ---
        'mu': wx['木'], 'huo': wx['火'], 'tu': wx['土'], 'jin': wx['金'], 'shui': wx['水'],

        # --- 五行藏干 ---
        'cg_mu': cg_wx['木'], 'cg_huo': cg_wx['火'], 'cg_tu': cg_wx['土'],
        'cg_jin': cg_wx['金'], 'cg_shui': cg_wx['水'],

        # --- 纳音 (四柱) ---
        'nayin_nian': get_nayin_wuxing(bazi['nianzhu_gan'], bazi['nianzhu_zhi']),
        'nayin_yue':  get_nayin_wuxing(bazi['yuezhu_gan'],  bazi['yuezhu_zhi']),
        'nayin_ri':   get_nayin_wuxing(bazi['rizhu_gan'],   bazi['rizhu_zhi']),
        'nayin_shi':  get_nayin_wuxing(bazi['shizhu_gan'],  bazi['shizhu_zhi']),

        # --- 十神 (年/月/时干 对 日干的关系) ---
        'shishen_nian': get_shishen(day_gan, bazi['nianzhu_gan']),
        'shishen_yue':  get_shishen(day_gan, bazi['yuezhu_gan']),
        'shishen_shi':  get_shishen(day_gan, bazi['shizhu_gan']),

        # --- 阴阳标记 ---
        'nianzhu_yy': 0 if TG_YY[bazi['nianzhu_gan']] == '阳' else 1,
        'rizhu_yy':   0 if TG_YY[bazi['rizhu_gan']]   == '阳' else 1,
    }
    return feat, bazi, wx


# =============================================================================
# v3.1 新增: 五行生克交互特征
# =============================================================================

def interaction_features(num_wx_idx, rizhu_wx_idx):
    """
    号码五行与日主五行的生克关系 → 5维 one-hot
    0=比和(同), 1=我生, 2=我克, 3=克我, 4=生我
    """
    diff = (num_wx_idx - rizhu_wx_idx + 5) % 5
    feats = [0] * 5
    feats[diff] = 1
    return feats  # [interact_same, interact_isheng, interact_woke, interact_kewo, interact_shengwo]


# =============================================================================
# 号码特征 (用于 pointwise 排序)
# =============================================================================

def _number_tiangan(n):
    """号码 1-33 的天干序号 (0-9)"""
    return (n - 1) % 10


def _number_dizhi(n):
    """号码 1-33 的地支序号 (0-11)"""
    return (n - 1) % 12


def _number_wuxing(n):
    """号码的天干五行 index"""
    return WX_ORDER[TG_WX[_number_tiangan(n)]]


def number_features(n):
    """返回单个号码的属性向量"""
    return [
        _number_tiangan(n),         # 天干 0-9
        _number_dizhi(n),           # 地支 0-11
        _number_wuxing(n),          # 五行 0-4
        n % 2,                      # 奇偶
    ]


def number_features_v31(n, rizhu_wx_idx, prev_reds=None):
    """
    v3.1 号码特征: 基础4维 + 交互5维 + 上期标记1维 = 10维
    （保留用于回测兼容，v3.2 使用 number_features_v32）
    """
    base = number_features(n)
    interact = interaction_features(_number_wuxing(n), rizhu_wx_idx)
    is_prev = 1 if (prev_reds is not None and n in prev_reds) else 0
    return base + interact + [is_prev]


# =============================================================================
# v3.2: 多周期频率特征 (替代 is_prev_red)
# =============================================================================

def _compute_multi_period_freqs(df, current_idx):
    """
    为当期所有号码计算多周期频率特征。
    返回 dict: {n: [freq_last3, freq_last5, freq_last10, days_cold_norm]}
    频率: 0.0~1.0, days_cold_norm: 0~1 (0=刚出, 1=长期未出)
    """
    freqs = {}
    for n in range(1, 34):
        # 回溯统计
        last3 = last5 = last10 = 0
        days_since = 999
        for j in range(current_idx - 1, -1, -1):
            row = df.iloc[j]
            drawn = set([int(row['r1']), int(row['r2']), int(row['r3']),
                        int(row['r4']), int(row['r5']), int(row['r6'])])
            dist_from_current = current_idx - j
            if n in drawn:
                if dist_from_current <= 3:
                    last3 += 1
                if dist_from_current <= 5:
                    last5 += 1
                if dist_from_current <= 10:
                    last10 += 1
                days_since = min(days_since, dist_from_current)

        freqs[n] = [
            last3 / 3.0,                                    # freq_last3
            last5 / 5.0,                                    # freq_last5
            last10 / 10.0,                                  # freq_last10
            min(days_since, 50) / 50.0 if days_since < 999 else 1.0,  # days_cold_norm
        ]
    return freqs


def number_features_v32(n, rizhu_wx_idx, freq_features=None):
    """
    v3.2 号码特征: 基础4维 + 交互5维 + 多周期频率4维 = 13维

    freq_features: dict {n: [freq3, freq5, freq10, days_cold]} 或 None
    """
    base = number_features(n)
    interact = interaction_features(_number_wuxing(n), rizhu_wx_idx)
    if freq_features is not None and n in freq_features:
        freq = freq_features[n]
    else:
        freq = [0.0, 0.0, 0.0, 0.5]  # 默认值
    return base + interact + freq


# =============================================================================
# 蓝球 AR(1) 均值回归模型
# =============================================================================

class BlueBallAR:
    """蓝球 AR(1) 模型: blue_t = mu + phi * (blue_{t-1} - mu)"""

    def __init__(self):
        self.mu = None
        self.phi = None
        self.sigma = None

    def fit(self, blue_series):
        y = np.array(blue_series, dtype=float)
        self.mu = np.mean(y)
        y_lag = y[:-1]
        y_cur = y[1:]
        dy_lag = y_lag - self.mu
        dy_cur = y_cur - self.mu
        if np.var(dy_lag) > 1e-6:
            self.phi = np.dot(dy_cur, dy_lag) / np.dot(dy_lag, dy_lag)
        else:
            self.phi = 0.0
        self.sigma = np.std(dy_cur - self.phi * dy_lag)
        return self

    def predict(self, last_blue):
        if self.mu is None:
            return 8.5
        pred = self.mu + self.phi * (last_blue - self.mu)
        return np.clip(pred, 1, 16)

    def predict_discrete(self, last_blue):
        return int(round(self.predict(last_blue)))


# =============================================================================
# v3.2: 蓝球双窗口 AR(1) — 短期 + 长期动态加权
# =============================================================================

class BlueBallARDual:
    """蓝球双窗口 AR(1): 短期(10期) + 长期(全部) 动态加权"""

    def __init__(self, short_window=10):
        self.short_window = short_window
        self.mu_short = None
        self.mu_long = None
        self.phi_short = None
        self.phi_long = None
        self.sigma_long = None
        self.w_short = 0.3  # 短期窗口默认权重

    def fit(self, blue_series):
        y = np.array(blue_series, dtype=float)
        n = len(y)

        # 长期 AR
        self.mu_long = np.mean(y)
        y_lag = y[:-1]
        y_cur = y[1:]
        dy_lag = y_lag - self.mu_long
        dy_cur = y_cur - self.mu_long
        if np.var(dy_lag) > 1e-6:
            self.phi_long = np.dot(dy_cur, dy_lag) / np.dot(dy_lag, dy_lag)
        else:
            self.phi_long = 0.0
        residuals = dy_cur - self.phi_long * dy_lag
        self.sigma_long = np.std(residuals)

        # 短期 AR (最近 short_window 期)
        sw = min(self.short_window, n)
        y_short = y[-sw:]
        self.mu_short = np.mean(y_short)
        if sw >= 3:
            yl_s = y_short[:-1]
            yc_s = y_short[1:]
            dyl_s = yl_s - self.mu_short
            dyc_s = yc_s - self.mu_short
            if np.var(dyl_s) > 1e-6:
                self.phi_short = np.dot(dyc_s, dyl_s) / np.dot(dyl_s, dyl_s)
            else:
                self.phi_short = self.phi_long
        else:
            self.mu_short = self.mu_long
            self.phi_short = self.phi_long

        # 动态权重: 偏差越大, 短期权重越高
        if self.sigma_long and self.sigma_long > 0:
            deviation = abs(self.mu_short - self.mu_long) / max(self.sigma_long, 0.5)
            self.w_short = min(0.75, 0.25 + deviation * 0.18)
        else:
            self.w_short = 0.30

        return self

    def predict(self, last_blue):
        if self.mu_long is None:
            return 8.5
        ar_short = self.mu_short + self.phi_short * (last_blue - self.mu_short) if self.phi_short else self.mu_short
        ar_long = self.mu_long + self.phi_long * (last_blue - self.mu_long) if self.phi_long else self.mu_long
        pred = self.w_short * ar_short + (1 - self.w_short) * ar_long
        return np.clip(pred, 1, 16)

    def predict_discrete(self, last_blue):
        return int(round(self.predict(last_blue)))


# =============================================================================
# 主模型 v3.2
# =============================================================================

class LotteryModelV3:
    """双色球预测模型 v3.1"""

    def __init__(self):
        self.df_history = None
        # v3.0 基础特征
        self.feature_cols_base = [
            'year', 'month', 'day', 'weekday', 'day_of_year', 'solar_term_dist',
            'nianzhu_gan', 'nianzhu_zhi', 'yuezhu_gan', 'yuezhu_zhi',
            'rizhu_gan', 'rizhu_zhi', 'shizhu_gan', 'shizhu_zhi',
            'mu', 'huo', 'tu', 'jin', 'shui',
            'cg_mu', 'cg_huo', 'cg_tu', 'cg_jin', 'cg_shui',
            'nayin_nian', 'nayin_yue', 'nayin_ri', 'nayin_shi',
            'shishen_nian', 'shishen_yue', 'shishen_shi',
            'nianzhu_yy', 'rizhu_yy',
        ]
        # v3.0 排序特征 = 基础 + 号码4维
        # v3.1 排序特征 = 基础 + 号码4维 + 交互5维 + 上期标记1维
        # v3.2 排序特征 = 基础 + 号码4维 + 交互5维 + 多周期频率4维 (46维)
        self.feature_cols_rank = self.feature_cols_base + [
            'num_gan', 'num_zhi', 'num_wx', 'num_parity',
            'interact_same', 'interact_isheng', 'interact_woke',
            'interact_kewo', 'interact_shengwo',
            'freq_last3', 'freq_last5', 'freq_last10', 'days_cold',
        ]
        self.model_red = None       # 红球排序分类器
        self.model_blue_clf = None  # 蓝球分类器 (替代回归)
        self.model_blue_ar = None   # v3.2: 蓝球双窗口 AR(1)
        self.backtest_results = None

        # v3.2: 趋势权重 + 策略特征权重
        self._trend_weights = {i: 1.0 for i in range(1, 34)}

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_history(self, filepath):
        df = pd.read_excel(filepath)
        df.columns = ['qi', 'date', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'blue']
        self.df_history = df

    # ------------------------------------------------------------------
    # v3.2: 构建排序训练数据 (多周期频率特征替代 is_prev_red)
    # ------------------------------------------------------------------

    def _build_rank_data(self, df):
        """为每期历史的每个号码(1-33)构建一行, label=1 表示开出"""
        rows = []
        labels = []
        for i, (_, row) in enumerate(df.iterrows()):
            feat, _, _ = extract_features_enhanced(row['date'])
            ri_wx_idx = WX_ORDER[TG_WX[feat['rizhu_gan']]]

            # v3.2: 计算多周期频率特征
            freq_features = _compute_multi_period_freqs(df, i) if i > 0 else None

            drawn = set([int(row['r1']), int(row['r2']), int(row['r3']),
                         int(row['r4']), int(row['r5']), int(row['r6'])])
            for n in range(1, 34):
                nf = number_features_v32(n, ri_wx_idx, freq_features)
                x = [feat[c] for c in self.feature_cols_base] + nf
                rows.append(x)
                labels.append(1 if n in drawn else 0)
        return np.array(rows), np.array(labels)

    # ------------------------------------------------------------------
    # v3.1: 训练
    # ------------------------------------------------------------------

    def train(self):
        if self.df_history is None:
            raise ValueError('Please load history data first')

        df = self.df_history

        # ---- 红球排序模型 (v3.1: +交互特征 +上期特征) ----
        X_rank, y_rank = self._build_rank_data(df)
        self.model_red = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.model_red.fit(X_rank, y_rank)

        # ---- 蓝球分类器 (v3.1: 16类分类替代回归) ----
        records = []
        for _, row in df.iterrows():
            feat, _, _ = extract_features_enhanced(row['date'])
            feat['blue'] = int(row['blue'])
            records.append(feat)
        df_blue = pd.DataFrame(records)
        Xb = df_blue[self.feature_cols_base].values
        yb = df_blue['blue'].values - 1  # 0-15

        self.model_blue_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            class_weight='balanced', random_state=42
        )
        self.model_blue_clf.fit(Xb, yb)

        # ---- v3.2: 蓝球双窗口 AR(1) ----
        blue_series = df['blue'].values
        self.model_blue_ar = BlueBallARDual(short_window=10)
        self.model_blue_ar.fit(blue_series)

        # ---- v3.2: 计算趋势权重 (替代 v3.1 降温) ----
        self._compute_trend_weights()

        print(f'[v3.2] Trained on {len(df)} periods')
        print(f'  Red rank model: {X_rank.shape[0]} samples, features={X_rank.shape[1]}, '
              f'{y_rank.sum():.0f} positive')
        print(f'  Blue classifier: 16-class, samples={len(df_blue)}')
        if hasattr(self.model_blue_ar, 'mu_long'):
            print(f'  Blue AR dual: mu_long={self.model_blue_ar.mu_long:.2f}, '
                  f'mu_short={self.model_blue_ar.mu_short:.2f}, '
                  f'w_short={self.model_blue_ar.w_short:.2f}')
        else:
            print(f'  Blue AR(1): mu={self.model_blue_ar.mu:.2f}, phi={self.model_blue_ar.phi:.4f}')
        return self

    # ------------------------------------------------------------------
    # v3.2: 基于实际频率的趋势权重 (替代 v3.1 降温)
    # ------------------------------------------------------------------

    def _compute_trend_weights(self):
        """
        v3.2: 基于号码实际出现频率计算趋势权重。
        - 短期热门 (last5 >= 3): 加温 20%
        - 偏热 (last5 >= 2): 加温 10%
        - 长期冷号 (last20 == 0): 轻微加温 5% (均值回归)
        - 短期冷号 (last5 == 0, 但 last20 > 0): 降温 10%
        - 其他: 1.0
        """
        df = self.df_history
        n_total = len(df)
        if n_total == 0:
            self._trend_weights = {i: 1.0 for i in range(1, 34)}
            return

        weights = {}
        for n in range(1, 34):
            count5 = sum(1 for j in range(max(0, n_total - 5), n_total)
                        if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']),
                                     int(df.iloc[j]['r3']), int(df.iloc[j]['r4']),
                                     int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]))
            count20 = sum(1 for j in range(max(0, n_total - 20), n_total)
                         if n in set([int(df.iloc[j]['r1']), int(df.iloc[j]['r2']),
                                      int(df.iloc[j]['r3']), int(df.iloc[j]['r4']),
                                      int(df.iloc[j]['r5']), int(df.iloc[j]['r6'])]))

            if count5 >= 3:
                weights[n] = 1.20   # 短期大热: 加温 20%
            elif count5 >= 2:
                weights[n] = 1.10   # 偏热: 加温 10%
            elif count20 == 0:
                weights[n] = 1.05   # 长冷号: 轻微加温 (均值回归预期)
            elif count5 == 0:
                weights[n] = 0.90   # 短期冷号 (但不在长期冷门): 降温 10%
            else:
                weights[n] = 1.00

        self._trend_weights = weights

        # 诊断输出
        hot = [n for n, w in weights.items() if w >= 1.15]
        warm = [n for n, w in weights.items() if 1.05 <= w < 1.15]
        cool = [n for n, w in weights.items() if w <= 0.90]
        if hot:
            print(f'  Trend HOT (w>=1.15): {hot}')
        if warm:
            print(f'  Trend warm (w=1.05-1.15): {warm}')
        if cool:
            print(f'  Trend cool (w<=0.90): {cool}')

    # ------------------------------------------------------------------
    # 回测
    # ------------------------------------------------------------------

    def backtest(self, start_period=60):
        """Walk-forward 回测 v3.2: 从第 start_period 期开始"""
        df = self.df_history
        results = []

        for i in range(start_period, len(df)):
            train_df = df.iloc[:i]
            test_row = df.iloc[i]

            sub = LotteryModelV3()
            sub.df_history = train_df

            # 红球 (v3.2 多周期频率)
            Xr, yr = sub._build_rank_data(train_df)
            sub.model_red = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            sub.model_red.fit(Xr, yr)

            # 蓝球分类器
            tf_records = []
            for _, tr in train_df.iterrows():
                f, _, _ = extract_features_enhanced(tr['date'])
                f['blue'] = int(tr['blue'])
                tf_records.append(f)
            tf = pd.DataFrame(tf_records)
            sub.model_blue_clf = RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced', random_state=42
            )
            sub.model_blue_clf.fit(
                tf[sub.feature_cols_base].values,
                tf['blue'].values.astype(int) - 1
            )

            # v3.2: 蓝球双窗口 AR
            sub.model_blue_ar = BlueBallARDual(short_window=10)
            sub.model_blue_ar.fit(train_df['blue'].values)

            # v3.2: 趋势权重 (用于红球排序)
            sub._compute_trend_weights()

            # 预测 (使用 v3.2 predict, 自适应区域约束)
            pred = sub.predict(test_row['date'])

            # 比对
            actual_red = set([int(test_row['r1']), int(test_row['r2']), int(test_row['r3']),
                              int(test_row['r4']), int(test_row['r5']), int(test_row['r6'])])
            actual_blue = int(test_row['blue'])
            hit_red = len(set(pred['pred_red']) & actual_red)
            hit_blue = 1 if pred['pred_blue'] == actual_blue else 0

            results.append({
                'qi': test_row['qi'],
                'date': test_row['date'],
                'pred_red': pred['pred_red'],
                'pred_blue': pred['pred_blue'],
                'actual_red': sorted(actual_red),
                'actual_blue': actual_blue,
                'hit_red': hit_red,
                'hit_blue': hit_blue,
            })

        self.backtest_results = results
        return results

    def backtest_summary(self):
        """回测摘要统计"""
        if not self.backtest_results:
            print('No backtest results. Run .backtest() first.')
            return

        n = len(self.backtest_results)
        red_hits = [r['hit_red'] for r in self.backtest_results]
        blue_hits = [r['hit_blue'] for r in self.backtest_results]

        from math import comb
        expected_red = sum(k * comb(6,k)*comb(27,6-k)/comb(33,6) for k in range(7))
        expected_blue = 1/16

        print(f'===== 回测摘要 (n={n}) =====')
        print(f'  红球平均命中: {np.mean(red_hits):.3f}  (随机期望: {expected_red:.3f})')
        print(f'  红球 ≥3 命中率: {sum(1 for h in red_hits if h>=3)/n*100:.1f}%')
        print(f'  蓝球命中率:    {np.mean(blue_hits)*100:.1f}%  (随机期望: {expected_blue*100:.1f}%)')
        if hasattr(self.model_blue_ar, 'phi_long'):
            print(f'  蓝球 AR phi_long: {self.model_blue_ar.phi_long:.4f}')
            print(f'  蓝球 AR w_short:  {self.model_blue_ar.w_short:.2f}')
        else:
            print(f'  蓝球 AR phi:    {self.model_blue_ar.phi:.4f}')

        recent = red_hits[-20:]
        print(f'  最近20期红球均值: {np.mean(recent):.3f}')

        return {
            'n': n, 'mean_red': np.mean(red_hits), 'mean_blue': np.mean(blue_hits),
            'expected_red': expected_red, 'expected_blue': expected_blue,
            'red_ge3_pct': sum(1 for h in red_hits if h>=3)/n*100,
        }

    # ------------------------------------------------------------------
    # v3.1: 预测 — 区域平衡 + 降温权重 + 蓝球分类器
    # ------------------------------------------------------------------

    def _predict_red_scores(self, date_val, hour=21, freq_multiplier=1.0):
        """
        v3.2: 对 33 个红球打分 (多周期频率特征 + 趋势权重 + 连号奖励)

        freq_multiplier: 频率特征权重倍数 (>1 追热, <1 追冷, =1 基准)
        """
        feat, _, _ = extract_features_enhanced(date_val, hour)
        ri_wx_idx = WX_ORDER[TG_WX[feat['rizhu_gan']]]
        base = [feat[c] for c in self.feature_cols_base]

        # v3.2: 计算多周期频率特征
        n_total = len(self.df_history) if self.df_history is not None else 0
        freq_features = _compute_multi_period_freqs(self.df_history, n_total) if n_total > 0 else None

        scores = []
        for n in range(1, 34):
            nf = number_features_v32(n, ri_wx_idx, freq_features)
            x = np.array([base + nf])
            prob = self.model_red.predict_proba(x)[0][1]

            # v3.2: 应用趋势权重 (替代 v3.1 降温)
            prob *= getattr(self, '_trend_weights', {}).get(n, 1.0)

            # v3.2: 频率特征差异化 (用于多样性策略)
            if freq_multiplier != 1.0 and freq_features is not None:
                freq_bonus = sum(freq_features[n][:3]) / 3.0  # 平均频率
                prob *= (1.0 + (freq_multiplier - 1.0) * freq_bonus)

            scores.append((n, prob))

        # v3.2: 连号奖励 — 对 Top 候选的相邻号码加分
        scores_dict = dict(scores)
        sorted_scores = sorted(scores, key=lambda t: t[1], reverse=True)
        top_candidates = set(n for n, s in sorted_scores[:12])

        boosted = {}
        for n in top_candidates:
            for adj in [n - 1, n + 1]:
                if 1 <= adj <= 33 and adj not in top_candidates:
                    # 连号奖励: 相邻号码获得额外分 (但不超 Top-1)
                    adj_bonus = scores_dict.get(n, 0) * 0.15  # 15% of neighbor's score
                    boosted[adj] = max(boosted.get(adj, 0), adj_bonus)

        scores = [(n, s + boosted.get(n, 0)) for n, s in scores]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores, {n: s for n, s in scores_dict.items()}  # 返回原始分数用于参考

    def _balanced_top6(self, scores, adaptive=True):
        """
        v3.2: 自适应区域平衡 Top-6 选择
        确保三区各至少选 1 个号码, 但若某区最佳得分 < 全局第12名得分的 60%, 则跳过该区

        adaptive=False 时恢复 v3.1 的硬约束行为 (用于回测)
        """
        # 按区分组
        zone_scores = {1: [], 2: [], 3: []}
        for n, s in scores:
            if n in ZONE1:
                zone_scores[1].append((n, s))
            elif n in ZONE2:
                zone_scores[2].append((n, s))
            else:
                zone_scores[3].append((n, s))

        # 排名第12的得分作为阈值参考
        threshold_12th = scores[11][1] if len(scores) > 11 else 0
        zone_threshold = threshold_12th * 0.60  # 某区最佳必须 > 60% of 12th

        selected = set()
        # 每区先取 top-1 (若高于阈值)
        for z in [1, 2, 3]:
            if zone_scores[z] and adaptive:
                best_in_zone = zone_scores[z][0][1]
                if best_in_zone < zone_threshold:
                    # 该区候选分过低, 跳过硬约束
                    continue
            for n, s in zone_scores[z]:
                if n not in selected:
                    selected.add(n)
                    break

        # 剩余从全局最高分取 (排除已选)
        for n, s in scores:
            if len(selected) >= 6:
                break
            if n not in selected:
                selected.add(n)

        return sorted(selected)

    def predict(self, date_val, hour=21):
        """v3.2 单次预测: 多周期频率 + 自适应区域约束 + 双窗口 AR + 连号奖励"""
        feat, bazi, wx = extract_features_enhanced(date_val, hour)
        Xb = np.array([[feat[c] for c in self.feature_cols_base]])

        # 红球: 自适应区域平衡 Top-6 (v3.2)
        scores, raw_scores = self._predict_red_scores(date_val, hour)
        red_top6 = self._balanced_top6(scores, adaptive=True)

        # 蓝球: 分类器 + 双窗口 AR 加权 (v3.2)
        blue_proba_raw = self.model_blue_clf.predict_proba(Xb)[0]  # may have <16 classes
        blue_classes = self.model_blue_clf.classes_  # actual class labels (0-15)
        # Build full 16-dim probability vector
        blue_proba_full = np.zeros(16)
        for idx, cls in enumerate(blue_classes):
            if idx < len(blue_proba_raw):
                blue_proba_full[int(cls)] = blue_proba_raw[idx]
        blue_proba = blue_proba_full
        blue_clf_argmax = int(np.argmax(blue_proba)) + 1
        blue_clf_expected = sum((i + 1) * blue_proba[i] for i in range(16))

        # v3.2: 双窗口 AR(1)
        if len(self.df_history) > 0:
            last_blue = self.df_history['blue'].iloc[-1]
        else:
            last_blue = 8
        blue_ar_val = self.model_blue_ar.predict(last_blue)

        # 加权集成: 期望值(50%) + AR(50%)
        blue_ens_continuous = 0.5 * blue_clf_expected + 0.5 * blue_ar_val
        blue_ens = int(round(np.clip(blue_ens_continuous, 1, 16)))

        return {
            'date': date_val,
            'bazi': bazi,
            'wuxing': wx,
            'features': feat,
            'pred_red': red_top6,
            'pred_blue': blue_ens,
            'red_scores': scores[:12],
            'red_raw_scores': raw_scores,
            'blue_clf_argmax': blue_clf_argmax,
            'blue_clf_expected': float(blue_clf_expected),
            'blue_ar': float(blue_ar_val),
            'blue_proba_top5': [(int(i)+1, float(blue_proba[i]))
                                for i in np.argsort(blue_proba)[::-1][:5]],
            'ar_mu_long': getattr(self.model_blue_ar, 'mu_long', None),
            'ar_mu_short': getattr(self.model_blue_ar, 'mu_short', None),
            'ar_w_short': getattr(self.model_blue_ar, 'w_short', None),
        }

    # ------------------------------------------------------------------
    # v3.2: 六组差异化多样性预测 (真多样性)
    # ------------------------------------------------------------------

    def predict_multi(self, date_val, hour=21, n=6):
        """
        v3.2: 6组差异化预测 (真多样性 — 不同组使用不同特征权重策略)
        组1: 基准(自适应区域平衡) — freq_multiplier=1.0
        组2: 追冷号 — freq_multiplier=0.3 (抑制高频特征, 冷号获更高分)
        组3: 追热号 — freq_multiplier=3.0 (放大频率特征, 热号获更高分)
        组4: 均衡分散(2-2-2) — freq_multiplier=1.0 + 每区2个
        组5: 五行偏重 — 根据日主五行偏好调整权重
        组6: 加权随机采样 — freq_multiplier=2.0 从 Top-15 随机
        """
        # 组1: 基准
        base = self.predict(date_val, hour)
        scores = base['red_scores']
        full_scores, _ = self._predict_red_scores(date_val, hour, freq_multiplier=1.0)
        full_scores_dict = {n: s for n, s in full_scores}

        results = [{
            'date': date_val,
            'index': 1,
            'pred_red': base['pred_red'],
            'pred_blue': base['pred_blue'],
            'strategy': '基准(自适应区域平衡)',
        }]

        # --- 组2: 追冷号 (抑制频率, 冷号获更多分) ---
        cold_scores, _ = self._predict_red_scores(date_val, hour, freq_multiplier=0.3)
        cold_full = {n: s for n, s in cold_scores}

        # 选出冷号高分中的 Top-6 (自适应区域约束)
        cold_top6 = self._balanced_top6(cold_scores, adaptive=True)

        # 蓝球: CLF 第二选择
        blue_proba = base.get('blue_proba_top5', [])
        blue2 = blue_proba[1][0] if len(blue_proba) > 1 else base['pred_blue']

        results.append({
            'date': date_val,
            'index': 2,
            'pred_red': cold_top6,
            'pred_blue': blue2,
            'strategy': '追冷号(freq×0.3)',
        })

        # --- 组3: 追热号 (放大频率特征) ---
        hot_scores, _ = self._predict_red_scores(date_val, hour, freq_multiplier=3.0)
        hot_top6 = self._balanced_top6(hot_scores, adaptive=True)

        # 蓝球: CLF argmax
        blue3 = base['blue_clf_argmax']

        results.append({
            'date': date_val,
            'index': 3,
            'pred_red': hot_top6,
            'pred_blue': int(blue3),
            'strategy': '追热号(freq×3.0)',
        })

        # --- 组4: 均衡分散 (每区恰好2个) ---
        selected4 = set()
        for z in [1, 2, 3]:
            z_nums = ZONE1 if z == 1 else ZONE2 if z == 2 else ZONE3
            z_list = [(n, s) for n, s in full_scores if n in z_nums and n not in selected4]
            count = 0
            for n, s in z_list:
                if count >= 2:
                    break
                selected4.add(n)
                count += 1

        # 蓝球: AR 离散预测
        if len(self.df_history) > 0:
            last_blue = self.df_history['blue'].iloc[-1]
        else:
            last_blue = 8
        blue4 = self.model_blue_ar.predict_discrete(last_blue)

        results.append({
            'date': date_val,
            'index': 4,
            'pred_red': sorted(selected4),
            'pred_blue': blue4,
            'strategy': '均衡分散(2-2-2)',
        })

        # --- 组5: 五行偏重 — 根据日主五行调整 ---
        feat, _, _ = extract_features_enhanced(date_val, hour)
        rizhu_wx = TG_WX[feat['rizhu_gan']]  # 日主五行
        # 五行补益: 生我 + 我 → 加温
        wx_boost = {'木': ['水', '木'], '火': ['木', '火'], '土': ['火', '土'],
                    '金': ['土', '金'], '水': ['金', '水']}
        boost_wx = wx_boost.get(rizhu_wx, ['木', '火'])

        wx_scores, _ = self._predict_red_scores(date_val, hour, freq_multiplier=1.0)
        wx_adjusted = []
        for n, s in wx_scores:
            n_wx = TG_WX[_number_tiangan(n)]
            if n_wx in boost_wx:
                s *= 1.08  # 补益五行加温 8%
            wx_adjusted.append((n, s))
        wx_adjusted.sort(key=lambda t: t[1], reverse=True)
        wx_top6 = self._balanced_top6(wx_adjusted, adaptive=True)

        # 蓝球: 从概率分布中按期望取
        blue5 = int(round(base['blue_clf_expected']))

        results.append({
            'date': date_val,
            'index': 5,
            'pred_red': wx_top6,
            'pred_blue': blue5,
            'strategy': f'五行偏重(喜{"/".join(boost_wx)})',
        })

        # --- 组6: 加权随机采样 ---
        np.random.seed(int(date_val.strftime('%Y%m%d')) + 5)
        top15_nums = [n for n, _ in full_scores[:15]]
        top15_probs = np.array([s for _, s in full_scores[:15]])
        top15_probs = top15_probs / top15_probs.sum()
        sampled = set()
        # 分区约束的加权采样
        for z_nums, z_want in [(ZONE1, 2), (ZONE2, 2), (ZONE3, 2)]:
            z_candidates = [n for n in top15_nums if n in z_nums and n not in sampled]
            if z_candidates:
                z_probs = np.array([full_scores_dict[n] for n in z_candidates])
                z_probs = z_probs / z_probs.sum()
                picks = np.random.choice(z_candidates, size=min(z_want, len(z_candidates)),
                                        replace=False, p=z_probs)
                sampled.update(picks)
        # 不够 6 个从剩余补
        for n, _ in full_scores:
            if len(sampled) >= 6:
                break
            if n not in sampled:
                sampled.add(n)

        # 蓝球: 随机采样
        if blue_proba:
            b_nums = [b[0] for b in blue_proba[:5]]
            b_probs = np.array([b[1] for b in blue_proba[:5]])
            b_probs = b_probs / b_probs.sum()
            blue6 = int(np.random.choice(b_nums, p=b_probs))
        else:
            blue6 = base['pred_blue']

        results.append({
            'date': date_val,
            'index': 6,
            'pred_red': sorted(sampled),
            'pred_blue': blue6,
            'strategy': '加权随机采样',
        })

        return results, base

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_n=15):
        """红球排序模型的特征重要性"""
        if self.model_red is None:
            print('Model not trained.')
            return []
        imp = self.model_red.feature_importances_
        idx = np.argsort(imp)[::-1]
        return [(self.feature_cols_rank[i], imp[i]) for i in idx[:top_n]]

    @property
    def blue_ar_phi(self):
        if self.model_blue_ar:
            if hasattr(self.model_blue_ar, 'phi_long'):
                return self.model_blue_ar.phi_long
            return self.model_blue_ar.phi
        return None


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('双色球预测模型 v3.2')
    print('多周期频率 + 趋势权重 + 双窗口AR + 自适应区域 + 连号奖励')
    print('=' * 60)

    model = LotteryModelV3()
    model.load_history(r'C:\Users\a\luspace\双色球200期号码.xlsx')
    model.train()

    # 特征重要性
    print('\n【特征重要性 Top-15】')
    for feat, imp in model.get_feature_importance(15):
        print(f'  {feat}: {imp:.4f}')

    # 回测
    print('\n【滚动回测】')
    model.backtest(start_period=60)
    model.backtest_summary()

    # 预测
    print('\n' + '=' * 60)
    target = datetime(2026, 6, 4)
    result = model.predict(target, hour=21)
    print(f'预测日期: {target.strftime("%Y-%m-%d")}')
    print(f'推荐红球: {result["pred_red"]}')
    print(f'推荐蓝球: {result["pred_blue"]}  (CLF={result["blue_clf_argmax"]}, '
          f'exp={result["blue_clf_expected"]:.1f}, AR={result["blue_ar"]:.0f})')
    if result.get('ar_mu_short') is not None:
        print(f'  AR Dual: mu_long={result["ar_mu_long"]:.2f}, mu_short={result["ar_mu_short"]:.2f}, '
              f'w_short={result["ar_w_short"]:.2f}')

    print('\n红球 Top-12 得分:')
    for n, s in result['red_scores']:
        bar = '█' * int(s * 200)
        print(f'  {n:2d}: {s:.4f} {bar}')

    print('\n*** 纯娱乐预测, 请理性购彩! ***')
