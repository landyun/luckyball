# 双色球 v3.0 预测计算 - 2026-06-14 21:19
import math

# ============ 1. 八字计算 ============
year, month, day, hour = 2026, 6, 14, 21

# 年柱
g_year = (year - 3) % 10 or 10
z_year = (year - 3) % 12 or 12

# 月柱
g_month = (year % 10 + month * 2) % 10 or 10
z_month = (month + 2) % 12 or 12

# 日柱 - JDN
a = (14 - month) // 12
y = year + 4800 - a
m = month + 12*a - 3
JDN = day + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045

g_day = (JDN + 6) % 10 or 10
z_day = (JDN + 6) % 12 or 12

# 时柱
z_hour = (hour // 2 + 1) % 12 or 12
g_hour = (g_day * 2 + z_hour - 1) % 10 or 10

# 名称
tg = ['','甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
dz = ['','子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
wx_tg = ['','木','木','火','火','土','土','金','金','水','水']
wx_dz = ['','水','土','木','木','土','火','火','土','金','金','土','水']

print('=== 八字 ===')
print(f'年柱: {tg[g_year]}{dz[z_year]} ({wx_tg[g_year]}{wx_dz[z_year]})')
print(f'月柱: {tg[g_month]}{dz[z_month]} ({wx_tg[g_month]}{wx_dz[z_month]})')
print(f'日柱: {tg[g_day]}{dz[z_day]} ({wx_tg[g_day]}{wx_dz[z_day]}) - 日主')
print(f'时柱: {tg[g_hour]}{dz[z_hour]} ({wx_tg[g_hour]}{wx_dz[z_hour]})')
print(f'JDN={JDN}')

# ============ 2. 藏干 ============
cang_gan = {
    1: [10], 2: [6,10,8], 3: [1,3,5], 4: [2],
    5: [5,2,10], 6: [3,7,5], 7: [4,6], 8: [6,4,2],
    9: [7,9,5], 10: [8], 11: [5,8,4], 12: [9,1],
}

# ============ 3. 纳音 ============
full_nayin = [
    ('甲子','乙丑','海中金'),('丙寅','丁卯','炉中火'),('戊辰','己巳','大林木'),
    ('庚午','辛未','路旁土'),('壬申','癸酉','剑锋金'),('甲戌','乙亥','山头火'),
    ('丙子','丁丑','涧下水'),('戊寅','己卯','城头土'),('庚辰','辛巳','白蜡金'),
    ('壬午','癸未','杨柳木'),('甲申','乙酉','泉中水'),('丙戌','丁亥','屋上土'),
    ('戊子','己丑','霹雳火'),('庚寅','辛卯','松柏木'),('壬辰','癸巳','长流水'),
    ('甲午','乙未','沙中金'),('丙申','丁酉','山下火'),('戊戌','己亥','平地木'),
    ('庚子','辛丑','壁上土'),('壬寅','癸卯','金箔金'),('甲辰','乙巳','覆灯火'),
    ('丙午','丁未','天河水'),('戊申','己酉','大驿土'),('庚戌','辛亥','钗钏金'),
    ('壬子','癸丑','桑柘木'),('甲寅','乙卯','大溪水'),('丙辰','丁巳','沙中土'),
    ('戊午','己未','天上火'),('庚申','辛酉','石榴木'),('壬戌','癸亥','大海水'),
]

def find_nayin(g, z):
    s = tg[g] + dz[z]
    for n1, n2, name in full_nayin:
        if s == n1 or s == n2:
            return name
    return '??'

print('\n纳音:')
for label, g, z in [('年柱', g_year, z_year), ('月柱', g_month, z_month),
                     ('日柱', g_day, z_day), ('时柱', g_hour, z_hour)]:
    print(f'{label} {tg[g]}{dz[z]}: {find_nayin(g, z)}')

# 五鼠遁修正
wushu_map = {1:1, 6:1, 2:3, 7:3, 3:5, 8:5, 4:7, 9:7, 5:9, 10:9}
correct_g_hour = (wushu_map[g_day] + z_hour - 1) % 10 or 10
print(f'\n五鼠遁修正时柱: {tg[correct_g_hour]}{dz[z_hour]} ({find_nayin(correct_g_hour, z_hour)})')

# ============ 4. 十神 ============
def shishen(other_g, ri_g):
    ri_wx5 = (ri_g - 1) // 2  # 0木1火2土3金4水
    other_wx5 = (other_g - 1) // 2
    diff = (other_wx5 - ri_wx5 + 5) % 5
    # 0同我 1我生 2我克 3克我 4生我
    rel_map = {0: 'same', 1: 'isheng', 2: 'woke', 3: 'kewo', 4: 'shengwo'}
    ri_yy = 'yin' if ri_g % 2 == 0 else 'yang'
    other_yy = 'yin' if other_g % 2 == 0 else 'yang'
    same_yy = (ri_yy == other_yy)

    ss_map = {
        ('same', True): '比肩', ('same', False): '劫财',
        ('isheng', True): '食神', ('isheng', False): '伤官',
        ('woke', True): '偏财', ('woke', False): '正财',
        ('kewo', True): '七杀', ('kewo', False): '正官',
        ('shengwo', True): '偏印', ('shengwo', False): '正印',
    }
    return ss_map[(rel_map[diff], same_yy)]

print('\n十神 (日主=乙木):')
for label, g_val in [('年干', g_year), ('月干', g_month), ('时干', correct_g_hour)]:
    ss = shishen(g_val, g_day)
    print(f'{label} {tg[g_val]} → {ss}')

# ============ 5. 五行统计 ============
main_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for g_val in [g_year, g_month, g_day, correct_g_hour]:
    main_wx[wx_tg[g_val]] += 1
for z_val in [z_year, z_month, z_day, z_hour]:
    main_wx[wx_dz[z_val]] += 1

cang_wx = {'木':0,'火':0,'土':0,'金':0,'水':0}
for z_val in [z_year, z_month, z_day, z_hour]:
    for cg in cang_gan[z_val]:
        cang_wx[wx_tg[cg]] += 1

print('\n=== 五行分布 ===')
print(f'主气: {main_wx}')
print(f'藏干: {cang_wx}')

# ============ 6. 节气 ============
mangzhong, xiazhi = 5, 21
dist_mz, dist_xz = day - mangzhong, xiazhi - day
nearest = min(dist_mz, dist_xz)
print(f'\n距芒种: {dist_mz}天, 距夏至: {dist_xz}天, 最近节气距: {nearest}天')

# ============ 7. 蓝球 AR(1) ============
blue_prev, mu, phi = 16, 8.5, -0.72
blue_ar = mu + phi * (blue_prev - mu)
print(f'\n=== 蓝球 AR(1) ===')
print(f'blue_AR = {mu} + ({phi}) * ({blue_prev} - {mu}) = {blue_ar:.2f}')

# ============ 8. 红球五行评分 ============
print('\n=== 红球五行评分 (木/水偏向) ===')
scores = []
for n in range(1, 34):
    ng = (n - 1) % 10 + 1
    nz = (n - 1) % 12 + 1
    n_wx_tg = wx_tg[ng]
    n_wx_dz = wx_dz[nz]
    score = 0.0
    if n_wx_tg in ('木','水'): score += 1.0
    if n_wx_dz in ('木','水'): score += 1.0
    if n_wx_tg == '木' and n_wx_dz == '水': score += 0.5
    if n_wx_tg == '水' and n_wx_dz == '木': score += 0.5
    # 奇偶 bonus (even preferred for balance since last draw was 5:1 odd)
    parity_bonus = 0.3 if n % 2 == 0 else 0.0
    score += parity_bonus
    scores.append((n, score, tg[ng], dz[nz], n_wx_tg, n_wx_dz))
    print(f'{n:2d}({tg[ng]}{dz[nz]}={n_wx_tg}/{n_wx_dz}): {score:.1f}')

scores.sort(key=lambda x: -x[1])
print('\n=== Top-12 得分 ===')
for i, (n, s, t, d, wt, wd) in enumerate(scores[:12]):
    print(f'#{i+1}: {n:02d} ({t}{d}={wt}/{wd}) = {s:.2f}')

# ============ 9. 蓝球评分 ============
print('\n=== 蓝球评分 (1-16) ===')
for n in range(1, 17):
    ng = (n - 1) % 10 + 1
    nz = (n - 1) % 12 + 1
    n_wx_tg = wx_tg[ng]
    n_wx_dz = wx_dz[nz]
    wuxing_score = 0.0
    if n_wx_tg in ('木','水'): wuxing_score += 1.0
    if n_wx_dz in ('木','水'): wuxing_score += 1.0
    # AR proximity bonus
    ar_prox = max(0, 1.0 - abs(n - blue_ar) / 8.0)
    total = wuxing_score * 0.4 + ar_prox * 0.6
    print(f'{n:2d}({tg[ng]}{dz[nz]}={n_wx_tg}/{n_wx_dz}): wx={wuxing_score:.0f} ar_prox={ar_prox:.2f} total={total:.2f}')

print('\nDone!')
