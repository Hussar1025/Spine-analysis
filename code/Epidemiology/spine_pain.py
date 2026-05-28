import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except Exception as e:
    print(f"[警告] pmdarima加载失败({e}),将使用手工网格搜索ARIMA")
    PMDARIMA_OK = False

SPINE_FILE = r"yourinputpath"
YEAR_FILE  = r"yourinputpath"
OUTPUT_DIR = r"youroutputpath"
TARGET_YEAR = 2030
LAST_DATA_YEAR = 2023
FORECAST_HORIZON = TARGET_YEAR - LAST_DATA_YEAR

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(" " * 16 + "脊柱疾病负担综合统计分析方案 v2.1 (优化版)")
print(" " * 23 + f"预测目标: {TARGET_YEAR}年")
print("=" * 80)

raw = pd.read_excel(SPINE_FILE, sheet_name='Sheet1', header=None,
                    skiprows=4, nrows=34)
raw.columns = ['Year', 'IncidentCases', 'ASIR',
               'PrevalentCases', 'ASPR', 'YLDs', 'ASYR']

def clean_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = x.replace(' ', '').replace('\u00a0', '').replace('−', '-')
    try:
        return float(x)
    except ValueError:
        return np.nan

for c in raw.columns:
    raw[c] = raw[c].apply(clean_num)

raw = raw.dropna(subset=['Year']).reset_index(drop=True)
raw['Year'] = raw['Year'].astype(int)
df = raw.set_index('Year')

targets = {
    'IncidentCases': '发病人数(万例)',
    'ASIR':          '标化发病率(/10万)',
    'PrevalentCases':'患病人数(万例)',
    'ASPR':          '标化患病率(/10万)',
    'YLDs':          'YLDs(万人年)',
    'ASYR':          '标化YLDs率(/10万)',
}

age_raw = pd.read_excel(YEAR_FILE, sheet_name='Sheet1', header=None, skiprows=4)
age_data = pd.DataFrame({
    '年龄段': age_raw[0].astype(str).str.replace('～', '-').values,
    '发病率_1990': pd.to_numeric(age_raw[1], errors='coerce').values,
    '发病率_2023': pd.to_numeric(age_raw[2], errors='coerce').values,
    '患病率_1990': pd.to_numeric(age_raw[4], errors='coerce').values,
    '患病率_2023': pd.to_numeric(age_raw[5], errors='coerce').values,
    'YLDs率_1990': pd.to_numeric(age_raw[7], errors='coerce').values,
    'YLDs率_2023': pd.to_numeric(age_raw[8], errors='coerce').values,
})
age_data = age_data.dropna().reset_index(drop=True)

print(f"\n[数据加载成功] 时序数据: {len(df)}年 ({df.index.min()}-{df.index.max()}) | 年龄数据: {len(age_data)}段")

def aapc_with_ci(series):
    y = np.log(series.values); x = np.arange(len(y)); n = len(y)
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean)**2).sum()
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    se = np.sqrt(((y - y_pred)**2).sum() / (n - 2) / ((x - x_mean)**2).sum())
    aapc = (np.exp(slope) - 1) * 100
    return aapc, (np.exp(slope - 1.96 * se) - 1) * 100, (np.exp(slope + 1.96 * se) - 1) * 100

desc_rows = []
for col, name in targets.items():
    s = df[col]
    aapc, lo, hi = aapc_with_ci(s)
    desc_rows.append({
        '指标': name, 'Code': col,
        '1990': round(s.iloc[0], 2), '2023': round(s.iloc[-1], 2),
        '峰值': round(s.max(), 2), '峰值年份': int(s.idxmax()),
        '总变化(%)': round((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100, 2),
        'AAPC(%)': round(aapc, 3),
        'AAPC_CI下': round(lo, 3), 'AAPC_CI上': round(hi, 3),
        'AAPC显著': '是' if (lo > 0 or hi < 0) else '否',
        '均值': round(s.mean(), 2), '标准差': round(s.std(), 2),
        '变异系数(%)': round(s.std() / s.mean() * 100, 2),
    })
desc_df = pd.DataFrame(desc_rows)
print(desc_df.drop(columns=['Code']).to_string(index=False))

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
for i, (col, name) in enumerate(targets.items()):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    s = df[col]
    ax.plot(s.index, s.values, '-o', color='#1f4e79', markersize=4, lw=1.8)
    ax.fill_between(s.index, s.values, alpha=0.15, color='#1f4e79')
    d = desc_rows[i]
    sig = '★' if d['AAPC显著'] == '是' else ''
    ax.set_title(f"{name}\nAAPC={d['AAPC(%)']:.2f}% ({d['AAPC_CI下']:.2f}, {d['AAPC_CI上']:.2f}) {sig}",
                 fontsize=11)
    ax.set_xlabel('年份'); ax.grid(alpha=0.3)
    ax.annotate(f'{s.iloc[0]:.1f}', xy=(s.index[0], s.iloc[0]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.annotate(f'{s.iloc[-1]:.1f}', xy=(s.index[-1], s.iloc[-1]),
                xytext=(-30, 5), textcoords='offset points', fontsize=8, color='red')
plt.suptitle('模块1: 六指标历史趋势与AAPC', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'Module1_descriptive_trends.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("已保存图: Module1_descriptive_trends.png")

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(targets))
for i, r in enumerate(desc_rows):
    c = '#c00000' if r['AAPC显著'] == '是' else '#999999'
    ax.plot([r['AAPC_CI下'], r['AAPC_CI上']], [i, i], '-', color=c, lw=2.5)
    ax.plot(r['AAPC(%)'], i, 's', color=c, markersize=12)
    ax.annotate(f"{r['AAPC(%)']:.3f}% ({r['AAPC_CI下']:.2f}, {r['AAPC_CI上']:.2f})",
                xy=(r['AAPC_CI上'], i), xytext=(10, 0), textcoords='offset points',
                fontsize=10, va='center')
ax.axvline(0, color='black', ls='--', lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([r['指标'] for r in desc_rows])
ax.set_xlabel('AAPC (%) with 95% CI')
ax.set_title('模块1: 各指标年均变化率(AAPC)森林图\n红色=显著, 灰色=不显著',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Module1_AAPC_forest.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("已保存图: Module1_AAPC_forest.png")

def fit_continuous_joinpoint(x_idx, log_values, joinpoints):
    n = len(x_idx)
    X = np.column_stack([np.ones(n), x_idx])  # 截距 + 时间
    for jp in joinpoints:
        X = np.column_stack([X, np.maximum(0, x_idx - jp)])
    coef, *_ = np.linalg.lstsq(X, log_values, rcond=None)
    y_pred = X @ coef
    rss = ((log_values - y_pred)**2).sum()

    bp_all = [int(x_idx[0])] + sorted([int(j) for j in joinpoints]) + [int(x_idx[-1])]
    segments = []
    current_slope = coef[1]
    for i in range(len(bp_all) - 1):
        start_i, end_i = bp_all[i], bp_all[i+1]
        if i > 0:
            current_slope = coef[1] + sum(coef[2 + k] for k in range(i))
        apc = (np.exp(current_slope) - 1) * 100
        segments.append({
            'start_idx': start_i, 'end_idx': end_i,
            'APC(%)': round(apc, 3), 'slope': current_slope,
            'n_years': end_i - start_i + 1,
        })
    return rss, segments, y_pred, coef

def find_continuous_joinpoints(series, max_jp=3):
    n = len(series)
    x_idx = np.arange(n)
    log_vals = np.log(series.values)
    results = []
    rss, segs, _, _ = fit_continuous_joinpoint(x_idx, log_vals, [])
    k = 2
    bic = n * np.log(rss / n) + k * np.log(n)
    results.append({'n_jp': 0, 'jp': [], 'bic': bic, 'segments': segs, 'rss': rss})
    for n_jp in range(1, max_jp + 1):
        best = None
        for combo in combinations(range(3, n - 3), n_jp):
            if any(combo[i+1] - combo[i] < 3 for i in range(len(combo) - 1)):
                continue
            try:
                rss, segs, _, _ = fit_continuous_joinpoint(x_idx, log_vals, list(combo))
            except Exception:
                continue
            k = 2 + n_jp
            bic = n * np.log(rss / n) + k * np.log(n)
            if best is None or bic < best['bic']:
                best = {'n_jp': n_jp, 'jp': list(combo), 'bic': bic,
                        'segments': segs, 'rss': rss}
        if best: results.append(best)
    return min(results, key=lambda x: x['bic'])

jp_summary = []; jp_details = {}; jp_seg_rows = []
for col, name in targets.items():
    s = df[col]
    best = find_continuous_joinpoints(s, max_jp=3)
    jp_years = [int(df.index[idx]) for idx in best['jp']]
    jp_summary.append({
        '指标': name, 'Code': col,
        '最优拐点数': best['n_jp'],
        '拐点年份': str(jp_years) if jp_years else '无',
        'BIC': round(best['bic'], 2),
        '分段数': len(best['segments']),
    })
    jp_details[col] = best
    for seg in best['segments']:
        yr_start = int(df.index[seg['start_idx']])
        yr_end = int(df.index[seg['end_idx']])
        jp_seg_rows.append({
            '指标': name,
            '起始年': yr_start, '结束年': yr_end,
            '持续年数': seg['n_years'],
            'APC(%)': seg['APC(%)'],
            '趋势': '上升↑' if seg['APC(%)'] > 0.1 else
                   ('下降↓' if seg['APC(%)'] < -0.1 else '平稳→'),
        })

print(pd.DataFrame(jp_summary).drop(columns=['Code']).to_string(index=False))
jp_seg_df = pd.DataFrame(jp_seg_rows)
print("\n--- 各分段APC ---")
print(jp_seg_df.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
seg_palette = ['#c00000', '#2e7d32', '#1565c0', '#f57c00']
for ax, (col, name) in zip(axes.flatten(), targets.items()):
    s = df[col]
    best = jp_details[col]
    x_idx = np.arange(len(s))
    log_vals = np.log(s.values)
    _, segs, y_pred_log, _ = fit_continuous_joinpoint(x_idx, log_vals, best['jp'])
    y_pred = np.exp(y_pred_log)
    ax.plot(s.index, s.values, 'o', color='#333333', markersize=5, zorder=10, label='观测')
    for i, seg in enumerate(segs):
        idx_range = np.arange(seg['start_idx'], seg['end_idx'] + 1)
        ax.plot(s.index[idx_range], y_pred[idx_range], '-',
                color=seg_palette[i % 4], lw=2.5,
                label=f"{int(s.index[seg['start_idx']])}-{int(s.index[seg['end_idx']])}: "
                      f"APC={seg['APC(%)']:.2f}%")
    for jp_idx in best['jp']:
        jp_year = int(df.index[jp_idx])
        ax.axvline(jp_year, color='red', ls=':', alpha=0.6, lw=1.5)
        y_text = s.min() + (s.max() - s.min()) * 0.05
        ax.annotate(f'拐点\n{jp_year}', xy=(jp_year, y_text),
                    ha='center', fontsize=8, color='red', fontweight='bold')
    ax.set_title(f'{name}  (拐点数={best["n_jp"]}, 连续分段)', fontsize=11)
    ax.set_xlabel('年份'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
plt.suptitle('模块2: Joinpoint趋势拐点(连续分段线性回归)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Module2_joinpoint.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("已保存图: Module2_joinpoint.png")

def stat_test(series, label):
    s = series.dropna()
    try: adf_p = adfuller(s, autolag='AIC')[1]
    except Exception: adf_p = np.nan
    try: kpss_p = kpss(s, regression='c', nlags='auto')[1]
    except Exception: kpss_p = np.nan
    return {
        '指标': label,
        'ADF_p': round(adf_p, 4),
        'ADF结论': '平稳' if adf_p < 0.05 else '非平稳',
        'KPSS_p': round(kpss_p, 4),
        'KPSS结论': '平稳' if kpss_p > 0.05 else '非平稳',
    }

stat_rows = []
diff_orders_diag = {} 
for col, name in targets.items():
    r0 = stat_test(df[col], f'{name}_原序列')
    r1 = stat_test(df[col].diff(), f'{name}_一阶差分')
    r2 = stat_test(df[col].diff().diff(), f'{name}_二阶差分')
    stat_rows.extend([r0, r1, r2])
    if r0['ADF结论'] == '平稳': diff_orders_diag[col] = 0
    elif r1['ADF结论'] == '平稳': diff_orders_diag[col] = 1
    else: diff_orders_diag[col] = 2

stat_df = pd.DataFrame(stat_rows)
print(stat_df.to_string(index=False))

fig, axes = plt.subplots(3, 6, figsize=(22, 10))
for col_idx, (col, name) in enumerate(targets.items()):
    s = df[col]
    series_list = [
        (s, '原', adfuller(s.dropna(), autolag='AIC')[1]),
        (s.diff(), 'd1', adfuller(s.diff().dropna(), autolag='AIC')[1]),
        (s.diff().diff(), 'd2', adfuller(s.diff().diff().dropna(), autolag='AIC')[1]),
    ]
    for row_idx, (data, lbl, p) in enumerate(series_list):
        ax = axes[row_idx, col_idx]
        ax.plot(data.index, data.values, '-o', color='#1f4e79', markersize=2, lw=1)
        ax.axhline(0, color='red', ls='--', lw=0.5)
        color = '#2e7d32' if p < 0.05 else '#c00000'
        ax.set_title(f'{name[:8]}\n{lbl} (ADF p={p:.3f})', fontsize=8, color=color)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
plt.suptitle('模块3: 平稳性诊断 (绿=平稳, 红=非平稳)',
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Module3_stationarity.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("已保存图: Module3_stationarity.png")

future_years = np.arange(LAST_DATA_YEAR + 1, TARGET_YEAR + 1)

def manual_arima_search(train, max_p=3, max_q=3, max_d=2):
    best_aic = np.inf; best_order = None; best_model = None
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0: continue
                try:
                    m = ARIMA(train, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic; best_order = (p, d, q); best_model = m
                except Exception: continue
    return best_model, best_order

def predict_arima(train, horizon):
    if PMDARIMA_OK:
        try:
            auto = pm.auto_arima(train, start_p=0, start_q=0, max_p=4, max_q=4,
                                  max_d=2, seasonal=False, stepwise=True,
                                  information_criterion='aicc',
                                  suppress_warnings=True, error_action='ignore')
            order = auto.order
            model = ARIMA(train, order=order).fit()
        except Exception:
            model, order = manual_arima_search(train)
    else:
        model, order = manual_arima_search(train)
    fc = model.get_forecast(steps=horizon)
    return fc.predicted_mean.values, fc.conf_int(alpha=0.05).values, f'ARIMA{order}', order

def predict_ets(train, horizon):
    model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=True)
    results = model.fit(optimized=True)
    fc = results.forecast(horizon).values
    try:
        pred = results.get_prediction(start=len(train), end=len(train) + horizon - 1)
        frame = pred.summary_frame(alpha=0.05)
        if 'mean_ci_lower' in frame.columns:
            ci = frame[['mean_ci_lower', 'mean_ci_upper']].values
            return fc, ci, 'ETS(A,Ad,N)'
    except Exception: pass
    resid = results.resid.dropna().values
    n_sim = 1000
    sim_paths = np.zeros((n_sim, horizon))
    for i in range(n_sim):
        noise = np.random.choice(resid, size=horizon, replace=True)
        sim_paths[i] = fc + noise.cumsum() * 0.5 
    ci_lo = np.percentile(sim_paths, 2.5, axis=0)
    ci_hi = np.percentile(sim_paths, 97.5, axis=0)
    return fc, np.column_stack([ci_lo, ci_hi]), 'ETS(A,Ad,N)'

def predict_linear(train, horizon):
    x = np.arange(len(train)).reshape(-1, 1)
    lr = LinearRegression().fit(x, train.values)
    x_f = np.arange(len(train), len(train) + horizon).reshape(-1, 1)
    fc = lr.predict(x_f)
    se = np.std(train.values - lr.predict(x))
    ci = np.column_stack([fc - 1.96 * se, fc + 1.96 * se])
    return fc, ci, 'Linear'

def predict_quadratic(train, horizon):
    x = np.arange(len(train))
    coef = np.polyfit(x, train.values, deg=2)
    p = np.poly1d(coef)
    x_f = np.arange(len(train), len(train) + horizon)
    fc = p(x_f)
    se = np.std(train.values - p(x))
    ci = np.column_stack([fc - 1.96 * se, fc + 1.96 * se])
    return fc, ci, 'Quadratic'

cv_results = {col: {} for col in targets}
val_rows = []
arima_d_chosen = {}

for col, name in targets.items():
    s_train = df[col].iloc[:-1]
    actual = df[col].iloc[-1]
    for method, func in [('ARIMA', predict_arima), ('ETS', predict_ets),
                          ('Linear', predict_linear), ('Quadratic', predict_quadratic)]:
        try:
            out = func(s_train, 1)
            fc, ci, info = out[0], out[1], out[2]
            if method == 'ARIMA' and len(out) >= 4:
                arima_d_chosen[col] = out[3][1]
            mape = abs(fc[0] - actual) / actual * 100
            cv_results[col][method] = mape
            val_rows.append({
                '指标': name, '模型': method, '模型信息': info,
                '2023实际': round(actual, 2),
                '2023预测': round(fc[0], 2),
                'MAPE(%)': round(mape, 3),
            })
        except Exception as e:
            cv_results[col][method] = np.inf
            val_rows.append({'指标': name, '模型': method,
                              '模型信息': f'失败:{str(e)[:30]}',
                              '2023实际': round(actual, 2),
                              '2023预测': np.nan, 'MAPE(%)': np.inf})

cv_df = pd.DataFrame(val_rows)
print(cv_df.to_string(index=False))

best_model_rows = []
for col, name in targets.items():
    best_method = min(cv_results[col], key=cv_results[col].get)
    best_model_rows.append({
        '指标': name, 'Code': col,
        '最优模型': best_method,
        '验证MAPE(%)': round(cv_results[col][best_method], 3),
    })
best_df = pd.DataFrame(best_model_rows)
print("\n--- 各指标最优模型 ---")
print(best_df.drop(columns=['Code']).to_string(index=False))

all_method_forecasts = {col: {} for col in targets}
final_forecasts = {}
forecast_rows = []
arima_d_final = {}

for col, name in targets.items():
    s = df[col]
    for method, func in [('ARIMA', predict_arima), ('ETS', predict_ets),
                          ('Linear', predict_linear), ('Quadratic', predict_quadratic)]:
        try:
            out = func(s, FORECAST_HORIZON)
            fc, ci, info = out[0], out[1], out[2]
            if method == 'ARIMA' and len(out) >= 4:
                arima_d_final[col] = out[3][1]
            all_method_forecasts[col][method] = {'fc': fc, 'ci': ci, 'info': info}
        except Exception: pass

    available = [m for m in ['ARIMA', 'ETS', 'Linear', 'Quadratic']
                  if cv_results[col].get(m, np.inf) < np.inf]
    if available:
        mapes = np.array([cv_results[col][m] for m in available])
        T = max(mapes.std(), 0.5)
        log_w = -mapes / T
        log_w -= log_w.max()
        w = np.exp(log_w); w /= w.sum()
        w = np.minimum(w, 0.6); w /= w.sum()
        weights = dict(zip(available, w))
        ens_fc = sum(weights[m] * all_method_forecasts[col][m]['fc'] for m in available)
        ens_lo = sum(weights[m] * all_method_forecasts[col][m]['ci'][:, 0] for m in available)
        ens_hi = sum(weights[m] * all_method_forecasts[col][m]['ci'][:, 1] for m in available)
        all_method_forecasts[col]['Ensemble'] = {
            'fc': ens_fc, 'ci': np.column_stack([ens_lo, ens_hi]),
            'info': 'Softmax-weighted ensemble', 'weights': weights
        }

    best_method = min(cv_results[col], key=cv_results[col].get)
    final_forecasts[col] = {'best_method': best_method}
    best_fc = all_method_forecasts[col][best_method]['fc']
    best_ci = all_method_forecasts[col][best_method]['ci']
    for i, yr in enumerate(future_years):
        forecast_rows.append({
            '指标': name, '年份': int(yr),
            '最优模型': best_method,
            '预测值': round(best_fc[i], 2),
            '95%CI下': round(best_ci[i, 0], 2),
            '95%CI上': round(best_ci[i, 1], 2),
            '集成预测': round(all_method_forecasts[col]['Ensemble']['fc'][i], 2)
                       if 'Ensemble' in all_method_forecasts[col] else np.nan,
        })
forecast_df = pd.DataFrame(forecast_rows)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
model_colors = {'ARIMA':'#c00000', 'ETS':'#2e7d32', 'Linear':'#1565c0',
                'Quadratic':'#f57c00', 'Ensemble':'#6a1b9a'}
for ax, (col, name) in zip(axes.flatten(), targets.items()):
    s = df[col]
    ax.plot(s.index, s.values, '-o', color='black', markersize=4, lw=1.8, label='观测', zorder=10)
    best_m = final_forecasts[col]['best_method']
    for m, color in model_colors.items():
        if m in all_method_forecasts[col]:
            fc = all_method_forecasts[col][m]['fc']
            is_best = (m == best_m)
            ls = '-' if is_best else '--'
            lw = 2.2 if is_best else 1.2
            alpha = 1.0 if is_best else 0.6
            label = f'★{m}(最优)' if is_best else m
            ax.plot(future_years, fc, ls, color=color, lw=lw, alpha=alpha,
                    label=label, marker='s' if is_best else None, markersize=4)
    best_ci = all_method_forecasts[col][best_m]['ci']
    ax.fill_between(future_years, best_ci[:, 0], best_ci[:, 1],
                    color=model_colors[best_m], alpha=0.12)
    ax.axvline(LAST_DATA_YEAR + 0.5, color='gray', ls=':', lw=0.8)
    best_2030 = all_method_forecasts[col][best_m]['fc'][-1]
    ax.annotate(f'{TARGET_YEAR}:{best_2030:.1f}',
                xy=(TARGET_YEAR, best_2030), xytext=(8, 0),
                textcoords='offset points', fontsize=9,
                color=model_colors[best_m], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_title(f'{name}  最优:{best_m}', fontsize=11)
    ax.set_xlabel('年份'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='best', ncol=2)
plt.suptitle(f'模块4: 五模型预测对比 (1990-{TARGET_YEAR})',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Module4_multimodel_forecast.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
methods_list = ['ARIMA', 'ETS', 'Linear', 'Quadratic']
mape_matrix = np.array([[cv_results[col].get(m, np.nan) for m in methods_list] for col in targets])
ax = axes[0]
x_pos = np.arange(len(targets))
width = 0.2
bar_colors = ['#c00000', '#2e7d32', '#1565c0', '#f57c00']
for i, m in enumerate(methods_list):
    ax.bar(x_pos + (i - 1.5) * width, mape_matrix[:, i], width, label=m, color=bar_colors[i], alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels([targets[c] for c in targets], rotation=20, fontsize=9, ha='right')
ax.set_ylabel('MAPE (%)'); ax.set_title('各模型验证误差对比(2023预测)', fontsize=12)
ax.legend(); ax.grid(alpha=0.3, axis='y')

ax = axes[1]
im = ax.imshow(mape_matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(methods_list)))
ax.set_xticklabels(methods_list)
ax.set_yticks(range(len(targets)))
ax.set_yticklabels([targets[c] for c in targets])
for i in range(len(targets)):
    best_j = np.nanargmin(mape_matrix[i])
    for j in range(len(methods_list)):
        val = mape_matrix[i, j]
        text_color = 'white' if val > np.nanmean(mape_matrix) else 'black'
        weight = 'bold' if j == best_j else 'normal'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=text_color, fontweight=weight)
        if j == best_j:
            ax.add_patch(plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, fill=False, edgecolor='blue', lw=3))
plt.colorbar(im, ax=ax, label='MAPE(%)'); ax.set_title('MAPE热图(蓝框=最优)', fontsize=12)
plt.suptitle('模块5: 留一交叉验证与最优模型选择', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Module5_model_validation.png'), dpi=150, bbox_inches='tight')
plt.close()

for metric in ['发病率', '患病率', 'YLDs率']:
    age_data[f'{metric}_变化(%)'] = ((age_data[f'{metric}_2023'] -
                                  age_data[f'{metric}_1990']) /
                                 age_data[f'{metric}_1990'] * 100).round(2)
    age_data[f'{metric}_AAPC(%)'] = (
        ((age_data[f'{metric}_2023'] / age_data[f'{metric}_1990']) ** (1/33) - 1) * 100
    ).round(3)

worsen_rows = []
for metric in ['发病率', '患病率', 'YLDs率']:
    worsen = age_data[age_data[f'{metric}_变化(%)'] > 0]['年龄段'].tolist()
    improve = age_data[age_data[f'{metric}_变化(%)'] < 0]['年龄段'].tolist()
    max_w = age_data.loc[age_data[f'{metric}_变化(%)'].idxmax()]
    max_i = age_data.loc[age_data[f'{metric}_变化(%)'].idxmin()]
    worsen_rows.append({
        '指标': metric,
        '恶化组数': len(worsen), '恶化年龄段': ', '.join(worsen) if worsen else '无',
        '改善组数': len(improve), '改善年龄段': ', '.join(improve) if improve else '无',
        '最恶化': f"{max_w['年龄段']} ({max_w[f'{metric}_变化(%)']}%)",
        '最改善': f"{max_i['年龄段']} ({max_i[f'{metric}_变化(%)']}%)",
    })
worsen_df = pd.DataFrame(worsen_rows)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
for col_idx, metric in enumerate(['发病率', '患病率', 'YLDs率']):
    ax = fig.add_subplot(gs[0, col_idx])
    x_pos = np.arange(len(age_data))
    width = 0.4
    ax.bar(x_pos - width/2, age_data[f'{metric}_1990'], width, label='1990', color='#1f4e79', alpha=0.85)
    ax.bar(x_pos + width/2, age_data[f'{metric}_2023'], width, label='2023', color='#c00000', alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(age_data['年龄段'], rotation=45, fontsize=7, ha='right')
    ax.set_title(f'{metric} (/10万)', fontsize=11, fontweight='bold')
    ax.set_xlabel('年龄段'); ax.legend(); ax.grid(alpha=0.3, axis='y')

ax = fig.add_subplot(gs[1, 0])
change_matrix = age_data[['发病率_变化(%)', '患病率_变化(%)', 'YLDs率_变化(%)']].values
im = ax.imshow(change_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-30, vmax=30)
ax.set_xticks(range(3)); ax.set_xticklabels(['发病率', '患病率', 'YLDs率'])
ax.set_yticks(range(len(age_data))); ax.set_yticklabels(age_data['年龄段'], fontsize=8)
for i in range(len(age_data)):
    for j in range(3):
        ax.text(j, i, f'{change_matrix[i, j]:.1f}%', ha='center', va='center', fontsize=8,
                color='white' if abs(change_matrix[i, j]) > 15 else 'black')
plt.colorbar(im, ax=ax, label='变化率(%)'); ax.set_title('年龄段变化率热图', fontsize=10, fontweight='bold')

ax = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(age_data))
colors_3 = ['#c00000', '#2e7d32', '#1565c0']
for metric, color in zip(['发病率', '患病率', 'YLDs率'], colors_3):
    ax.plot(x_pos, age_data[f'{metric}_变化(%)'], '-o', color=color, label=metric, markersize=5, lw=1.5)
ax.axhline(0, color='black', ls='--', lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(age_data['年龄段'], rotation=45, fontsize=7, ha='right')
ax.set_ylabel('变化率(%)'); ax.set_title('三指标各年龄段变化率', fontsize=10, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(x_pos, age_data['发病率_1990'], '-o', color='#1f4e79', label='1990', markersize=5)
ax.plot(x_pos, age_data['发病率_2023'], '-s', color='#c00000', label='2023', markersize=5)
peak_1990 = age_data['发病率_1990'].idxmax()
peak_2023 = age_data['发病率_2023'].idxmax()
ax.axvline(peak_1990, color='#1f4e79', ls=':', alpha=0.5)
ax.axvline(peak_2023, color='#c00000', ls=':', alpha=0.5)
ax.annotate(f"1990峰\n{age_data.loc[peak_1990, '年龄段']}", xy=(peak_1990, age_data.loc[peak_1990, '发病率_1990']),
            xytext=(-30, 20), textcoords='offset points', fontsize=8, color='#1f4e79', arrowprops=dict(arrowstyle='->', color='#1f4e79'))
ax.annotate(f"2023峰\n{age_data.loc[peak_2023, '年龄段']}", xy=(peak_2023, age_data.loc[peak_2023, '发病率_2023']),
            xytext=(10, 20), textcoords='offset points', fontsize=8, color='#c00000', arrowprops=dict(arrowstyle='->', color='#c00000'))
ax.set_xticks(x_pos); ax.set_xticklabels(age_data['年龄段'], rotation=45, fontsize=7, ha='right')
ax.set_ylabel('发病率 (/10万)'); ax.set_title('发病率峰值年龄迁移', fontsize=10, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('模块6: 年龄结构变化综合分析 (1990 vs 2023)', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'Module6_age_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

summary_rows = []
for col, name in targets.items():
    s = df[col]
    desc = next(r for r in desc_rows if r['Code'] == col)
    jp = jp_details[col]
    best_m = final_forecasts[col]['best_method']
    best_fc = all_method_forecasts[col][best_m]['fc']
    best_ci = all_method_forecasts[col][best_m]['ci']
    ens_fc = all_method_forecasts[col]['Ensemble']['fc']

    jp_years_str = str([int(df.index[i]) for i in jp['jp']]) if jp['jp'] else '无'
    seg_apcs = ' | '.join([
        f"{int(df.index[seg['start_idx']])}-{int(df.index[seg['end_idx']])}:APC={seg['APC(%)']}%"
        for seg in jp['segments']])

    d_diag = diff_orders_diag.get(col, '?')
    d_auto = arima_d_final.get(col, '?')
    d_match = '一致' if d_diag == d_auto else '不一致'

    summary_rows.append({
        '指标': name,
        '1990基线': desc['1990'], '2023现值': desc['2023'],
        '历史峰值': desc['峰值'], '峰值年': desc['峰值年份'],
        '总变化(%)': desc['总变化(%)'],
        'AAPC(%)': desc['AAPC(%)'],
        'AAPC_95%CI': f"({desc['AAPC_CI下']}, {desc['AAPC_CI上']})",
        '趋势显著性': desc['AAPC显著'],
        '拐点数': jp['n_jp'], '拐点年份': jp_years_str,
        '分段APC(连续)': seg_apcs,
        'd_人工诊断': d_diag, 'd_ARIMA自动': d_auto, 'd阶一致性': d_match,
        '最优模型': best_m, '验证MAPE(%)': round(cv_results[col][best_m], 3),
        f'{TARGET_YEAR}预测': round(best_fc[-1], 2),
        f'{TARGET_YEAR}_CI下': round(best_ci[-1, 0], 2),
        f'{TARGET_YEAR}_CI上': round(best_ci[-1, 1], 2),
        f'{TARGET_YEAR}_集成': round(ens_fc[-1], 2),
        f'23→30增幅(%)': round((best_fc[-1] - desc['2023']) / desc['2023'] * 100, 2),
    })
summary_df = pd.DataFrame(summary_rows)

fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.3)
for i, (col, name) in enumerate(targets.items()):
    ax = fig.add_subplot(gs[0, i])
    desc = next(r for r in desc_rows if r['Code'] == col)
    best_m = final_forecasts[col]['best_method']
    best_2030 = all_method_forecasts[col][best_m]['fc'][-1]
    growth_pct = (best_2030 - desc['2023']) / desc['2023'] * 100
    ax.axis('off')
    color = '#c00000' if growth_pct > 0 else '#2e7d32'
    text = f"""【{name}】

1990: {desc['1990']:.1f}
2023: {desc['2023']:.1f}
{TARGET_YEAR}: {best_2030:.1f}

AAPC: {desc['AAPC(%)']:.2f}%
拐点: {jp_details[col]['n_jp']}个
模型: {best_m}

23→30:
{growth_pct:+.1f}%
"""
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor=color, lw=2))

for i, (col, name) in enumerate(targets.items()):
    ax = fig.add_subplot(gs[1, i])
    s = df[col]
    best_m = final_forecasts[col]['best_method']
    best_fc = all_method_forecasts[col][best_m]['fc']
    best_ci = all_method_forecasts[col][best_m]['ci']
    ax.plot(s.index, s.values, '-', color='#1f4e79', lw=1.5)
    ax.plot(future_years, best_fc, '--', color='#c00000', lw=1.5)
    ax.fill_between(future_years, best_ci[:, 0], best_ci[:, 1], color='#c00000', alpha=0.15)
    ax.axvline(LAST_DATA_YEAR + 0.5, color='gray', ls=':', lw=0.8)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.3)

for i, (col, name) in enumerate(targets.items()):
    ax = fig.add_subplot(gs[2, i])
    jp = jp_details[col]
    seg_labels = [f"{int(df.index[s['start_idx']])}-{int(df.index[s['end_idx']])}" for s in jp['segments']]
    seg_apcs = [s['APC(%)'] for s in jp['segments']]
    bar_colors = ['#c00000' if v > 0 else '#2e7d32' for v in seg_apcs]
    ax.barh(range(len(seg_labels)), seg_apcs, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(seg_labels))); ax.set_yticklabels(seg_labels, fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('APC(%)', fontsize=8); ax.set_title(name, fontsize=9)
    for j, v in enumerate(seg_apcs):
        ax.text(v, j, f'{v:.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
    ax.grid(alpha=0.3, axis='x')

plt.suptitle('★ 综合汇总: 六指标"故事卡片+趋势+连续分段APC" ★', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'Module7_summary_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()

output_xlsx = os.path.join(OUTPUT_DIR, 'spine_complete_analysis_v2.xlsx')
with pd.ExcelWriter(output_xlsx, engine='openpyxl') as w:
    summary_df.to_excel(w, sheet_name='★综合汇总表', index=False)
    df.reset_index().to_excel(w, sheet_name='0_原始时序', index=False)
    desc_df.drop(columns=['Code']).to_excel(w, sheet_name='M1_描述统计_AAPC', index=False)
    pd.DataFrame(jp_summary).drop(columns=['Code']).to_excel(w, sheet_name='M2_拐点汇总', index=False)
    jp_seg_df.to_excel(w, sheet_name='M2_拐点分段(连续型)', index=False)
    stat_df.to_excel(w, sheet_name='M3_平稳性检验', index=False)
    cv_df.to_excel(w, sheet_name='M4_模型验证', index=False)
    best_df.drop(columns=['Code']).to_excel(w, sheet_name='M5_最优模型', index=False)
    forecast_df.to_excel(w, sheet_name=f'M5_2024_{TARGET_YEAR}逐年预测', index=False)
    age_data.to_excel(w, sheet_name='M6_年龄维度', index=False)
    worsen_df.to_excel(w, sheet_name='M6_年龄变化总览', index=False)

print("All analysis OK!")