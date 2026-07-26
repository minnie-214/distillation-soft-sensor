"""
Distillation Column Soft Sensor — Preprocessing & Feature Engineering
=====================================================================
Python version (pandas + scikit-learn + matplotlib)
Maps to JD: 工况时序数据降噪、时延补偿、特征工程

Usage: python distillation_soft_sensor.py
Output: output/report_python.png
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

N_POINTS = 1000
SEED = 42
BASE = {"F": 300, "T_top": 82, "L": 150, "B": 118, "purity": 0.95}
PULSE_PROB = 0.005
DELAYS = {"F": 15, "T_top": 3, "L": 5, "B": 10}
MISSING_RANDOM = 0.02
MISSING_BLOCK = 0.005
STEP_AT = 500
STEP_SIZE = 40
WINDOW_SIZE = 30
STEP = 5
FILTER_WIN = 5
TEST_RATIO = 0.3

rng = np.random.RandomState(SEED)

# ============================================================
# Step 1: Data Generation
# ============================================================
print("[1/5] Generating distillation column data...")
t = np.arange(N_POINTS)
F = (BASE["F"] + np.sin(t/60*np.pi)*40 + np.sin(t/15*np.pi)*10 + (rng.rand(N_POINTS)-0.5)*8)
F = np.clip(F, 200, 450);
F[t >= STEP_AT] += STEP_SIZE

T_top = BASE["T_top"] + (F-BASE["F"])*0.025 + np.sin(t/25*np.pi)*3.0 + (rng.rand(N_POINTS)-0.5)*0.8
T_top = np.clip(T_top, 75, 92)

L = BASE["L"] + np.sin(t/35*np.pi)*35 + (rng.rand(N_POINTS)-0.5)*10
L = np.clip(L, 80, 220)

B = BASE["B"] + (F-BASE["F"])*0.015 + np.sin(t/30*np.pi)*2.0 + (rng.rand(N_POINTS)-0.5)*0.8
B = np.clip(B, 110, 130)

purity = np.zeros(N_POINTS)
for i in range(N_POINTS):
    idxF = max(0, i-DELAYS["F"]); idxT = max(0, i-DELAYS["T_top"])
    idxL = max(0, i-DELAYS["L"]); idxB = max(0, i-DELAYS["B"])
    D = F[idxF]*(0.35+0.1*np.sin(i/80*np.pi)); R = L[idxL]/max(D,1)
    alpha = np.clip(2.4-(T_top[idxT]-80)*0.015, 1.5, 3.5)
    Rn = (R-0.5)/2.0; baseP = 0.88+0.10*(1-np.exp(-Rn*2.0))
    purity[i] = baseP -0.015*(T_top[idxT]-BASE["T_top"]) +0.005*(B[idxB]-BASE["B"]) -0.0003*(F[idxF]-BASE["F"]) -0.0008*(T_top[idxT]-BASE["T_top"])*(F[idxF]-BASE["F"])/50
purity = np.clip(purity, 0.75, 0.995)

# Noise
F += (rng.rand(N_POINTS)-0.5)*6; T_top += (rng.rand(N_POINTS)-0.5)*0.6
L += (rng.rand(N_POINTS)-0.5)*6; B += (rng.rand(N_POINTS)-0.5)*0.6
purity += (rng.rand(N_POINTS)-0.5)*0.004
pulse = rng.rand(N_POINTS) < PULSE_PROB; T_top[pulse] += (rng.rand(pulse.sum())-0.5)*4
purity = np.clip(purity, 0.70, 1.0)

# Missing values
df = pd.DataFrame({"time":t, "F":F, "T_top":T_top, "L":L, "B":B, "purity":purity})
for col in ["F","T_top","L","B"]:
    df.loc[rng.rand(N_POINTS) < MISSING_RANDOM, col] = np.nan
    for s in np.where(rng.rand(N_POINTS) < MISSING_BLOCK)[0]:
        for k in range(rng.randint(3,7)):
            if s+k < N_POINTS: df.loc[s+k, col] = np.nan

print(f"  -> {N_POINTS} time steps generated")
print(f"  -> Purity range: {df.purity.min()*100:.1f}% ~ {df.purity.max()*100:.1f}%")

# ============================================================
# Step 2: Preprocessing
# ============================================================
print("\n[2/5] Running preprocessing pipeline...")
df_imp = df.interpolate(method="linear", limit_direction="both").bfill().ffill()

def estimate_delay(x, y, max_lag=30):
    best_lag, best_corr = 0, -np.inf
    for lag in range(max_lag+1):
        if len(x) <= lag: continue
        c = np.corrcoef(x[:len(y)-lag], y[lag:])[0,1]
        if np.isfinite(c) and c > best_corr: best_corr=c; best_lag=lag
    return best_lag

delays = {col: estimate_delay(df_imp[col].values, df_imp.purity.values) for col in ["F","T_top","L","B"]}
print(f"  -> Estimated delays: {delays}")

for col in ["F","T_top","L","B","purity"]:
    df_imp[col] = df_imp[col].rolling(FILTER_WIN, center=True, min_periods=1).mean()

def extract_aligned(df, delays, ws, step):
    rows = []; md = max(delays.values())
    for i in range(md+ws, len(df), step):
        feats = []
        for col, d in delays.items(): feats.extend(df[col].values[i-ws-d:i-d])
        rows.append((feats, df.purity.values[i]))
    return rows

aligned = extract_aligned(df_imp, delays, WINDOW_SIZE, STEP)
fixed = [([v for col in ["F","T_top","L","B"] for v in df_imp[col].values[i-WINDOW_SIZE:i]], df_imp.purity.values[i]) for i in range(WINDOW_SIZE, len(df_imp), STEP)]

def extract_features(windows):
    X, y = [], []
    for feats, target in windows:
        ws = len(feats)//4; vec = []
        for v in range(4):
            s = feats[v*ws:(v+1)*ws]
            vec += [np.mean(s), np.std(s), np.min(s), np.max(s), np.polyfit(range(len(s)),s,1)[0], np.ptp(s)]
        X.append(vec); y.append(target)
    return np.array(X), np.array(y)

X_a, y_a = extract_features(aligned); X_f, y_f = extract_features(fixed)
print(f"  -> Aligned windows: {len(aligned)}, Fixed windows: {len(fixed)}")
print(f"  -> Feature dim: {X_a.shape[1]}")

# ============================================================
# Step 3: Modeling
# ============================================================
print("\n[3/5] Training models...")
sp_a = int(len(X_a)*(1-TEST_RATIO)); sp_f = int(len(X_f)*(1-TEST_RATIO))

rf_a = RandomForestRegressor(100, max_depth=8, random_state=42).fit(X_a[:sp_a], y_a[:sp_a])
pred_a = rf_a.predict(X_a[sp_a:]); r2_a = r2_score(y_a[sp_a:], pred_a); rmse_a = np.sqrt(mean_squared_error(y_a[sp_a:], pred_a))

rf_f = RandomForestRegressor(100, max_depth=8, random_state=42).fit(X_f[:sp_f], y_f[:sp_f])
pred_f = rf_f.predict(X_f[sp_f:]); r2_f = r2_score(y_f[sp_f:], pred_f); rmse_f = np.sqrt(mean_squared_error(y_f[sp_f:], pred_f))

lr = LinearRegression().fit(X_a[:sp_a], y_a[:sp_a])
pred_lr = lr.predict(X_a[sp_a:]); r2_lr = r2_score(y_a[sp_a:], pred_lr); rmse_lr = np.sqrt(mean_squared_error(y_a[sp_a:], pred_lr))

feat_names = [f"{c}_{s}" for c in ["F","T_top","L","B"] for s in ["mean","std","min","max","slope","range"]]
imp = sorted(zip(feat_names, rf_a.feature_importances_), key=lambda x:-x[1])

print(f"  -> RF + Aligned:  R2={r2_a*100:.1f}%, RMSE={rmse_a*100:.2f}")
print(f"  -> RF + Fixed:    R2={r2_f*100:.1f}%, RMSE={rmse_f*100:.2f}")
print(f"  -> Linear Reg:    R2={r2_lr*100:.1f}%, RMSE={rmse_lr*100:.2f}")

# ============================================================
# Step 4: Visualization
# ============================================================
print("\n[4/5] Generating report...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distillation Soft Sensor -- Preprocessing Analysis", fontsize=15, fontweight="bold")

ax = axes[0,0]; pn = min(300, N_POINTS); st = max(0, N_POINTS-pn)
ax.plot(t[st:], df.purity.values[st:], alpha=0.5, label="Raw (noisy)", lw=0.8)
ax.plot(t[st:], df_imp.purity.values[st:], label="Filtered", lw=1.5)
ax.set_title("Purity: Raw vs Filtered"); ax.set_xlabel("Time (min)"); ax.set_ylabel("Purity")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.scatter(y_a[sp_a:]*100, pred_a*100, alpha=0.5, s=15, label=f"Aligned (R2={r2_a*100:.1f}%)")
ax.scatter(y_f[sp_f:]*100, pred_f*100, alpha=0.5, s=15, label=f"Fixed (R2={r2_f*100:.1f}%)")
ax.plot([75,100],[75,100],"k--",alpha=0.3)
ax.set_title("Prediction vs Actual"); ax.set_xlabel("Actual Purity (%)"); ax.set_ylabel("Predicted Purity (%)")
ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(75,100); ax.set_ylim(75,100)

ax = axes[1,0]; top_n = min(10, len(imp))
ax.barh([x[0] for x in imp[:top_n]][::-1], [x[1] for x in imp[:top_n]][::-1], color="#1565c0")
ax.set_title("Feature Importance (Top 10)"); ax.set_xlabel("Importance"); ax.grid(alpha=0.3, axis="x")

ax = axes[1,1]
models = ["RF+Aligned", "RF+Fixed", "Linear"]; r2s = [r2_a*100, r2_f*100, r2_lr*100]
colors = ["#2e7d32","#e65100","#c62828"]
bars = ax.bar(models, r2s, color=colors, width=0.5)
for b, v in zip(bars, r2s): ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}%", ha="center")
ax.set_title("R2 Comparison"); ax.set_ylabel("R2 (%)"); ax.set_ylim(0,100); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, "report_python.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  -> Saved: {fig_path}")

# ============================================================
# Step 5: Summary
# ============================================================
print("\n[5/5] Pipeline complete!")
print("\n" + "="*50)
print("  Results Summary")
print("="*50)
print(f"\n  1. RF + Aligned:  R2={r2_a*100:.1f}%, RMSE={rmse_a*100:.2f}")
print(f"  2. RF + Fixed:    R2={r2_f*100:.1f}%, RMSE={rmse_f*100:.2f}")
print(f"  3. Linear Reg:    R2={r2_lr*100:.1f}%, RMSE={rmse_lr*100:.2f}")
print(f"\n  -> Time delay alignment ensures causal consistency")
print(f"  -> Feature importance guides process optimization")
print(f"  -> Directly maps to JD: noise filtering, delay compensation, feature engineering")
print("="*50)
