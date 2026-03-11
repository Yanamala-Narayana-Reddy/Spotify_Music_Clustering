"""
Spotify Music Clustering - Streamlit Web App
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Music Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #0d0d1a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 2px; }
    .insight-box {
        background: #080810;
        border-left: 3px solid #1db954;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────
PALETTE = ["#00f5d4", "#fc5c65", "#f7b731", "#a29bfe", "#fd9644", "#26de81", "#45aaf2", "#fd79a8"]
BG   = "#06060e"
BG2  = "#0d0d1a"
GRID = "#1e1e2e"
TCOL = "#e0e0e0"

# ═══════════════════════════════════════════════════════════════
# 1. GENERATE DATASET
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def generate_dataset(n_per_genre=100, seed=42):
    np.random.seed(seed)
    genres = [
        dict(name="EDM / Dance",         danceability=(0.78, 0.08), energy=(0.85, 0.08), tempo=(132, 8),  loudness=(-4, 2),  valence=(0.68, 0.10)),
        dict(name="Chill / Acoustic",    danceability=(0.45, 0.09), energy=(0.28, 0.09), tempo=(76, 12),  loudness=(-14, 4), valence=(0.50, 0.13)),
        dict(name="Hip-Hop / Rap",       danceability=(0.80, 0.07), energy=(0.65, 0.09), tempo=(94, 11),  loudness=(-6, 2),  valence=(0.44, 0.14)),
        dict(name="Classical / Ambient", danceability=(0.24, 0.09), energy=(0.21, 0.10), tempo=(86, 26),  loudness=(-22, 5), valence=(0.30, 0.11)),
        dict(name="Pop / Mainstream",    danceability=(0.70, 0.07), energy=(0.72, 0.08), tempo=(116, 9),  loudness=(-5, 2),  valence=(0.72, 0.11)),
    ]
    rows = []
    for g in genres:
        N = n_per_genre
        def s(p): return np.random.normal(p[0], p[1], N)
        rows.append(pd.DataFrame({
            "danceability": np.clip(s(g["danceability"]), 0, 1),
            "energy":       np.clip(s(g["energy"]),       0, 1),
            "tempo":        np.clip(s(g["tempo"]),       40, 220),
            "loudness":     np.clip(s(g["loudness"]),   -40, 0),
            "valence":      np.clip(s(g["valence"]),     0, 1),
            "true_genre":   g["name"],
        }))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════════
# 2. RUN ML PIPELINE
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def run_pipeline(n_per_genre, k):
    df = generate_dataset(n_per_genre)
    features = ["danceability", "energy", "tempo", "loudness", "valence"]

    # Normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])

    # Elbow + Silhouette
    K_range = list(range(2, 11))
    inertias, silhouettes, db_scores = [], [], []
    for ki in K_range:
        km = KMeans(n_clusters=ki, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lbl))
        db_scores.append(davies_bouldin_score(X, lbl))

    # Final KMeans
    km_final = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    df["cluster"] = km_final.fit_predict(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    df["tsne1"] = X_tsne[:, 0]
    df["tsne2"] = X_tsne[:, 1]

    # Centroids in original scale
    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_),
        columns=features
    )
    C_pca = pca.transform(km_final.cluster_centers_)

    metrics = {
        "inertia":    round(km_final.inertia_, 2),
        "silhouette": round(silhouette_score(X, df["cluster"]), 4),
        "db":         round(davies_bouldin_score(X, df["cluster"]), 4),
        "iters":      km_final.n_iter_,
        "pca_var":    round(pca.explained_variance_ratio_.sum() * 100, 1),
        "pca_var1":   round(pca.explained_variance_ratio_[0] * 100, 1),
        "pca_var2":   round(pca.explained_variance_ratio_[1] * 100, 1),
    }

    return df, centroids_orig, C_pca, K_range, inertias, silhouettes, db_scores, metrics

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎵 Spotify Clustering")
    st.markdown("---")
    st.markdown("### Parameters")
    n_per_genre = st.slider("Tracks per genre", 50, 300, 100, 10)
    k = st.slider("Number of clusters (K)", 2, 8, 5)
    viz_mode = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)
    color_by = st.radio("Color by", ["KMeans Cluster", "True Genre"], horizontal=True)
    st.markdown("---")
    st.markdown("### Audio Features")
    for f, icon in [("Danceability","💃"), ("Energy","⚡"), ("Tempo","🥁"), ("Loudness","🔊"), ("Valence","😊")]:
        st.markdown(f"{icon} **{f}**")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("[Kaggle - Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🎵 Spotify Music Clustering")
st.markdown("**Unsupervised ML · KMeans · PCA · t-SNE · scikit-learn**")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════
with st.spinner("Running ML pipeline... please wait"):
    df, centroids_orig, C_pca, K_range, inertias, silhouettes, db_scores, metrics = run_pipeline(n_per_genre, k)

cluster_colors = [PALETTE[i % len(PALETTE)] for i in range(k)]
total_tracks = len(df)

# ═══════════════════════════════════════════════════════════════
# METRICS ROW
# ═══════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, color in [
    (c1, total_tracks,            "Total Tracks",  "#00f5d4"),
    (c2, k,                       "Clusters",      "#fc5c65"),
    (c3, metrics["iters"],        "KMeans Iters",  "#f7b731"),
    (c4, metrics["silhouette"],   "Silhouette",    "#a29bfe"),
    (c5, f"{metrics['pca_var']}%","PCA Variance",  "#1db954"),
]:
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='color:{color};'>{val}</div>
      <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PLOT SETTINGS
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG2,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TCOL,
    "xtick.color":      "#666",
    "ytick.color":      "#666",
    "text.color":       TCOL,
    "grid.color":       GRID,
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
})

# ═══════════════════════════════════════════════════════════════
# ROW 1: Elbow + Silhouette
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Elbow Curve")
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
    ax.plot(K_range, inertias, color="#1db954", lw=2, marker="o", markersize=6, markerfacecolor="#fff")
    ax.axvline(k, color="#fc5c65", lw=1.5, linestyle="--", label=f"K={k}")
    ax.fill_between(K_range, inertias, alpha=0.08, color="#1db954")
    ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TCOL)
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### Silhouette & Davies-Bouldin")
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG)
    ax2b = ax.twinx()
    ax.plot(K_range, silhouettes, color="#00f5d4", lw=2, marker="s", markersize=5, label="Silhouette")
    ax2b.plot(K_range, db_scores, color="#fc5c65", lw=2, marker="^", markersize=5, linestyle="--", label="Davies-Bouldin")
    ax.axvline(k, color="#f7b731", lw=1.5, linestyle=":", label=f"K={k}")
    ax.set_ylabel("Silhouette", color="#00f5d4", fontsize=8)
    ax2b.set_ylabel("Davies-Bouldin", color="#fc5c65", fontsize=8)
    ax2b.tick_params(axis="y", colors="#fc5c65")
    ax2b.set_facecolor(BG2)
    for sp in ax2b.spines.values():
        sp.set_edgecolor(GRID)
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=7, facecolor=BG2, edgecolor=GRID, labelcolor=TCOL)
    ax.set_xlabel("K")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# ROW 2: PCA / t-SNE Scatter
# ═══════════════════════════════════════════════════════════════
st.markdown(f"#### {viz_mode} Projection — {color_by}")

fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)

if viz_mode == "PCA":
    xc, yc = "pca1", "pca2"
    xlabel = f"PC1 ({metrics['pca_var1']}%)"
    ylabel = f"PC2 ({metrics['pca_var2']}%)"
else:
    xc, yc = "tsne1", "tsne2"
    xlabel, ylabel = "t-SNE 1", "t-SNE 2"

genre_palette = {
    "EDM / Dance":         "#00f5d4",
    "Chill / Acoustic":    "#f7b731",
    "Hip-Hop / Rap":       "#fc5c65",
    "Classical / Ambient": "#a29bfe",
    "Pop / Mainstream":    "#fd9644"
}

if color_by == "KMeans Cluster":
    for ci in range(k):
        mask = df["cluster"] == ci
        ax.scatter(df.loc[mask, xc], df.loc[mask, yc],
                   c=cluster_colors[ci], s=25, alpha=0.75, edgecolors="none",
                   label=f"Cluster {ci+1} (n={mask.sum()})")
    if viz_mode == "PCA":
        ax.scatter(C_pca[:, 0], C_pca[:, 1], marker="X", s=200, c="white",
                   edgecolors="#333", zorder=6, linewidths=1.2, label="Centroid")
else:
    for gname, gcol in genre_palette.items():
        mask = df["true_genre"] == gname
        ax.scatter(df.loc[mask, xc], df.loc[mask, yc],
                   c=gcol, s=25, alpha=0.75, edgecolors="none", label=gname)

ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TCOL, markerscale=1.4, ncol=2)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.grid(True, alpha=0.3)
ax.set_facecolor(BG2)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ═══════════════════════════════════════════════════════════════
# CLUSTER PROFILES
# ═══════════════════════════════════════════════════════════════
st.markdown("#### Cluster Profiles")
cols = st.columns(k)

for ci in range(k):
    with cols[ci]:
        color = cluster_colors[ci]
        count = int((df["cluster"] == ci).sum())
        st.markdown(f"<div style='border:1px solid {color}55;border-radius:10px;padding:12px;background:#080810;'>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{color};font-weight:700;font-size:1rem;'>Cluster {ci+1}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#888;font-size:0.75rem;margin-bottom:8px;'>{count} tracks</div>", unsafe_allow_html=True)
        for feat in ["danceability", "energy", "valence"]:
            val = float(centroids_orig.loc[ci, feat])
            st.markdown(f"""
            <div style='margin-top:6px;'>
              <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#888;'>
                <span>{feat.capitalize()}</span>
                <span style='color:{color};'>{val:.2f}</span>
              </div>
              <div style='height:5px;background:#1e1e2e;border-radius:3px;margin-top:3px;'>
                <div style='height:100%;width:{val*100:.0f}%;background:{color};border-radius:3px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        tempo_n = float((centroids_orig.loc[ci, "tempo"] - 40) / 180)
        loud_n  = float((centroids_orig.loc[ci, "loudness"] + 40) / 40)
        for feat, val in [("Tempo", tempo_n), ("Loudness", loud_n)]:
            st.markdown(f"""
            <div style='margin-top:6px;'>
              <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#888;'>
                <span>{feat}</span>
                <span style='color:{color};'>{val:.2f}</span>
              </div>
              <div style='height:5px;background:#1e1e2e;border-radius:3px;margin-top:3px;'>
                <div style='height:100%;width:{min(val*100,100):.0f}%;background:{color};border-radius:3px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TEMPO vs LOUDNESS
# ═══════════════════════════════════════════════════════════════
st.markdown("#### Tempo vs Loudness by Cluster")
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
for ci in range(k):
    mask = df["cluster"] == ci
    ax.scatter(df.loc[mask, "tempo"], df.loc[mask, "loudness"],
               c=cluster_colors[ci], s=20, alpha=0.6, edgecolors="none", label=f"C{ci+1}")
    ax.scatter(float(centroids_orig.loc[ci, "tempo"]), float(centroids_orig.loc[ci, "loudness"]),
               marker="X", s=180, c=cluster_colors[ci], edgecolors="white", linewidths=1, zorder=5)
ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor=TCOL, ncol=k)
ax.set_xlabel("Tempo (BPM)")
ax.set_ylabel("Loudness (dB)")
ax.grid(True, alpha=0.3)
ax.set_facecolor(BG2)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ═══════════════════════════════════════════════════════════════
# INSIGHT BOX
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='insight-box'>
  <div style='color:#1db954;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>
    Insight
  </div>
  <div style='color:#aaa;font-size:0.9rem;line-height:1.7;'>
    KMeans identified <strong style='color:#1db954;'>{k} distinct clusters</strong> across {total_tracks} tracks
    using 5 audio features. The model converged in
    <strong style='color:#f7b731;'>{metrics['iters']} iterations</strong> with a silhouette score of
    <strong style='color:#00f5d4;'>{metrics['silhouette']}</strong>.
    PCA explains <strong style='color:#a29bfe;'>{metrics['pca_var']}%</strong> of total variance in 2 components.
    High-energy + high-danceability clusters capture EDM and Pop,
    while low-energy clusters represent Classical and Acoustic music.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.75rem;'>"
    "Built with scikit-learn · pandas · matplotlib · Streamlit | Grow with Gyan"
    "</div>",
    unsafe_allow_html=True
)
