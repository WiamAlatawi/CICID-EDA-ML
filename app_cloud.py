import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')


os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CICIDS2017 – EDA Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark navy background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 600; color: #e6edf3; }
.metric-sub { font-size: 0.75rem; color: #58a6ff; margin-top: 2px; }

/* Insight cards */
.insight-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-card.danger { border-left-color: #f85149; }
.insight-card.success { border-left-color: #3fb950; }
.insight-card.warning { border-left-color: #d29922; }
.insight-title { font-weight: 600; font-size: 0.9rem; color: #e6edf3; margin-bottom: 4px; }
.insight-body { font-size: 0.82rem; color: #8b949e; line-height: 1.5; }

/* Section headers */
.section-header {
    font-size: 0.7rem; font-weight: 600; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 1px solid #30363d; padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
}

/* Page title */
.page-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin-bottom: 0; }
.page-sub { font-size: 0.85rem; color: #8b949e; }

/* Plotly chart bg */
.js-plotly-plot { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'normal': '#3fb950',
    'attack': '#f85149',
    'blue':   '#58a6ff',
    'purple': '#bc8cff',
    'yellow': '#d29922',
    'bg':     '#0d1117',
    'card':   '#161b22',
    'border': '#30363d',
    'text':   '#e6edf3',
    'muted':  '#8b949e',
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color=COLORS['text']),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d', zerolinecolor='#30363d'),
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import kagglehub
    path = kagglehub.dataset_download("elshewey/intrusion-detection-cicids2017")
    df = pd.read_csv(f'{path}/cicids2017_binary_balanced.csv')

    # ── Clean: Inf ──
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── Clean: Duplicates ──
    df.drop_duplicates(inplace=True)

    # ── Clean: Negative values ──
    cols_no_neg = [c for c in [
        'Flow Duration','Flow Bytes/s','Flow Packets/s',
        'Flow IAT Mean','Flow IAT Max','Flow IAT Min',
        'Fwd IAT Min','Fwd Header Length','Bwd Header Length','min_seg_size_forward'
    ] if c in df.columns]
    for col in cols_no_neg:
        df.loc[df[col] < 0, col] = np.nan
    df.fillna(df.median(numeric_only=True), inplace=True)

    # ── Feature Engineering ──
    df['fwd_bwd_ratio']      = df['Fwd Packets/s'] / (df['Fwd Packets/s'] + df['Bwd Packets/s'] + 1e-9)
    df['iat_cv']             = df['Flow IAT Std']  / (df['Flow IAT Mean'].abs() + 1e-9)
    df['Traffic_Intensity']  = df['Flow Bytes/s']  * df['Flow Packets/s']
    df['Active_Idle_Ratio']  = df['Active Mean']   / (df['Idle Mean'] + 1e-9)
    df['Packet_Variability'] = df['Packet Length Std'] / (df['Packet Length Mean'] + 1e-9)
    df['Header_Efficiency']  = df['Total Length of Fwd Packets'] / (df['Fwd Header Length'] + 1e-9)
    df['Flow_Activity']      = df['Flow Packets/s'] * df['Active Mean']
    df['Packet_Size_Range']  = df['Max Packet Length'] - df['Min Packet Length']
    df['Packet_Range_Ratio'] = (df['Max Packet Length'] - df['Min Packet Length']) / (df['Packet Length Mean'] + 1e-9)
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ CICIDS2017 EDA")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🎯 Target & Class Balance",
        "📈 Feature Distributions",
        "🔗 Correlations",
        "⚙️ Feature Engineering",
        "💡 Insights"
    ])
    st.markdown("---")
    st.markdown("<div style='color:#8b949e;font-size:0.75rem'>CIC-IDS-2017 · Binary Classification<br>Normal vs Attack Traffic</div>", unsafe_allow_html=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading & preprocessing dataset…"):
    df = load_data()

TARGET   = 'Attack_Binary'
num_cols = df.select_dtypes(include=np.number).columns.tolist()
feat_cols = [c for c in num_cols if c not in [TARGET]]
new_feats = ['fwd_bwd_ratio','iat_cv','Traffic_Intensity','Active_Idle_Ratio',
             'Packet_Variability','Header_Efficiency','Flow_Activity',
             'Packet_Size_Range','Packet_Range_Ratio']
orig_feats = [c for c in feat_cols if c not in new_feats]

df_normal = df[df[TARGET] == 0]
df_attack = df[df[TARGET] == 1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="page-title">🛡️ CICIDS2017 – Intrusion Detection EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Exploratory analysis of network traffic to distinguish normal activity from cyberattacks</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Total Records",    f"{len(df):,}",       "After deduplication"),
        (c2, "Features",         f"{len(orig_feats)}",  f"+ {len(new_feats)} engineered"),
        (c3, "Attack Traffic",   f"{(df[TARGET]==1).mean()*100:.1f}%", "of all records"),
        (c4, "Missing Values",   f"{df.isnull().sum().sum()}",  "After imputation"),
    ]
    for col, label, val, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">EDA Pipeline Summary</div>', unsafe_allow_html=True)

    steps = [
        ("✅", "Data Loading",       "Downloaded from Kaggle · cicids2017_binary_balanced.csv"),
        ("✅", "Quality Checks",      "No missing values · Duplicates removed · Inf replaced"),
        ("✅", "Negative Values",     "10 columns cleaned · Init_Win -1 preserved (meaningful signal)"),
        ("✅", "Feature Engineering", "9 new features derived from traffic behavior patterns"),
        ("✅", "Correlation Analysis","Top 15 features correlated with attack label identified"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="insight-card success">
            <div class="insight-title">{icon} {title}</div>
            <div class="insight-body">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – TARGET & CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Target & Class Balance":
    st.markdown('<div class="page-title">🎯 Target Variable Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Distribution of Attack_Binary across the dataset</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    dist     = df[TARGET].value_counts()
    dist_pct = df[TARGET].value_counts(normalize=True) * 100

    c1, c2 = st.columns(2)

    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=['Normal (0)', 'Attack (1)'],
            values=dist.values,
            hole=0.55,
            marker=dict(colors=[COLORS['normal'], COLORS['attack']],
                        line=dict(color=COLORS['bg'], width=3)),
            textinfo='label+percent',
            textfont=dict(size=13, color=COLORS['text'])
        ))
        fig_pie.update_layout(
            title="Class Distribution", **PLOTLY_LAYOUT,
            showlegend=False,
            annotations=[dict(text=f"{len(df):,}<br>records",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color=COLORS['text']))]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = go.Figure(go.Bar(
            x=['Normal (0)', 'Attack (1)'],
            y=dist.values,
            marker_color=[COLORS['normal'], COLORS['attack']],
            text=[f"{v:,}" for v in dist.values],
            textposition='outside',
            textfont=dict(color=COLORS['text'])
        ))
        fig_bar.update_layout(title="Record Count by Class", **PLOTLY_LAYOUT,
                               yaxis_title="Count", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">Class Balance Insight</div>', unsafe_allow_html=True)
    balance_ratio = dist.min() / dist.max()
    if balance_ratio > 0.8:
        st.markdown(f"""
        <div class="insight-card success">
            <div class="insight-title">✅ Dataset is Well-Balanced</div>
            <div class="insight-body">
                Normal: <strong>{dist_pct[0]:.1f}%</strong> · Attack: <strong>{dist_pct[1]:.1f}%</strong><br>
                Balance ratio {balance_ratio:.2f} – no need for SMOTE or resampling.
                Models can be trained without class-weight adjustments.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-card warning">
            <div class="insight-title">⚠️ Class Imbalance Detected</div>
            <div class="insight-body">Balance ratio {balance_ratio:.2f} – consider SMOTE or class weights.</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Distributions":
    st.markdown('<div class="page-title">📈 Feature Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare distributions of each feature between Normal and Attack traffic</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected = st.selectbox("Select a feature to inspect", orig_feats)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                      ('Attack',  COLORS['attack'],  df_attack)]:
            fig.add_trace(go.Histogram(
                x=subset[selected].clip(upper=subset[selected].quantile(0.99)),
                name=label, opacity=0.7,
                marker_color=color, nbinsx=50,
                histnorm='probability density'
            ))
        fig.update_layout(barmode='overlay', title=f"Distribution: {selected}", **PLOTLY_LAYOUT,
                          xaxis_title=selected, yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig_box = go.Figure()
        for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                      ('Attack',  COLORS['attack'],  df_attack)]:
            fig_box.add_trace(go.Box(
                y=subset[selected].clip(upper=subset[selected].quantile(0.99)),
                name=label, marker_color=color,
                line_color=color, boxmean=True
            ))
        fig_box.update_layout(title=f"Boxplot: {selected}", **PLOTLY_LAYOUT, yaxis_title=selected)
        st.plotly_chart(fig_box, use_container_width=True)

    # Stats table
    stats = pd.DataFrame({
        'Normal':  df_normal[selected].describe(),
        'Attack':  df_attack[selected].describe(),
    }).round(3)
    st.dataframe(stats, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Correlations":
    st.markdown('<div class="page-title">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How strongly each feature is linearly associated with attack detection</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    corr        = df[num_cols].corr()
    corr_target = corr[TARGET].drop(TARGET).sort_values()

    fig_corr = go.Figure(go.Bar(
        x=corr_target.values,
        y=corr_target.index,
        orientation='h',
        marker_color=[COLORS['attack'] if v > 0 else COLORS['normal'] for v in corr_target.values],
        text=[f"{v:.3f}" for v in corr_target.values],
        textposition='outside',
        textfont=dict(size=9, color=COLORS['muted'])
    ))
    fig_corr.add_vline(x=0, line_color=COLORS['border'], line_width=1)
    fig_corr.update_layout(
        title="Feature Correlation with Attack_Binary",
        height=max(500, len(corr_target)*16),
        **PLOTLY_LAYOUT,
        xaxis_title="Pearson Correlation",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Heatmap top 15
    st.markdown('<div class="section-header">Top 15 Features Heatmap</div>', unsafe_allow_html=True)
    top_feats = corr_target.abs().nlargest(15).index.tolist() + [TARGET]
    fig_heat = px.imshow(
        df[top_feats].corr(),
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.2f',
        aspect='auto'
    )
    fig_heat.update_layout(
        title="Heatmap – Top 15 Features + Target",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.markdown('<div class="page-title">⚙️ Engineered Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">9 derived features designed to capture attack behavior patterns</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    feat_meta = {
        'fwd_bwd_ratio':      ("Forward Traffic Ratio",     "Proportion of forward packets vs total – attacks often send more than they receive"),
        'iat_cv':             ("IAT Coefficient of Variation","Timing regularity – automated attacks send packets at unnaturally constant rates"),
        'Traffic_Intensity':  ("Traffic Intensity",          "Bytes/s × Packets/s – DDoS floods produce extreme combined values"),
        'Active_Idle_Ratio':  ("Active-to-Idle Ratio",       "Active time vs idle – persistent attacks stay active with minimal idle breaks"),
        'Packet_Variability': ("Packet Size Variability",     "Std ÷ Mean of packet length – attacks may use uniform or chaotic sizes"),
        'Header_Efficiency':  ("Header Efficiency",           "Payload vs header ratio – abnormal structures suggest crafted packets"),
        'Flow_Activity':      ("Flow Activity Score",         "Packets/s × Active time – combined aggression metric"),
        'Packet_Size_Range':  ("Packet Size Range",           "Max − Min packet size – spread reveals traffic heterogeneity"),
        'Packet_Range_Ratio': ("Packet Range Ratio",          "Size range relative to mean – normalized diversity of packet sizes"),
    }

    # Correlation of new features with target
    corr_new = df[new_feats + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)

    for feat in new_feats:
        if feat not in df.columns:
            continue
        title, desc = feat_meta.get(feat, (feat, ""))
        corr_val = corr_new.get(feat, 0)
        color_bar = COLORS['attack'] if corr_val > 0 else COLORS['normal']

        with st.expander(f"**{title}** (`{feat}`)  — corr: {corr_val:+.3f}"):
            st.markdown(f"<div style='color:{COLORS['muted']};font-size:0.85rem;margin-bottom:12px'>{desc}</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                              ('Attack',  COLORS['attack'],  df_attack)]:
                    clipped = subset[feat].clip(
                        upper=subset[feat].quantile(0.97),
                        lower=subset[feat].quantile(0.03)
                    )
                    fig.add_trace(go.Histogram(x=clipped, name=label, opacity=0.7,
                                               marker_color=color, nbinsx=40,
                                               histnorm='probability density'))
                fig.update_layout(barmode='overlay', title="Distribution",
                                   height=280, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                              ('Attack',  COLORS['attack'],  df_attack)]:
                    fig2.add_trace(go.Box(
                        y=subset[feat].clip(upper=subset[feat].quantile(0.97)),
                        name=label, marker_color=color, line_color=color
                    ))
                fig2.update_layout(title="Boxplot by Class", height=280, **PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            # mean comparison
            mn = df_normal[feat].mean()
            ma = df_attack[feat].mean()
            st.markdown(f"""
            <div style='display:flex;gap:2rem;font-family:JetBrains Mono,monospace;font-size:0.85rem'>
              <span style='color:{COLORS["normal"]}'>Normal mean: {mn:.4f}</span>
              <span style='color:{COLORS["attack"]}'>Attack mean: {ma:.4f}</span>
              <span style='color:{COLORS["muted"]}'>Δ = {abs(ma-mn):.4f}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.markdown('<div class="page-title">💡 Key Insights & Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Cross-feature patterns that separate attacks from normal traffic</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insight 1: Traffic Asymmetry ──────────────────────────────────────────
    st.markdown('<div class="section-header">1 · Traffic Direction Asymmetry</div>', unsafe_allow_html=True)

    fwd_n = df_normal['fwd_bwd_ratio'].mean()
    fwd_a = df_attack['fwd_bwd_ratio'].mean()

    fig = go.Figure()
    categories  = ['fwd_bwd_ratio', 'Traffic_Intensity', 'Flow_Activity']
    labels_nice = ['Fwd/Bwd Ratio', 'Traffic Intensity', 'Flow Activity']

    for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                  ('Attack',  COLORS['attack'],  df_attack)]:
        vals = [(subset[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-9) for c in categories]
        means = [v.mean() for v in vals]
        fig.add_trace(go.Bar(name=label, x=labels_nice, y=means, marker_color=color))

    fig.update_layout(barmode='group', title="Normalized Mean – Traffic Behavior Features",
                       **PLOTLY_LAYOUT, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card danger">
        <div class="insight-title">🔴 Attacks send disproportionately more forward traffic</div>
        <div class="insight-body">
            Average <code>fwd_bwd_ratio</code>: Normal = <strong>{fwd_n:.3f}</strong> · Attack = <strong>{fwd_a:.3f}</strong><br>
            This confirms that attack flows are heavily directional — the attacker pushes data or requests
            without waiting for proportional responses. Combined with high <code>Traffic_Intensity</code>,
            this is one of the clearest separating signals in the dataset.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Insight 2: Timing Regularity ─────────────────────────────────────────
    st.markdown('<div class="section-header">2 · Packet Timing: Humans vs Bots</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        iat_n = df_normal['iat_cv'].clip(0, df['iat_cv'].quantile(0.97))
        iat_a = df_attack['iat_cv'].clip(0, df['iat_cv'].quantile(0.97))
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=iat_n, name='Normal', opacity=0.7,
                                     marker_color=COLORS['normal'], nbinsx=50, histnorm='probability density'))
        fig2.add_trace(go.Histogram(x=iat_a, name='Attack', opacity=0.7,
                                     marker_color=COLORS['attack'], nbinsx=50, histnorm='probability density'))
        fig2.update_layout(barmode='overlay', title="IAT Coefficient of Variation",
                            **PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        iat_cv_n = df_normal['iat_cv'].median()
        iat_cv_a = df_attack['iat_cv'].median()
        st.markdown(f"""
        <div class="insight-card warning" style="margin-top:1.5rem">
            <div class="insight-title">⏱ Attack timing is robotic</div>
            <div class="insight-body">
                Median IAT-CV · Normal: <strong>{iat_cv_n:.3f}</strong> · Attack: <strong>{iat_cv_a:.3f}</strong><br><br>
                Low <code>iat_cv</code> = very regular packet intervals → automated tooling.<br>
                High <code>iat_cv</code> = bursty, irregular timing → some flooding attacks.<br><br>
                Either extreme is suspicious compared to human browsing traffic, which clusters in the mid-range.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Insight 3: Packet Size Patterns ──────────────────────────────────────
    st.markdown('<div class="section-header">3 · Packet Size Fingerprints</div>', unsafe_allow_html=True)

    size_feats = ['Packet_Size_Range', 'Packet_Variability', 'Packet_Range_Ratio']
    means_n = [df_normal[f].mean() for f in size_feats]
    means_a = [df_attack[f].mean()  for f in size_feats]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(r=means_n, theta=size_feats, fill='toself',
                                    name='Normal', line_color=COLORS['normal'], opacity=0.7))
    fig3.add_trace(go.Scatterpolar(r=means_a, theta=size_feats, fill='toself',
                                    name='Attack', line_color=COLORS['attack'], opacity=0.7))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor='#21262d'),
                   bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        title="Packet Size Feature Radar",
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📦 Attack packets have wider size variance</div>
        <div class="insight-body">
            Attacks show larger <code>Packet_Size_Range</code> and <code>Packet_Variability</code>,
            suggesting mixed-strategy attacks (small probe packets + large payload bursts).
            Normal traffic has more consistent packet sizing consistent with established TCP sessions.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Insight 4: Active vs Idle ─────────────────────────────────────────────
    st.markdown('<div class="section-header">4 · Connection Persistence: Active vs Idle</div>', unsafe_allow_html=True)

    fig4 = go.Figure()
    for label, color, subset in [('Normal', COLORS['normal'], df_normal),
                                   ('Attack',  COLORS['attack'],  df_attack)]:
        fig4.add_trace(go.Scatter(
            x=subset['Active Mean'].sample(min(500, len(subset)), random_state=42),
            y=subset['Idle Mean'].sample(min(500, len(subset)), random_state=42),
            mode='markers', name=label,
            marker=dict(color=color, size=4, opacity=0.5)
        ))
    fig4.update_layout(title="Active Mean vs Idle Mean (500 sample each class)",
                        **PLOTLY_LAYOUT, xaxis_title="Active Mean", yaxis_title="Idle Mean",
                        height=350)
    st.plotly_chart(fig4, use_container_width=True)

    ari_n = df_normal['Active_Idle_Ratio'].median()
    ari_a = df_attack['Active_Idle_Ratio'].median()
    st.markdown(f"""
    <div class="insight-card success">
        <div class="insight-title">🟢 Normal traffic takes breaks; attacks don't</div>
        <div class="insight-body">
            Median <code>Active_Idle_Ratio</code> · Normal: <strong>{ari_n:.2f}</strong> · Attack: <strong>{ari_a:.2f}</strong><br>
            Attack flows cluster in the high-active, low-idle corner — they maintain persistent
            pressure without natural pause. This combined with <code>Flow_Activity</code> creates
            a reliable "relentlessness" signal.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Insight 5: Header Efficiency ─────────────────────────────────────────
    st.markdown('<div class="section-header">5 · Header Efficiency – Payload Legitimacy</div>', unsafe_allow_html=True)

    he_n = df_normal['Header_Efficiency'].clip(0, df['Header_Efficiency'].quantile(0.97)).mean()
    he_a = df_attack['Header_Efficiency'].clip(0, df['Header_Efficiency'].quantile(0.97)).mean()

    fig5 = go.Figure()
    fig5.add_trace(go.Box(
        y=df_normal['Header_Efficiency'].clip(0, df['Header_Efficiency'].quantile(0.97)),
        name='Normal', marker_color=COLORS['normal']
    ))
    fig5.add_trace(go.Box(
        y=df_attack['Header_Efficiency'].clip(0, df['Header_Efficiency'].quantile(0.97)),
        name='Attack', marker_color=COLORS['attack']
    ))
    fig5.update_layout(title="Header Efficiency by Class",
                        **PLOTLY_LAYOUT, height=320, yaxis_title="Header Efficiency")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">📋 Attack packets carry more payload per header byte</div>
        <div class="insight-body">
            Mean <code>Header_Efficiency</code> · Normal: <strong>{he_n:.1f}</strong> · Attack: <strong>{he_a:.1f}</strong><br>
            Higher efficiency in attacks suggests bulk data transfer or crafted packets designed
            to maximize impact per connection. Combined with <code>Traffic_Intensity</code>,
            this characterizes volumetric attack types (e.g. DoS, DDoS).
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Combined Summary ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary – The Attack Fingerprint</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.4rem;">
    <table style="width:100%;border-collapse:collapse;font-size:0.83rem;color:#e6edf3">
    <tr style="border-bottom:1px solid #30363d">
        <th style="text-align:left;padding:6px 10px;color:#8b949e">Signal</th>
        <th style="text-align:center;padding:6px;color:#8b949e">Normal</th>
        <th style="text-align:center;padding:6px;color:#8b949e">Attack</th>
        <th style="text-align:left;padding:6px 10px;color:#8b949e">Interpretation</th>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
        <td style="padding:7px 10px">Forward Traffic Ratio</td>
        <td style="text-align:center;color:#3fb950">Balanced</td>
        <td style="text-align:center;color:#f85149">High →</td>
        <td style="padding:7px 10px;color:#8b949e">One-way bombardment</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
        <td style="padding:7px 10px">IAT Regularity</td>
        <td style="text-align:center;color:#3fb950">Variable</td>
        <td style="text-align:center;color:#f85149">Robotic</td>
        <td style="padding:7px 10px;color:#8b949e">Automated tooling</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
        <td style="padding:7px 10px">Traffic Intensity</td>
        <td style="text-align:center;color:#3fb950">Low</td>
        <td style="text-align:center;color:#f85149">Extreme</td>
        <td style="padding:7px 10px;color:#8b949e">Flooding pattern</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
        <td style="padding:7px 10px">Active / Idle</td>
        <td style="text-align:center;color:#3fb950">Intermittent</td>
        <td style="text-align:center;color:#f85149">Persistent</td>
        <td style="padding:7px 10px;color:#8b949e">No natural pauses</td>
    </tr>
    <tr>
        <td style="padding:7px 10px">Header Efficiency</td>
        <td style="text-align:center;color:#3fb950">Moderate</td>
        <td style="text-align:center;color:#f85149">High</td>
        <td style="padding:7px 10px;color:#8b949e">Bulk payload transfer</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
