import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from utils.nav import render_top_nav
from utils.icons import svg_icon, heading

# 日本語グリフ対応: CJKフォントを優先し、無ければ順にフォールバックする
# (Windows: Yu Gothic/Meiryo, Linux/Docker: fonts-noto-cjk 導入時の Noto Sans CJK JP)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP', 'Noto Sans JP', 'Yu Gothic', 'Meiryo', 'MS Gothic',
    'IPAexGothic', 'TakaoPGothic', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ================================================================
# SECTION 0: ページ設定（必ず最初のst.*コール）
# ================================================================
st.set_page_config(
    page_title="ミッション03: AI育成 | AI Inquiry Lab",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# SECTION 1: セキュリティガード
# ================================================================
if "user_role" not in st.session_state:
    st.session_state.user_role = "viewer"

if st.query_params.to_dict():
    st.error("不正なアクセスを検知しました。URLを直接書き換えないでください。")
    st.stop()

st.markdown("""
<script>
    if (window.top !== window.self) { window.top.location = window.self.location; }
    document.querySelectorAll('a').forEach(link => { link.setAttribute('rel', 'noopener noreferrer'); });
</script>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 2: 定数・パス設定
# ================================================================
SCRIPT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PARENT_DIR, "assets")
CSS_FILE   = os.path.join(ASSETS_DIR, "style.css")

DATASETS = {
    "ANDゲート":       "and",
    "ORゲート":        "or",
    "線形分類":        "linear",
    "XOR（不可能）":   "xor",
}

ACTIVATION_FUNS = {
    "シグモイド σ(z)": "sigmoid",
    "ReLU max(0,z)":  "relu",
    "tanh(z)":        "tanh",
}

# --- アチーブメント（実績バッジ）定義 -----------------------------------
# (key, 表示名, アイコン名, 説明) — 4つのマイルストーンで全タブをゲーム化する
BADGE_DEFS_badge3 = [
    ("neuron_fire",  "ニューロン発火", "zap",             "Tab①でニューロンを発火(出力>0.5)させた"),
    ("good_student", "優秀な生徒",     "graduation-cap",  "Tab②でXOR以外のデータに精度95%以上を達成"),
    ("deep_designer","深層の設計者",   "layers",          "Tab③で隠れ層を2層構成にした"),
    ("balance",      "バランス感覚",   "scale",           "Tab④で過学習スコアを1.0〜1.5(健全域)に収めた"),
]

# ================================================================
# SECTION 3: ユーティリティ関数
# ================================================================
@st.cache_data
def _read_css_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def load_css(path: str) -> None:
    if os.path.exists(path):
        st.markdown(f"<style>{_read_css_text(path)}</style>", unsafe_allow_html=True)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def activate(x: np.ndarray, fn: str) -> np.ndarray:
    return {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_act}[fn](x)

# ================================================================
# SECTION 4: シミュレーション関数群
# ================================================================
@st.cache_data
def generate_dataset(key: str, n: int = 120, seed: int = 42) -> tuple:
    np.random.seed(seed)
    if key == "and":
        base   = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        labels = np.array([0, 0, 0, 1], dtype=float)
        X = np.tile(base, (n // 4, 1)) + np.random.normal(0, 0.06, (n, 2))
        y = np.tile(labels, n // 4)
    elif key == "or":
        base   = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        labels = np.array([0, 1, 1, 1], dtype=float)
        X = np.tile(base, (n // 4, 1)) + np.random.normal(0, 0.06, (n, 2))
        y = np.tile(labels, n // 4)
    elif key == "linear":
        X = np.random.randn(n, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
    else:
        X = np.random.randn(n, 2) * 0.8
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    return X, y

@st.cache_data
def train_perceptron(X, y, lr, epochs, act_fn="sigmoid"):
    np.random.seed(0)
    w = np.random.randn(X.shape[1]) * 0.1
    b = 0.0
    losses, accuracies = [], []
    for _ in range(epochs):
        z    = X @ w + b
        pred = activate(z, act_fn)
        eps  = 1e-8
        loss = -np.mean(y * np.log(np.clip(pred, eps, 1)) + (1 - y) * np.log(np.clip(1 - pred, eps, 1)))
        losses.append(float(loss))
        dz = pred - y
        w -= lr * (X.T @ dz / len(y))
        b -= lr * float(np.mean(dz))
        accuracies.append(float(np.mean((pred > 0.5) == y)) * 100)
    return w, b, losses, accuracies

def plot_activation_curve(z_val: float, fn_name: str):
    x   = np.linspace(-6, 6, 300)
    y   = activate(x, fn_name)
    out = float(activate(np.array([z_val]), fn_name)[0])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.tick_params(colors="#000000")
    ax.plot(x, y, color="#0284c7", lw=2.5, label=fn_name)
    ax.axhline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.axvline(0, color="#cbd5e1", lw=0.8, ls="--")
    ax.scatter([z_val], [out], color="#b45309", s=120, zorder=5)
    ax.vlines(z_val, min(y.min(), out) - 0.05, out, color="#b45309", ls=":", alpha=0.7)
    ax.hlines(out, -6, z_val, color="#b45309", ls=":", alpha=0.7)
    ax.set_xlabel("z (weighted sum)", color="#000000", fontsize=9)
    ax.set_ylabel("Output", color="#000000", fontsize=9)
    ax.set_title(f"Activation: {fn_name}", color="#000000", fontsize=10)
    ax.legend(labelcolor="#000000", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)
    ax.grid(True, alpha=0.15, color="#cbd5e1")
    plt.tight_layout(pad=0.5)
    return fig, out

def plot_loss_acc(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("#ffffff")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors="#000000")
        for sp in ax.spines.values():
            sp.set_color("#cbd5e1")
        ax.grid(True, alpha=0.15, color="#cbd5e1")
    ep = np.arange(1, len(losses) + 1)
    ax1.plot(ep, losses, color="#c026d3", lw=2)
    ax1.set_title("Loss", color="#000000")
    ax1.set_xlabel("Epoch", color="#000000")
    ax1.set_ylabel("Loss", color="#000000")
    ax2.plot(ep, accuracies, color="#059669", lw=2)
    ax2.set_ylim(0, 105)
    ax2.set_title("Accuracy", color="#000000")
    ax2.set_xlabel("Epoch", color="#000000")
    ax2.set_ylabel("Accuracy (%)", color="#000000")
    plt.suptitle("Learning Curves", color="#b45309", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_decision_boundary(X, y, w, b, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.tick_params(colors="#000000")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    z_grid = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
    ax.contourf(xx, yy, z_grid, levels=50, cmap="RdYlGn", alpha=0.3)
    ax.contour(xx, yy, z_grid, levels=[0.5], colors="#b45309", linewidths=2)
    ax.scatter(X[y==1, 0], X[y==1, 1], c="#059669", s=30, label="Class 1", edgecolors="white", lw=0.4, zorder=3)
    ax.scatter(X[y==0, 0], X[y==0, 1], c="#c026d3", s=30, label="Class 0", edgecolors="white", lw=0.4, zorder=3)
    ax.legend(labelcolor="#000000", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)
    ax.set_title("Decision Boundary", color="#000000", fontsize=10)
    plt.tight_layout(pad=0.5)
    return fig

def plot_overfitting(degree, noise, seed=7):
    np.random.seed(seed)
    n = 18
    x_tr   = np.sort(np.random.uniform(-1, 1, n))
    y_tr   = np.sin(np.pi * x_tr) + np.random.normal(0, noise, n)
    x_te   = np.linspace(-1.3, 1.3, 200)
    y_true = np.sin(np.pi * x_te)
    degree = max(1, min(degree, 12))
    try:
        coeffs     = np.polyfit(x_tr, y_tr, degree)
        poly       = np.poly1d(coeffs)
        y_fit      = np.clip(poly(x_te), -5, 5)
        train_mse  = float(np.mean((y_tr - poly(x_tr)) ** 2))
        test_mse   = float(np.mean((y_true - poly(x_te)) ** 2))
    except Exception:
        y_fit = np.zeros_like(x_te)
        train_mse = test_mse = 0.0
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.tick_params(colors="#000000")
    ax.scatter(x_tr, y_tr, c="#059669", s=60, zorder=5, label="Training data", edgecolors="white", lw=0.5)
    ax.plot(x_te, y_true, color="#0284c7", lw=2, ls="--", label="True function")
    ax.plot(x_te, y_fit,  color="#c026d3", lw=2, label=f"Model (deg {degree})")
    ax.set_ylim(-4, 4)
    ax.set_xlim(-1.5, 1.5)
    ax.grid(True, alpha=0.15, color="#cbd5e1")
    ax.legend(labelcolor="#000000", facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)
    ax.set_title(f"Polynomial Fit (degree {degree})", color="#000000", fontsize=11)
    plt.tight_layout()
    return fig, train_mse, test_mse

def award_badge_badge3(key: str) -> bool:
    """バッジを付与し、今回初めて獲得した(未獲得→獲得)場合のみTrueを返す。"""
    badges = st.session_state.setdefault(
        "training_badges_badge3", {k: False for k, _, _, _ in BADGE_DEFS_badge3}
    )
    if not badges.get(key, False):
        badges[key] = True
        return True
    return False

def render_badge_shelf_badge3() -> None:
    """獲得状況を反映したバッジ棚（4チップ）を描画する。"""
    badges = st.session_state.get(
        "training_badges_badge3", {k: False for k, _, _, _ in BADGE_DEFS_badge3}
    )
    earned_n = sum(1 for v in badges.values() if v)
    chips = ""
    for key, label, icon_name, _desc in BADGE_DEFS_badge3:
        earned = badges.get(key, False)
        state_cls = "train-earned" if earned else "train-locked"
        icon_cls  = "icon-yellow"  if earned else "icon-dim"
        chips += (
            f'<div class="train-badge-chip {state_cls}" title="{_desc}">'
            f'{svg_icon(icon_name, size=18, css_class=icon_cls)}'
            f'<span class="train-badge-label">{label}</span></div>'
        )
    st.markdown(
        f'<div class="train-badge-shelf">{chips}</div>'
        f'<p class="train-badge-caption">達成バッジ '
        f'<b style="color:#b45309;">{earned_n}/4</b> — '
        f'4つのタブを操作してマイルストーンを集めよう</p>',
        unsafe_allow_html=True,
    )

def plot_nn_diagram(layer_sizes):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='datalim')
    n_layers    = len(layer_sizes)
    xs          = np.linspace(0.1, 0.9, n_layers)
    layer_cols  = ["#0284c7", "#059669", "#b45309", "#c026d3", "#0284c7"]
    all_pos     = []
    max_n       = max(layer_sizes)
    for li, (x, n) in enumerate(zip(xs, layer_sizes)):
        col  = layer_cols[li % len(layer_cols)]
        gap  = min(0.65 / max_n, 0.12)
        ys   = [0.5 + (j - (n - 1) / 2) * gap for j in range(n)]
        all_pos.append([(x, y) for y in ys])
        if li > 0:
            for px, py in all_pos[li - 1]:
                for cx, cy in all_pos[li]:
                    ax.plot([px, cx], [py, cy], color="#cbd5e1", lw=0.6, alpha=0.5, zorder=1)
        for nx, ny in all_pos[li]:
            circle = plt.Circle((nx, ny), 0.022, color=col, ec="white", lw=0.8, zorder=3)
            ax.add_patch(circle)
        lbl = "Input" if li == 0 else ("Output" if li == n_layers - 1 else f"Hidden {li}")
        ax.text(x, 0.07, lbl,        ha="center", color=col,       fontsize=8, fontweight="bold")
        ax.text(x, 0.02, f"({n})",   ha="center", color="#64748b", fontsize=7)
    ax.set_title("Network Architecture", color="#b45309", fontsize=13, pad=5)
    plt.tight_layout()
    return fig

# ================================================================
# SECTION 5: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

render_top_nav("training")

st.markdown(f"""
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("dna", size=30)} AI育成</h1>
    <p class="sub-title-text">MISSION 03 — AIを生み出し、鍛え、成長を見守る体験</p>
</div>
""", unsafe_allow_html=True)

# バッジ棚用のスコープ付きCSS（style.cssは他ワークストリームが編集中のため触らない）
st.markdown("""
<style>
.train-badge-shelf {
    display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 2px;
}
.train-badge-chip {
    display:inline-flex; align-items:center; gap:8px;
    padding:7px 14px; border-radius:999px;
    border:1px solid #cbd5e1; background:#ffffff;
    font-size:0.85rem; font-weight:600; color:#64748b;
}
.train-badge-chip.train-earned {
    border-color:#b45309; color:#b45309;
    box-shadow:0 0 10px rgba(255,215,0,0.25);
}
.train-badge-chip.train-locked { opacity:0.45; }
.train-badge-label { white-space:nowrap; }
.train-badge-caption {
    font-size:0.78rem; color:#64748b; margin:2px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 6: セッション状態の初期化
# ================================================================
if "train_result_3"   not in st.session_state: st.session_state["train_result_3"]   = None
if "best_accuracy_3"  not in st.session_state: st.session_state["best_accuracy_3"]  = 0.0
if "train_attempts_3" not in st.session_state: st.session_state["train_attempts_3"] = 0
if "training_badges_badge3" not in st.session_state:
    st.session_state["training_badges_badge3"] = {k: False for k, _, _, _ in BADGE_DEFS_badge3}

# ================================================================
# SECTION 7: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#b45309; font-weight:bold; font-size:0.9rem;">{svg_icon("dna", size=15, color="#b45309")} TRAINING MODE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(heading("学習パラメータ", "zap", level=3), unsafe_allow_html=True)
    lr_sidebar     = st.slider("学習率 (η)", 0.001, 1.0, 0.9, 0.005, format="%.3f", key="lr_sidebar")
    epochs_sidebar = st.slider("エポック数",  10,   500, 100, 10,               key="ep_sidebar")

    st.divider()
    st.markdown(heading("Navigation", "compass", level=3), unsafe_allow_html=True)
    st.page_link("main_app.py",                      label="司令室 (Home)")
    st.page_link("pages/1_vision.py",                label="ミッション01: AIの目")
    st.page_link("pages/2_adversarial.py",           label="ミッション02: AI騙し")
    st.page_link("pages/3_training.py",              label="ミッション03: AI育成")
    st.page_link("pages/4_llm_mechanism.py",         label="ミッション04: LLMの脳内")
    st.page_link("pages/5_cpu_gpu.py",               label="ミッション05: CPU対GPU")
    st.page_link("pages/6_physical_digital_ai.py",   label="ミッション06: AIの二形態")
    st.page_link("pages/7_rl_game.py",                label="ミッション07: 育成ゲーム")
    st.page_link("pages/8_machine_learning.py",      label="ミッション08: 機械学習の仕組み")
    st.divider()
    st.success("SHIELD: ONLINE")

# ================================================================
# SECTION 8: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{svg_icon("graduation-cap", size=20)} AIを「育てる」とはどういうことか？</h3>
ChatGPTも自動運転AIも、最初は何も知らない「白紙の状態」から始まります。<br>
大量のデータを与え、正解と間違いを繰り返し教えることで、徐々に賢くなっていく——<br>
これが<b>機械学習（Machine Learning）</b>の本質です。<br><br>
このミッションでは、あなた自身がAIの「先生」となり、
<b>ニューロンの誕生 → 学習の成功と失敗 → ネットワーク構築 → 過学習の克服</b>まで、
AI育成の全プロセスを手を動かしながら体験します。
</div>
""", unsafe_allow_html=True)

# ヒーロー直下の「バッジ棚」。ここでコンテナ位置を確保し、タブ実行後に最終状態で描画する
# （タブ操作で獲得したバッジも同一実行内で正しく反映される）。
badge_shelf_box_badge3 = st.container()

st.markdown(heading("STEP 2：タブを選んでAI育成の全プロセスを体験しよう！", "microscope", level=2),
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① ニューロンを育てる",
    "② ミニAIを鍛える",
    "③ ネットワークを組む",
    "④ 過学習の罠",
])

# ================================================================
# SECTION 9: タブコンテンツ
# ================================================================

# ---------------------------------------------------------------
# TAB 1: ニューロンの仕組み
# ---------------------------------------------------------------
with tab1:
    st.markdown(heading("AIの最小単位「ニューロン」を動かせ！", "zap", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_text1, col_text2 = st.columns([3, 2])

        with col_text1:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("brain", size=20)} ニューロンとは何か？</h3>
            人間の脳には約<b>860億個</b>の神経細胞（ニューロン）があります。
            AIのニューラルネットワークは、この生物ニューロンを<b>数式でモデル化</b>したものです。<br><br>
            <b>人工ニューロン（パーセプトロン）の動作：</b>
            <ol>
                <li>複数の「入力（x）」を受け取る</li>
                <li>それぞれに「重み（w）」を掛けて足し合わせる</li>
                <li>「バイアス（b）」を加える → これが <b>加重和 z</b></li>
                <li>「活性化関数」で 0〜1 の信号に変換する → これが <b>出力</b></li>
            </ol>
            <hr style="border-color:#cbd5e1; margin:10px 0;">
            <b style="color:#b45309;">{svg_icon("briefcase", size=16, color="#b45309")} 何に役立つ？</b><br>
            ニューロンは、スパムメールの自動判定・スマートフォンの顔認証・医療用のがん診断画像判定など、
            <b>あらゆるAIアプリケーションの基礎部品</b>です。
            ニューロンが集まることで、複雑な判断を行う知性が生まれます。<br><br>
            <b style="color:#059669;">{svg_icon("check-circle", size=16, color="#059669")} どうなっていると良い？</b><br>
            「重要な入力に大きな重み（w）・関係ない入力に小さな重み（w ≈ 0）」と
            適切に調整されている状態が理想です。
            学習前はランダムな値ですが、訓練を重ねるほど重みが最適化されていきます。<br><br>
            下のスライダーを動かして、ニューロンが「発火」する瞬間を体感しよう！
            </div>
            """, unsafe_allow_html=True)

        with col_text2:
            st.markdown(f"""
            <div style="background:#eef2fb; border:2px solid #b45309; padding:16px;
                        font-family:monospace; border-left:6px solid #0284c7;">
                <div style="color:#b45309; font-weight:bold; margin-bottom:10px;">{svg_icon("play", size=13, color="#b45309")} 数式</div>
                <div style="color:#0284c7; font-size:1.05rem;">z = x₁·w₁ + x₂·w₂ + b</div>
                <div style="color:#64748b; margin:6px 0;">（加重和）</div>
                <div style="color:#059669; font-size:1.05rem;">output = σ(z)</div>
                <div style="color:#64748b; margin:6px 0;">（活性化）</div>
                <br>
                <div style="color:#c026d3; font-size:0.8rem;">
                ● z が大きい → 強く発火 (→1)<br>
                ● z が小さい → 発火しない (→0)<br>
                ● z ≈ 0　　 → 境界（確率 50%）
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown(heading("インタラクティブ・ニューロン", "gamepad", level=3), unsafe_allow_html=True)

    n_col1, n_col2, n_col3 = st.columns([1, 1, 2])

    with n_col1:
        st.markdown(f'<b>{svg_icon("download", size=16)} 入力値</b>', unsafe_allow_html=True)
        x1 = st.slider("入力 x₁", -3.0, 3.0,  1.0, 0.1, key="n_x1")
        x2 = st.slider("入力 x₂", -3.0, 3.0,  0.5, 0.1, key="n_x2")

    with n_col2:
        st.markdown(f'<b>{svg_icon("scale", size=16)} 重み・バイアス</b>', unsafe_allow_html=True)
        w1   = st.slider("重み w₁",    -3.0, 3.0,  0.8,  0.1, key="n_w1")
        w2   = st.slider("重み w₂",    -3.0, 3.0,  0.6,  0.1, key="n_w2")
        bias = st.slider("バイアス b", -3.0, 3.0, -2.0,  0.1, key="n_bias")
        act_choice = st.selectbox("活性化関数", list(ACTIVATION_FUNS.keys()), key="n_act")
        act_fn     = ACTIVATION_FUNS[act_choice]

    z_val   = x1 * w1 + x2 * w2 + bias
    out_val = float(activate(np.array([z_val]), act_fn)[0])
    fired   = out_val > 0.5

    with n_col3:
        fig_act, _ = plot_activation_curve(z_val, act_fn)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_act)
        plt.close(fig_act)

    m1, m2, m3 = st.columns(3)
    m1.metric("加重和 z",    f"{z_val:.3f}")
    m2.metric("出力（確率）", f"{out_val:.3f}")
    m3.metric("ニューロン状態", "発火中！（→クラス1）" if fired else "静止（→クラス0）")

    st.markdown(f"""
    <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:14px;
                font-family:monospace; font-size:0.95rem; margin-top:8px;">
        <span style="color:#0284c7;">z</span> =
        (<span style="color:#000000;">{x1:.1f}</span> × <span style="color:#b45309;">{w1:.1f}</span>) +
        (<span style="color:#000000;">{x2:.1f}</span> × <span style="color:#b45309;">{w2:.1f}</span>) +
        <span style="color:#c026d3;">{bias:.1f}</span> =
        <span style="color:#059669; font-weight:bold;">{z_val:.3f}</span>
        &nbsp;&nbsp;→&nbsp;&nbsp;
        output = {act_choice}(<span style="color:#059669;">{z_val:.3f}</span>) =
        <span style="color:#b45309; font-weight:bold;">{out_val:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    if fired:
        st.success(f"このニューロンは「発火」しました！ 出力 {out_val:.3f} > 0.5 → クラス 1 として判定")
        # 【バッジ】ニューロン発火 — 初回獲得時のみ祝福
        if award_badge_badge3("neuron_fire"):
            st.success("バッジ獲得：「ニューロン発火」！ 初めてニューロンを発火させました。")
    else:
        st.info(f"このニューロンは「静止」しています。 出力 {out_val:.3f} ≤ 0.5 → クラス 0 として判定")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>{svg_icon("lightbulb", size=20)} 重みとバイアスの役割</h3>
    <ul>
    <li><b>重み（w）</b>：その入力が「どれくらい重要か」を決める係数。大きいほど影響力が強い。</li>
    <li><b>バイアス（b）</b>：「どれくらい活性化しやすいか」のオフセット。発火のしきい値を制御する。</li>
    <li><b>学習とは</b>：大量のデータを見ながら、w と b を少しずつ調整していくプロセスです。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 2: 学習シミュレーター
# ---------------------------------------------------------------
with tab2:
    st.markdown(heading("ミニAIを鍛えあげろ！", "dumbbell", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("book-open", size=20)} 機械学習の3ステップ</h3>
        AIが学習するプロセスは、3つのステップの繰り返しです：<br>
        <b>① 予測（Forward Pass）→ ② 誤差計算（Loss）→ ③ 重みの更新（Backward Pass）</b><br>
        これを何度もくり返すことで、AIは徐々に賢くなります。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">{svg_icon("briefcase", size=16, color="#b45309")} 何に役立つ？</b><br>
        この「予測 → 誤差計算 → 修正」という3ステップは、
        医療診断・商品レコメンド・機械翻訳・迷惑メール検出など、
        <b>世の中のほぼすべての機械学習の根幹</b>となっています。ChatGPT も同じ仕組みで訓練されました。<br><br>
        <b style="color:#059669;">{svg_icon("check-circle", size=16, color="#059669")} どうなっていると良い？</b><br>
        「Loss（誤差）グラフが滑らかに右下がり」になり、同時に「Accuracy（精度）が右上がり」になっている状態が理想です。
        グラフがギザギザと乱高下している場合は学習率が大きすぎるサインです。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("データセットとパラメータを設定しよう", "target", level=3), unsafe_allow_html=True)

    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)

    with cfg_c1:
        dataset_name = st.selectbox("データセット",    list(DATASETS.keys()), key="ds_select")
        dataset_key  = DATASETS[dataset_name]
        n_samples    = st.slider("サンプル数", 40, 200, 100, 20, key="n_samples_3")

    with cfg_c2:
        lr_t2     = st.slider("学習率 η",  0.001, 1.0, 0.9, 0.005, format="%.3f", key="lr_tab2")
        epochs_t2 = st.slider("エポック数", 10,   500, 100, 10,               key="ep_tab2")

    with cfg_c3:
        act_ch2 = st.selectbox("活性化関数", list(ACTIVATION_FUNS.keys()), key="act_tab2")
        act_fn2 = ACTIVATION_FUNS[act_ch2]
        if st.button("学習スタート！", use_container_width=True, key="train_btn"):
            X_tr, y_tr = generate_dataset(dataset_key, n_samples)
            w_f, b_f, losses, accs = train_perceptron(X_tr, y_tr, lr_t2, epochs_t2, act_fn2)
            st.session_state["train_result_3"] = {
                "X": X_tr, "y": y_tr, "w": w_f, "b": b_f,
                "losses": losses, "accuracies": accs,
                "dataset_name": dataset_name, "dataset_key": dataset_key,
            }
            best_acc = max(accs)
            if best_acc > st.session_state["best_accuracy_3"]:
                st.session_state["best_accuracy_3"] = best_acc
            st.session_state["train_attempts_3"] += 1

    if dataset_key == "xor":
        st.warning("XOR 問題は単層パーセプトロンでは解けません！どんな設定でも精度が 50% 付近に留まります。これが「ディープラーニング」が必要な理由です！")

    result = st.session_state.get("train_result_3")

    if result:
        st.markdown(heading("学習結果レポート", "bar-chart", level=3), unsafe_allow_html=True)

        final_loss = result["losses"][-1]
        final_acc  = result["accuracies"][-1]
        best_acc   = max(result["accuracies"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("最終 Loss",      f"{final_loss:.4f}")
        r2.metric("最終精度",        f"{final_acc:.1f}%")
        r3.metric("最高精度",        f"{best_acc:.1f}%")
        r4.metric("トレーニング回数", f"{st.session_state['train_attempts_3']} 回")

        res_c1, res_c2 = st.columns([3, 2])
        with res_c1:
            fig_lc = plot_loss_acc(result["losses"], result["accuracies"])
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_lc)
            plt.close(fig_lc)
        with res_c2:
            fig_db = plot_decision_boundary(result["X"], result["y"], result["w"], result["b"], result["dataset_name"])
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_db)
            plt.close(fig_db)

        w = result["w"]; b_v = result["b"]
        st.markdown(f"""
        <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px;
                    font-family:monospace; font-size:0.9rem;">
            <span style="color:#b45309;">最終的な重みパラメータ:</span><br>
            w₁ = <span style="color:#059669;">{w[0]:.4f}</span> &nbsp;|&nbsp;
            w₂ = <span style="color:#059669;">{w[1]:.4f}</span> &nbsp;|&nbsp;
            b  = <span style="color:#c026d3;">{b_v:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        if best_acc >= 95:
            st.success(f"エクセレント！ 精度 {best_acc:.1f}% 達成！AI の学習が成功しました！")
        elif best_acc >= 80:
            st.success(f"グッジョブ！ 精度 {best_acc:.1f}% — 学習率やエポック数を調整してさらに改善できるか試してみよう！")
        elif result["dataset_key"] != "xor":
            st.warning(f"精度 {best_acc:.1f}% — まだ改善の余地あり。学習率やエポック数を変えてみよう！")

        # 【バッジ】優秀な生徒 — XOR以外で最高精度95%以上
        if best_acc >= 95 and result["dataset_key"] != "xor" and award_badge_badge3("good_student"):
            st.success("バッジ獲得：「優秀な生徒」！ XOR以外のデータで95%以上の精度を達成しました。")

        st.info(f"今セッション最高精度：**{st.session_state['best_accuracy_3']:.1f}%**")

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("book", size=20)} 学習曲線の読み方</h3>
        <ul>
        <li><b>Loss が下がる</b>：AI が正解に近づいている証拠</li>
        <li><b>Accuracy が上がる</b>：正解できる割合が増えている</li>
        <li><b>学習率が大きすぎる</b>：Loss が乱高下して収束しない</li>
        <li><b>学習率が小さすぎる</b>：ゆっくり収束するが時間がかかる</li>
        </ul>
        最適な学習率を探す作業を「<b>ハイパーパラメータ調整</b>」と言います。これは AI エンジニアの核心スキルです！
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("「学習スタート！」ボタンを押して AI を鍛えよう！")

# ---------------------------------------------------------------
# TAB 3: ネットワーク構造ビジュアライザー
# ---------------------------------------------------------------
with tab3:
    st.markdown(heading("ニューラルネットワークを自分で組もう！", "network", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("layers", size=20)} 「深さ」が生み出す表現力</h3>
        単一ニューロンでは直線でしか分類できません（② のタブの XOR でそれを体験しましたね）。<br>
        ニューロンを複数の「層（レイヤー）」に積み重ねることで、
        AI は非常に複雑なパターンを学習できるようになります。<br>
        これが「<b>Deep Learning（深層学習）</b>」と呼ばれる理由です。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">{svg_icon("briefcase", size=16, color="#b45309")} 何に役立つ？</b><br>
        深層学習は現代AIの中核技術です。
        <b>ChatGPT</b>（文章生成）・<b>Stable Diffusion</b>（画像生成）・<b>AlphaFold</b>（タンパク質構造予測）・
        <b>自動運転</b>・<b>リアルタイム通訳</b>など、2020年代のAI革命はすべて深層学習が土台になっています。<br><br>
        <b style="color:#059669;">{svg_icon("check-circle", size=16, color="#059669")} どうなっていると良い？</b><br>
        各層が「異なる抽象レベル」の特徴を担当しているのが理想的な状態です。
        例えば画像認識では：浅い層→エッジや色 / 中間層→目・鼻・テクスチャ / 深い層→「これは犬だ」という概念。
        また、<b>訓練精度とテスト精度がどちらも高く、その差が小さい</b>（過学習していない）状態が健全です。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("アーキテクチャを設計しよう", "sliders", level=3), unsafe_allow_html=True)

    ctrl_col, graph_col = st.columns([2, 3])

    with ctrl_col:
        n_input   = st.slider("入力ノード数",     1, 8, 2, key="arch_in")
        n_hidden1 = st.slider("隠れ層1 ノード数", 1, 8, 4, key="arch_h1")
        n_hidden2 = st.slider("隠れ層2 ノード数", 0, 8, 3, key="arch_h2")
        n_output  = st.slider("出力ノード数",     1, 4, 1, key="arch_out")

        layer_sizes = [n_input, n_hidden1]
        if n_hidden2 > 0:
            layer_sizes.append(n_hidden2)
        layer_sizes.append(n_output)

        total_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("総レイヤー数", f"{len(layer_sizes)}")
        m2.metric("総ノード数",   f"{sum(layer_sizes)}")
        st.metric("学習可能パラメータ数", f"{total_params:,}")

    with graph_col:
        fig_nn = plot_nn_diagram(layer_sizes)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_nn)
        plt.close(fig_nn)

    # 【バッジ】深層の設計者 — 隠れ層を2層構成にした
    if n_hidden1 > 0 and n_hidden2 > 0 and award_badge_badge3("deep_designer"):
        st.success("バッジ獲得：「深層の設計者」！ 隠れ層2層のディープなネットワークを設計しました。")

    st.markdown(heading("有名なネットワークとの比較", "bar-chart", level=3), unsafe_allow_html=True)
    cmp_cols = st.columns(4)
    famous_nets = [
        ("あなたのネット",  sum(layer_sizes), f"{total_params:,}",    "#b45309"),
        ("AlexNet (2012)", 8,                "60,000,000",            "#059669"),
        ("GPT-3 (2020)",  96,                "175,000,000,000",       "#0284c7"),
        ("GPT-4 (2023)",  "???",             "〜1.7兆（推定）",        "#c026d3"),
    ]
    for col, (name, layers, params, color) in zip(cmp_cols, famous_nets):
        col.markdown(f"""
        <div style="background:#ffffff; border:2px solid {color}; border-left:6px solid {color};
                    padding:12px; text-align:center;">
            <div style="color:{color}; font-weight:bold; font-size:0.9rem;">{name}</div>
            <div style="color:#000000; font-size:0.8rem; margin-top:6px;">層数: {layers}</div>
            <div style="color:#64748b; font-size:0.75rem;">パラメータ: {params}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>{svg_icon("lightbulb", size=20)} なぜ「深い」ネットワークが強いのか？</h3>
    <ul>
    <li><b>層 1</b>：シンプルな特徴（エッジ・色の変化）を学ぶ</li>
    <li><b>層 2</b>：複合的な特徴（目・鼻・テクスチャ）を学ぶ</li>
    <li><b>層 3 以降</b>：抽象的な概念（「これは犬の顔だ」）を学ぶ</li>
    </ul>
    階層的に学ぶことで、少ないデータでも複雑なパターンを汎化できます。
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("フォワードパスを体験しよう", "sparkles", level=3), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.caption("入力値を設定すると、最初の隠れ層で値がどう変わるかを計算します（重みはランダム初期化の例）。")

        fp_cols_list = st.columns(min(n_input, 4))
        x_inputs = []
        for i, col in enumerate(fp_cols_list[:n_input]):
            val = col.slider(f"x{i+1}", -2.0, 2.0, float(i) * 0.5 - 0.5, 0.1, key=f"fp_x{i}")
            x_inputs.append(val)

        np.random.seed(42)
        x_vec = np.array(x_inputs)
        W1 = np.random.randn(n_hidden1, n_input) * 0.5
        h1 = np.tanh(W1 @ x_vec + np.zeros(n_hidden1))

        fp_c1, fp_c2 = st.columns(2)
        with fp_c1:
            in_str = " | ".join([f'<span style="color:#0284c7;">x{i+1}={v:.2f}</span>' for i, v in enumerate(x_inputs)])
            st.markdown(f"""
            <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px;
                        font-family:monospace; font-size:0.85rem;">
            <span style="color:#b45309;">入力層</span><br>{in_str}
            </div>
            """, unsafe_allow_html=True)
        with fp_c2:
            h1_str = " | ".join([f'<span style="color:#059669; font-weight:bold;">{v:.2f}</span>' for v in h1])
            st.markdown(f"""
            <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px;
                        font-family:monospace; font-size:0.85rem;">
            <span style="color:#b45309;">隠れ層1（tanh 活性化後）</span><br>{h1_str}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 過学習の罠
# ---------------------------------------------------------------
with tab4:
    st.markdown(heading("過学習の罠に落ちるな！", "alert-triangle", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        ov_text, ov_box = st.columns([3, 2])

        with ov_text:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("bug", size=20)} 過学習（Overfitting）とは？</h3>
            AI が「訓練データを丸暗記」してしまい、初めて見るデータには全く対応できなくなる現象です。<br><br>
            テスト前に過去問だけを暗記した学生が、少し違う問題が出ると解けなくなるのと同じです。<br><br>
            <b>過学習の症状：</b>
            <ul>
            <li>訓練データの精度：超高い（≒ 100%）</li>
            <li>テストデータの精度：低い（汎化できていない）</li>
            </ul>
            <hr style="border-color:#cbd5e1; margin:10px 0;">
            <b style="color:#b45309;">{svg_icon("briefcase", size=16, color="#b45309")} 何に役立つ？（なぜ学ぶのか？）</b><br>
            「過学習」を知っているエンジニアは、AIをリリースする前に必ずテストデータで検証し、問題を未然に防げます。
            現実では、訓練データだけで99%を出せても、本番環境（初めて見るデータ）では50%しか出ないAIが量産されています。
            過学習を理解することは、<b>実用的なAIを作る最重要スキル</b>の一つです。<br><br>
            <b style="color:#059669;">{svg_icon("check-circle", size=16, color="#059669")} どうなっていると良い？</b><br>
            <b>訓練 MSE とテスト MSE の比率（過学習スコア）が 1.0〜1.5 程度</b>であれば健全な状態です。
            比率が 2 を超えたら要注意、5 を超えたら重篤な過学習です。
            右のシミュレーターで次数を上げながら比率の変化を観察してみましょう。
            </div>
            """, unsafe_allow_html=True)

        with ov_box:
            st.markdown(f"""
            <div style="background:#eef2fb; border:2px solid #dc2626; padding:18px;
                        font-family:monospace; border-left:6px solid #b45309;">
                <div style="color:#b45309; font-weight:bold; margin-bottom:10px;">{svg_icon("play", size=13, color="#b45309")} 過学習の比較</div>
                <div style="color:#059669;">良いモデル</div>
                <div style="color:#000000; font-size:0.82rem;">訓練精度: 92% / テスト精度: 90%</div>
                <div style="color:#64748b; font-size:0.78rem; margin-bottom:10px;">→ バランスよく汎化できている</div>
                <div style="color:#dc2626;">過学習モデル</div>
                <div style="color:#000000; font-size:0.82rem;">訓練精度: 99% / テスト精度: 55%</div>
                <div style="color:#64748b; font-size:0.78rem;">→ 訓練データを丸暗記している</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("過学習シミュレーター", "microscope", level=3), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        ov_c1, ov_c2 = st.columns(2)
        with ov_c1:
            poly_degree = st.slider("モデルの複雑さ（多項式の次数）", 1, 12, 10, key="poly_deg")
            noise_level = st.slider("データのノイズ量",              0.05, 1.0, 0.3, 0.05, key="noise_lv")
        with ov_c2:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #cbd5e1; padding:14px; font-size:0.88rem;">
            <span style="color:#b45309;">{svg_icon("graduation-cap", size=15, color="#b45309")} 実験方法：</span>
            <ul style="color:#000000; margin:8px 0;">
            <li>次数 1〜2：直線/2次曲線。シンプルすぎる（<b>未学習</b>）</li>
            <li>次数 3〜5：適度な複雑さ。<b>ちょうど良い</b></li>
            <li>次数 8〜12：過度な複雑さ。訓練点を全部通ろうとして<b>過学習</b></li>
            </ul>
            ノイズを増やしながら次数を上げると、過学習がより顕著になります！
            </div>
            """, unsafe_allow_html=True)

        fig_ov, train_mse, test_mse = plot_overfitting(poly_degree, noise_level)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_ov)
        plt.close(fig_ov)

        ov_m1, ov_m2, ov_m3 = st.columns(3)
        ov_m1.metric("訓練 MSE（低いほど良）",  f"{train_mse:.4f}")
        ov_m2.metric("テスト MSE（低いほど良）", f"{test_mse:.4f}",
                     f"{test_mse - train_mse:+.4f}" if train_mse > 0 else "")
        overfit_ratio = test_mse / max(train_mse, 1e-6)
        ov_m3.metric("過学習スコア（1 に近いほど良）", f"{overfit_ratio:.2f}")

        if overfit_ratio > 5:
            st.error(f"重篤な過学習！ テスト MSE が訓練 MSE の {overfit_ratio:.1f} 倍です。次数を下げてください！")
        elif overfit_ratio > 2:
            st.warning(f"過学習の兆候。次数 {poly_degree} はこのデータには複雑すぎるかもしれません。")
        elif poly_degree <= 2 and test_mse > 0.3:
            st.warning("未学習（Underfitting）状態です。モデルが単純すぎて真の関数を捉えられていません。")
        else:
            st.success("バランスの良いフィット！ 訓練とテストの MSE が近い値です。")

        # 【バッジ】バランス感覚 — 過学習スコアが健全域(1.0〜1.5)
        if 1.0 <= overfit_ratio <= 1.5 and award_badge_badge3("balance"):
            st.success("バッジ獲得：「バランス感覚」！ 過学習も未学習もない健全なフィット(スコア1.0〜1.5)を達成しました。")

        st.markdown(f"""
        <div class="explanation-box" style="margin-top:16px;">
        <h3>{svg_icon("shield", size=20)} 過学習を防ぐ主な方法</h3>
        <table style="color:#000000; width:100%; font-size:0.88rem; border-collapse:collapse;">
        <tr style="border-bottom:1px solid #cbd5e1;">
            <th style="text-align:left; padding:6px; color:#b45309;">手法</th>
            <th style="text-align:left; padding:6px; color:#b45309;">概要</th>
        </tr>
        <tr style="border-bottom:1px solid #cbd5e1;">
            <td style="padding:6px; color:#059669;">{svg_icon("bar-chart", size=15, color="#059669")} データ拡張</td>
            <td style="padding:6px;">画像の反転・回転などで訓練データを人工的に増やす</td>
        </tr>
        <tr style="border-bottom:1px solid #cbd5e1;">
            <td style="padding:6px; color:#059669;">{svg_icon("droplet", size=15, color="#059669")} ドロップアウト</td>
            <td style="padding:6px;">学習中にランダムでニューロンを無効化し、依存関係を壊す</td>
        </tr>
        <tr style="border-bottom:1px solid #cbd5e1;">
            <td style="padding:6px; color:#059669;">{svg_icon("scale", size=15, color="#059669")} 正則化（L1/L2）</td>
            <td style="padding:6px;">重みが大きくなりすぎないようにペナルティを課す</td>
        </tr>
        <tr>
            <td style="padding:6px; color:#059669;">{svg_icon("stop-circle", size=15, color="#059669")} 早期終了</td>
            <td style="padding:6px;">検証データの精度が下がり始めたら学習を止める</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("学習のまとめ：AI 育成マスターへの道", "graduation-cap", level=3), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        summ_cols = st.columns(4)
        summ_cols[0].markdown(f"""
        <div style="background:#ffffff; border:2px solid #0284c7; border-left:6px solid #0284c7; padding:14px;">
            <div style="text-align:center;">{svg_icon("brain", size=30, color="#0284c7")}</div>
            <div style="color:#0284c7; font-weight:bold; text-align:center; margin:6px 0;">ニューロン</div>
            <div style="color:#000000; font-size:0.82rem; line-height:1.6;">
                AIの最小単位。重み×入力の合計を活性化関数に通して発火/静止を決める。
            </div>
            <div style="color:#b45309; font-size:0.78rem; margin-top:10px; font-weight:bold;">{svg_icon("briefcase", size=14, color="#b45309")} 役立つ場面</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                顔認証・スパム検出・医療診断など、すべてのAI技術の基礎部品。ニューロンが何億個も集まりChatGPTになる。
            </div>
            <div style="color:#059669; font-size:0.78rem; margin-top:8px; font-weight:bold;">{svg_icon("check-circle", size=14, color="#059669")} 良い状態とは？</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                重要な入力に大きな重み、関係ない入力に重み≈0が割り当てられている状態。
            </div>
        </div>
        """, unsafe_allow_html=True)
        summ_cols[1].markdown(f"""
        <div style="background:#ffffff; border:2px solid #059669; border-left:6px solid #059669; padding:14px;">
            <div style="text-align:center;">{svg_icon("trending-down", size=30, color="#059669")}</div>
            <div style="color:#059669; font-weight:bold; text-align:center; margin:6px 0;">勾配降下法</div>
            <div style="color:#000000; font-size:0.82rem; line-height:1.6;">
                損失関数の坂を下る方向に重みを少しずつ修正する最適化アルゴリズム。
            </div>
            <div style="color:#b45309; font-size:0.78rem; margin-top:10px; font-weight:bold;">{svg_icon("briefcase", size=14, color="#b45309")} 役立つ場面</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                ChatGPTから自動運転まで、あらゆる機械学習モデルの「学習」そのものを担う仕組み。これなしにAIは育たない。
            </div>
            <div style="color:#059669; font-size:0.78rem; margin-top:8px; font-weight:bold;">{svg_icon("check-circle", size=14, color="#059669")} 良い状態とは？</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                Lossグラフが滑らかに右下がり。ギザギザ→学習率が大きすぎ、平坦→小さすぎ。
            </div>
        </div>
        """, unsafe_allow_html=True)
        summ_cols[2].markdown(f"""
        <div style="background:#ffffff; border:2px solid #b45309; border-left:6px solid #b45309; padding:14px;">
            <div style="text-align:center;">{svg_icon("network", size=30, color="#b45309")}</div>
            <div style="color:#b45309; font-weight:bold; text-align:center; margin:6px 0;">深層学習</div>
            <div style="color:#000000; font-size:0.82rem; line-height:1.6;">
                多くの層を重ねることで非線形な複雑パターンを学習できるようになる。
            </div>
            <div style="color:#b45309; font-size:0.78rem; margin-top:10px; font-weight:bold;">{svg_icon("briefcase", size=14, color="#b45309")} 役立つ場面</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                ChatGPT・画像生成・自動翻訳・AlphaFold（創薬）など、2020年代のAI革命すべての土台。
            </div>
            <div style="color:#059669; font-size:0.78rem; margin-top:8px; font-weight:bold;">{svg_icon("check-circle", size=14, color="#059669")} 良い状態とは？</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                各層が異なる抽象度の特徴を担当し、訓練精度とテスト精度の両方が高い状態。
            </div>
        </div>
        """, unsafe_allow_html=True)
        summ_cols[3].markdown(f"""
        <div style="background:#ffffff; border:2px solid #c026d3; border-left:6px solid #c026d3; padding:14px;">
            <div style="text-align:center;">{svg_icon("alert-triangle", size=30, color="#c026d3")}</div>
            <div style="color:#c026d3; font-weight:bold; text-align:center; margin:6px 0;">過学習</div>
            <div style="color:#000000; font-size:0.82rem; line-height:1.6;">
                訓練データへの過剰適合。防ぐには正則化・ドロップアウト・データ拡張などが有効。
            </div>
            <div style="color:#b45309; font-size:0.78rem; margin-top:10px; font-weight:bold;">{svg_icon("briefcase", size=14, color="#b45309")} 役立つ場面</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                「過学習を知る」ことで本番リリース前に問題を発見できる。AIエンジニアの必須チェック項目。
            </div>
            <div style="color:#059669; font-size:0.78rem; margin-top:8px; font-weight:bold;">{svg_icon("check-circle", size=14, color="#059669")} 良い状態とは？</div>
            <div style="color:#64748b; font-size:0.78rem; line-height:1.5;">
                過学習スコア（テストMSE÷訓練MSE）が1.0〜1.5。5を超えると重篤な過学習。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # まとめ：ゲーム化されたマイルストーンと実務の関係（懐疑的な大人向けの学び）
        st.markdown(f"""
        <div class="explanation-box" style="margin-top:16px;">
        <h3>{svg_icon("check-circle", size=20)} なぜ「バッジ集め」が本物の学びなのか？</h3>
        このページ上部の4つのバッジは、ただのゲームではありません。
        実は、プロのMLエンジニアが<b>モデルを本番リリースする前に必ず確認するチェックリスト</b>とほぼ同じ構造になっています。<br><br>
        <ul>
        <li><b>{svg_icon("zap", size=15)} ニューロン発火</b> → トイケース（最小の例）で期待通り反応するか？（<i>スモークテスト</i>）</li>
        <li><b>{svg_icon("graduation-cap", size=15)} 優秀な生徒</b> → あらかじめ決めた<b>精度の合格ライン</b>（例: 95%）を超えているか？</li>
        <li><b>{svg_icon("layers", size=15)} 深層の設計者</b> → 問題の複雑さに見合うだけ<b>モデルが十分に深い</b>か？</li>
        <li><b>{svg_icon("scale", size=15)} バランス感覚</b> → 過学習していないか（<b>訓練とテストの差が小さい</b>か）？</li>
        </ul>
        「動く・基準を満たす・十分な表現力がある・汎化している」——この4点をチェックして初めて、AIは世に出せます。
        あなたが遊びながら集めた4つのバッジは、そのまま<b>信頼できるAIを見分ける眼</b>そのものなのです。
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# SECTION 9.5: バッジ棚の描画（タブ実行後の最終状態でヒーロー直下に反映）
# ================================================================
with badge_shelf_box_badge3:
    render_badge_shelf_badge3()

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。掌握する。育てる。</p>
    <p>MISSION 03: AI育成 — ニューロンの誕生から深層学習まで</p>
</div>
""", unsafe_allow_html=True)
