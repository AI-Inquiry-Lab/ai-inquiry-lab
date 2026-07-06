import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64
import matplotlib
import matplotlib.pyplot as plt
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

try:
    import torch
    import torchvision.models as tv_models
    from torchvision.models import MobileNet_V2_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ================================================================
# SECTION 0: ページ設定（必ず最初のst.*コール）
# ================================================================
st.set_page_config(
    page_title="ミッション02: AI騙し | AI Inquiry Lab",
    page_icon=":material/theater_comedy:",
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
DATA_DIR   = os.path.join(PARENT_DIR, "data")
ASSETS_DIR = os.path.join(PARENT_DIR, "assets")
CSS_FILE   = os.path.join(ASSETS_DIR, "style.css")


ATTACK_INFO = {
    "ガウスノイズ": {
        "key": "gaussian",
        "icon": "wind",
        "color": "#0284c7",
        "desc": "ランダムなノイズを全ピクセルに加算。自然界の撮影ノイズに似た形状で、シンプルかつ汎用的な攻撃手法です。",
    },
    "塩コショウ": {
        "key": "salt_pepper",
        "icon": "sparkles",
        "color": "#c026d3",
        "desc": "ランダムに白（塩）と黒（胡椒）の点を散りばめます。デジタル信号の欠損ノイズを再現した古典的な手法です。",
    },
    "疑似FGSM": {
        "key": "pseudo_fgsm",
        "icon": "zap",
        "color": "#b45309",
        "desc": "画像の「エッジ（境目）」の方向に沿った構造的なノイズ。本物のFGSMが利用する勾配方向を近似した、最も賢い攻撃です。",
    },
    "カラーシフト": {
        "key": "color_shift",
        "icon": "image",
        "color": "#059669",
        "desc": "赤・緑・青の各色成分を微妙にずらします。人間の目には色が同じに見えても、AIの数値判断を混乱させます。",
    },
}

# 攻撃キー → 表示情報（リーダーボード用の逆引き）
ATTACK_BY_KEY = {info["key"]: {"name": name, **info} for name, info in ATTACK_INFO.items()}

# 攻撃対象に選べるベース画像（DATA_DIR に存在するもののみ後段でフィルタ）
ADV_BASE_IMAGES = {
    "bird.jpg":       "鳥",
    "cat.jpg":        "猫",
    "building.jpg":   "建物",
    "mountain.jpg":   "山",
    "soccerball.jpg": "サッカーボール",
    "hyousiki.jpg":   "道路標識",
}

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
    else:
        st.warning(f"CSSファイルが見つかりません: {path}")

def get_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ================================================================
# SECTION 4: AIモデル（推論エンジン）
# ================================================================
if TORCH_AVAILABLE:
    @st.cache_resource
    def load_model():
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = tv_models.mobilenet_v2(weights=weights)
        model.eval()
        return model, weights.meta["categories"], weights.transforms()
    _model, _categories, _preprocess = load_model()
else:
    _model, _categories, _preprocess = None, [], None

@st.cache_data(show_spinner=False)
def predict_image(img_array: np.ndarray) -> tuple:
    """画像をAIに推論させ (ラベル, 確信度, Top5リスト) を返す。
    同じ画像に対する推論結果はキャッシュし、無関係な操作での再実行のたびに
    重いモデル推論をやり直さないようにする（_model/_preprocessはロード後不変）。"""
    if not TORCH_AVAILABLE or _model is None:
        return "AI利用不可", 0.0, []
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    input_tensor = _preprocess(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = _model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_probs, top5_idx = torch.topk(probs, 5)
    top_label = _categories[top5_idx[0].item()]
    top_conf  = float(top5_probs[0].item() * 100)
    top5 = [
        (_categories[top5_idx[i].item()], float(top5_probs[i].item() * 100))
        for i in range(5)
    ]
    return top_label, top_conf, top5

# ================================================================
# SECTION 5: 画像攻撃関数群
# ================================================================
def attack_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def attack_salt_pepper(img: np.ndarray, density: float) -> np.ndarray:
    result = img.copy()
    h, w = img.shape[:2]
    mask_salt   = np.random.random((h, w)) < density / 2
    mask_pepper = np.random.random((h, w)) < density / 2
    result[mask_salt]   = 255
    result[mask_pepper] = 0
    return result

def attack_pseudo_fgsm(img: np.ndarray, epsilon: float) -> np.ndarray:
    """FGSMを近似した、エッジ勾配方向への構造的ノイズ"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_sign = np.sign(gx + gy)
    perturbed = np.stack([
        img[:, :, 0].astype(np.float32) + epsilon * gradient_sign,
        img[:, :, 1].astype(np.float32) + epsilon * gradient_sign * 0.85,
        img[:, :, 2].astype(np.float32) + epsilon * gradient_sign * 1.1,
    ], axis=2)
    return np.clip(perturbed, 0, 255).astype(np.uint8)

def attack_color_shift(img: np.ndarray, strength: float) -> np.ndarray:
    """色チャンネルをランダム方向にシフトして混乱させる"""
    result = img.astype(np.float32).copy()
    shifts = np.random.choice([-strength, strength], size=(3,))
    for ch, shift in enumerate(shifts):
        result[:, :, ch] = np.clip(result[:, :, ch] + shift, 0, 255)
    return result.astype(np.uint8)

def apply_attack(img: np.ndarray, key: str, strength: float) -> np.ndarray:
    if key == "gaussian":
        return attack_gaussian(img, sigma=strength * 1.5)
    elif key == "salt_pepper":
        return attack_salt_pepper(img, density=strength / 500)
    elif key == "pseudo_fgsm":
        return attack_pseudo_fgsm(img, epsilon=strength)
    elif key == "color_shift":
        return attack_color_shift(img, strength=strength)
    return img

def compute_noise_visibility(original: np.ndarray, perturbed: np.ndarray) -> float:
    diff = np.abs(original.astype(np.float32) - perturbed.astype(np.float32))
    return float(np.mean(diff) / 255 * 100)

def judge_deception(
    orig_conf: float, pert_conf: float,
    orig_label: str, pert_label: str
) -> tuple:
    """騙しの成否を判定してメッセージ・カラーを返す"""
    label_changed = orig_label != pert_label
    conf_drop     = orig_conf - pert_conf

    if label_changed and conf_drop > 30:
        return (svg_icon("sparkles", size=22, color="#059669") +
                " 大成功！AIを完全に騙した！"), "#059669"
    elif label_changed:
        return (svg_icon("check-circle", size=22, color="#059669") +
                " 成功！AIが別の物を見ている！"), "#059669"
    elif conf_drop > 20:
        return (svg_icon("alert-triangle", size=22, color="#b45309") +
                " 部分成功！AIが迷い始めている！"), "#b45309"
    elif conf_drop > 5:
        return (svg_icon("refresh-cw", size=22, color="#0284c7") +
                " 惜しい！AIが揺れている"), "#0284c7"
    else:
        return (svg_icon("shield", size=22, color="#dc2626") +
                " 失敗…AIは騙せなかった"), "#dc2626"

def draw_confidence_bar(top5: list, highlight_label: str, bar_color: str) -> plt.Figure:
    """Top-5確信度の横棒グラフを生成して返す"""
    labels = [f"{l[:18]}..." if len(l) > 18 else l for l, _ in top5][::-1]
    values = [v for _, v in top5][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(labels, values, color=[
        "#c026d3" if (labels[::-1][i] == highlight_label or
                      (labels[::-1][i] + "...") == highlight_label[:18] + "...")
        else bar_color
        for i in range(len(labels))
    ][::-1])
    ax.set_xlim(0, 100)
    ax.tick_params(colors="#000000", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#000000", fontsize=7)
    plt.tight_layout(pad=0.5)
    return fig

# ================================================================
# SECTION 6: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

render_top_nav("adversarial")

st.markdown(f"""
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("mask", size=30, color="#c026d3")} AI騙し</h1>
    <p class="sub-title-text">MISSION 02 — 微小な「嘘」でAIの認識を崩壊させろ</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 7: セッション状態の初期化
# ================================================================
if "best_score_2" not in st.session_state:
    st.session_state["best_score_2"] = 0

# 攻撃タイプ別リーダーボード（このページ専用。キーは _adv2 で衝突回避）
if "adv_leaderboard" not in st.session_state:
    # 例: {"gaussian": {"score": 120, "conf_drop": 30.1, "visibility": 0.42}}
    st.session_state["adv_leaderboard"] = {}

if "adv_full_clear_done_adv2" not in st.session_state:
    st.session_state["adv_full_clear_done_adv2"] = False

# ================================================================
# SECTION 8: サイドバー（攻撃設定センター）
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#c026d3; font-weight:bold; font-size:0.9rem;">
            {svg_icon("mask", size=15, color="#c026d3")} AI HACKING MODE
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_attack_name = st.selectbox(
        "攻撃タイプを選択",
        list(ATTACK_INFO.keys()),
        key="attack_select"
    )
    selected_attack_key  = ATTACK_INFO[selected_attack_name]["key"]

    attack_strength = st.slider("攻撃強度", min_value=1, max_value=100, value=20)

    st.divider()
    st.info(
        f"**{selected_attack_name}**\n\n{ATTACK_INFO[selected_attack_name]['desc']}",
    )

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
# SECTION 9: 画像ロード（固定: 鳥の画像を使用）
# ================================================================
bird_path = os.path.join(DATA_DIR, "bird.jpg")
target_image = Image.open(bird_path).convert("RGB")

# ================================================================
# SECTION 10: メインコンテンツ（4タブ構成）
# ================================================================
st.markdown(heading("ミッション開始", "flask", level=2), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① AIの弱点を知る",
    "② 騙しシミュレーター",
    "③ 攻撃の仕組み",
    "④ 防御と未来",
])

# ---------------------------------------------------------------
# TAB 1: AIの弱点を知る
# ---------------------------------------------------------------
with tab1:
    st.markdown(heading("なぜAIは騙されるのか？", "brain", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        col_text, col_box = st.columns([3, 2])
        with col_text:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("mask", color="#0284c7")} AIと人間の「見え方」の根本的な違い</h3>
            人間は物体の<b>「形（シルエット）」</b>を見て直感的に判断します。
            パンダを見るとき、黒と白のパターン・丸い体形・耳の形を無意識に総合して判断しています。<br><br>
            しかし現代のAI（畳み込みニューラルネットワーク）は、<b>「テクスチャ（模様・質感）」</b>に
            強く依存して判断する傾向があります。

            <h3>{svg_icon("zap", color="#0284c7")} ほんの少しの変化で騙せる理由</h3>
            AIが判断に使っている数値は、人間の目には見えないほど微小な変化に<b>極めて敏感</b>です。<br>
            画像全体の変化量がわずか<b>0.007%</b>のノイズを加えるだけで、AIの内部計算が狂い、
            全く別の物体として認識してしまうことがあります。
            </div>
            """, unsafe_allow_html=True)

        with col_box:
            st.markdown(f"""
            <div style="background:#eef2fb; border:2px solid #0284c7; padding:20px;
                        font-family:monospace; border-left:6px solid #c026d3;">
                <div style="color:#b45309; font-weight:bold; margin-bottom:12px;">
                    {svg_icon("play", size=14, color="#b45309")} ADVERSARIAL DEMO
                </div>
                <div style="color:#059669;">元画像: パンダ</div>
                <div style="color:#000000; margin:4px 0;">確信度: 99.3%</div>
                <br>
                <div style="color:#dc2626; font-size:1.3rem;">＋ ε ノイズ</div>
                <div style="color:#64748b; font-size:0.8rem;">（人間には見えない変化量）</div>
                <br>
                <div style="color:#059669;">↓ AIの判定が変わる</div>
                <br>
                <div style="color:#c026d3; font-weight:bold;">認識結果: テナガザル</div>
                <div style="color:#000000; margin:4px 0;">確信度: 94.1%</div>
                <br>
                <div style="border-top:1px solid #cbd5e1; padding-top:10px;
                            color:#64748b; font-size:0.78rem;">
                    ← Goodfellow et al., 2014<br>
                    最初に発見された対抗的サンプル
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("AIが騙される3つの根本理由", "bar-chart", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("image", color="#0284c7")} テクスチャ偏重</h4>
                <p>CNNは模様・質感を過度に重視します。
                「ゾウの皮膚の質感」でゾウと判断するため、
                質感を少し変えるだけで騙せます。</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("layers", color="#0284c7")} 高次元の脆弱性</h4>
                <p>画像は数百万次元の空間で表現されます。
                人間には見えない方向へほんの少し動かすだけで
                AIの「判断境界線」を越えられます。</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("zap", color="#0284c7")} 勾配の逆用</h4>
                <p>AIは「どの方向に変化させると最も迷うか」を
                計算できます。攻撃者はこれを逆用して
                最小限の変化で最大の混乱を引き起こします。</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("人間 vs AI — 認識の仕組み比較", "puzzle", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown(f"""
            <div style="background:#ffffff; border:2px solid #059669;
                        border-left:6px solid #059669; padding:18px;">
                <h3 style="color:#059669; background:transparent !important;
                           border:none !important; padding:0 !important;
                           box-shadow:none !important; margin:0 0 12px 0;">
                    {svg_icon("users", size=20, color="#059669")} 人間の認識
                </h3>
                <ul style="color:#000000; line-height:2;">
                    <li>形・シルエットを重視</li>
                    <li>少々ノイズがあっても正しく認識</li>
                    <li>文脈・前後関係から推測</li>
                    <li>「これは何？」を直感的に判断</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with comp_cols[1]:
            st.markdown(f"""
            <div style="background:#ffffff; border:2px solid #c026d3;
                        border-left:6px solid #c026d3; padding:18px;">
                <h3 style="color:#c026d3; background:transparent !important;
                           border:none !important; padding:0 !important;
                           box-shadow:none !important; margin:0 0 12px 0;">
                    {svg_icon("bot", size=20, color="#c026d3")} AIの認識
                </h3>
                <ul style="color:#000000; line-height:2;">
                    <li>テクスチャ・模様を重視</li>
                    <li>微小な数値変化に過敏</li>
                    <li>文脈を持たず数値のみで判断</li>
                    <li>「確率が最大の答え」を出力</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("AIの弱点を実際に体験してみよう！", "microscope", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        img_arr_weak = np.array(target_image.resize((400, 400)))

        experiment = st.radio(
            "試してみる弱点を選択：",
            ["ノイズ耐性", "回転耐性", "コントラスト依存"],
            horizontal=True,
            key="weakness_exp"
        )

        test_img = img_arr_weak.copy()

        if experiment == "ノイズ耐性":
            noise_level = st.slider("ノイズ強度（大きいほど荒くなる）", 0, 128, 20, key="noise_slider_w")
            if noise_level > 0:
                noise    = np.random.normal(0, noise_level, img_arr_weak.shape)
                test_img = np.clip(img_arr_weak.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            st.caption("高周波ノイズはCNNの特徴抽出を破壊します。")

        elif experiment == "回転耐性":
            angle = st.slider("回転角度（度）", 0, 180, 45, key="rotate_slider_w")
            if angle > 0:
                h, w     = img_arr_weak.shape[:2]
                M        = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                test_img = cv2.warpAffine(img_arr_weak, M, (w, h))
            st.caption("多くのCNNは回転不変性を完全には持ちません。")

        else:
            contrast = st.slider("コントラスト倍率", 0.1, 2.0, 0.3, step=0.05, key="contrast_slider_w")
            test_img = cv2.convertScaleAbs(img_arr_weak, alpha=contrast, beta=0)
            st.caption("低コントラストはエッジ情報を弱め、AIを迷わせます。")

        col_orig_w, col_test_w = st.columns(2)

        with col_orig_w:
            st.markdown("""
            <div class="img-frame">
                <span class="step-badge">ORIGINAL</span>
                <div class="caption-text">元の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(img_arr_weak, use_container_width=True)
            if TORCH_AVAILABLE:
                orig_lbl_w, orig_pct_w = predict_image(img_arr_weak)[:2]
                st.metric("AIの判定", orig_lbl_w[:24] if len(orig_lbl_w) <= 24 else orig_lbl_w[:21] + "…")
                st.metric("確信度",   f"{orig_pct_w:.1f}%")

        with col_test_w:
            st.markdown("""
            <div class="img-frame frame-phase2">
                <span class="step-badge">MODIFIED</span>
                <div class="caption-text">変化後の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(test_img, use_container_width=True)
            if TORCH_AVAILABLE:
                test_lbl_w, test_pct_w = predict_image(test_img)[:2]
                delta_w = test_pct_w - orig_pct_w
                st.metric("AIの判定", test_lbl_w[:24] if len(test_lbl_w) <= 24 else test_lbl_w[:21] + "…")
                st.metric("確信度",   f"{test_pct_w:.1f}%", f"{delta_w:+.1f}%")

        if TORCH_AVAILABLE:
            if orig_lbl_w != test_lbl_w:
                st.error(f"AIが「{orig_lbl_w}」→「{test_lbl_w}」へ誤認識しました！")
            elif abs(orig_pct_w - test_pct_w) > 10:
                st.warning(f"AIの確信度が{abs(orig_pct_w - test_pct_w):.1f}%も変化しました！")
            else:
                st.success("AIはまだ正確に認識しています。スライダーを動かして攻撃を強めてみよう！")
        else:
            st.info("② 騙しシミュレーター タブでAIへの実際の影響を体験できます！")

# ---------------------------------------------------------------
# TAB 2: 騙しシミュレーター
# ---------------------------------------------------------------
with tab2:
    st.markdown(heading("AI騙しチャレンジ！", "gamepad", level=2), unsafe_allow_html=True)

    # ===== ベース画像ピッカー =====
    # DATA_DIR に実在するファイルだけを候補にする（欠損は静かに除外）
    available_bases = {
        fname: jp for fname, jp in ADV_BASE_IMAGES.items()
        if os.path.exists(os.path.join(DATA_DIR, fname))
    }
    if available_bases:
        picker_col, hint_col = st.columns([2, 3])
        with picker_col:
            base_fname = st.selectbox(
                "攻撃するベース画像を選ぶ",
                list(available_bases.keys()),
                format_func=lambda f: f"{available_bases[f]}（{f}）",
                key="adv_base_img_select_adv2",
            )
        try:
            sim_image = Image.open(os.path.join(DATA_DIR, base_fname)).convert("RGB")
        except Exception:
            sim_image = target_image
            base_fname = "bird.jpg"
        with hint_col:
            if base_fname == "hyousiki.jpg":
                st.markdown(f"""
                <div class="explanation-box" style="margin:0;">
                <h3>{svg_icon("alert-triangle", color="#0284c7")} 道路標識で試すと？</h3>
                この標識画像で攻撃を試すと、<b>実際の自動運転車がなぜ危険にさらされるか</b>がより実感できます。
                「③ 攻撃の仕組み」タブの<b>自動運転への脅威</b>と合わせて確かめてみよう。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="explanation-box" style="margin:0;">
                <h3>{svg_icon("image", color="#0284c7")} 画像を変えて実験</h3>
                被写体を変えると、攻撃の効きやすさが変わります。<b>道路標識</b>を選ぶと、
                自動運転を狙った現実の攻撃を体験できます。
                </div>
                """, unsafe_allow_html=True)
    else:
        sim_image = target_image

    img_array = np.array(sim_image.resize((400, 400)))

    # --- 攻撃を実行 ---
    perturbed_array = apply_attack(img_array, selected_attack_key, attack_strength)

    # --- ノイズ可視化（10倍増幅）---
    diff = np.abs(img_array.astype(np.float32) - perturbed_array.astype(np.float32))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

    # --- AI推論 ---
    visibility    = compute_noise_visibility(img_array, perturbed_array)
    orig_label, orig_conf, orig_top5 = predict_image(img_array)
    pert_label, pert_conf, pert_top5 = predict_image(perturbed_array)
    conf_drop     = max(0.0, orig_conf - pert_conf)
    result_msg, result_color = judge_deception(orig_conf, pert_conf, orig_label, pert_label)

    # ===== 画像3枚の比較表示 =====
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        col_orig, col_pert, col_diff = st.columns(3)

        with col_orig:
            st.markdown("""
            <div class="img-frame">
                <span class="step-badge">ORIGINAL</span>
                <div class="caption-text">元の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(img_array, use_container_width=True)
            if TORCH_AVAILABLE:
                st.metric("AIの判定", orig_label[:22] if len(orig_label) <= 22 else orig_label[:19] + "…")
                st.metric("確信度", f"{orig_conf:.1f}%")
            else:
                st.info("（AI推論なし）")

        with col_pert:
            st.markdown("""
            <div class="img-frame frame-phase2">
                <span class="step-badge">ATTACKED</span>
                <div class="caption-text">攻撃後の画像</div>
            </div>""", unsafe_allow_html=True)
            st.image(perturbed_array, use_container_width=True)
            if TORCH_AVAILABLE:
                st.metric("AIの判定", pert_label[:22] if len(pert_label) <= 22 else pert_label[:19] + "…")
                st.metric("確信度", f"{pert_conf:.1f}%", f"{pert_conf - orig_conf:+.1f}%")

        with col_diff:
            st.markdown("""
            <div class="img-frame frame-result">
                <span class="step-badge badge-result">NOISE ×10</span>
                <div class="caption-text" style="color:#c0392b;">ノイズ可視化（10倍）</div>
            </div>""", unsafe_allow_html=True)
            st.image(diff_amplified, use_container_width=True)
            st.metric("ノイズ視認性", f"{visibility:.3f}%")
            st.caption("明るいほどノイズが大きい場所")

    # ===== 騙し判定バナー =====
    st.markdown(f"""
    <div style="background:#ffffff; border:3px solid {result_color};
                border-left:10px solid {result_color}; padding:20px 24px;
                text-align:center; margin:20px 0; box-shadow:6px 6px 0px #000;">
        <span style="color:{result_color}; font-size:1.4rem; font-weight:bold;
                     font-family:'DotGothic16', sans-serif;">
            {result_msg}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ===== 確信度グラフ =====
    if TORCH_AVAILABLE and orig_top5 and pert_top5:
        st.markdown(heading("AIの確信度の変化", "bar-chart", level=2), unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

            chart_col_l, chart_col_r = st.columns(2)
            with chart_col_l:
                st.markdown("**元画像に対する判定 Top-5**")
                fig = draw_confidence_bar(orig_top5, orig_label, "#0284c7")
                st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
                st.pyplot(fig)
                plt.close(fig)
            with chart_col_r:
                st.markdown("**攻撃後の画像に対する判定 Top-5**")
                fig = draw_confidence_bar(pert_top5, orig_label, "#dc2626")
                st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
                st.pyplot(fig)
                plt.close(fig)

    # ===== スコアシステム =====
    st.markdown(heading("騙し効率スコア", "trophy", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        if TORCH_AVAILABLE:
            efficiency    = (conf_drop / max(visibility, 0.001)) * 10
            label_bonus   = 200 if orig_label != pert_label else 0
            total_score   = int(efficiency + label_bonus)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("確信度ダメージ",           f"{conf_drop:.1f}%")
            sc2.metric("ノイズ視認性（低いほど優秀）", f"{visibility:.3f}%")
            sc3.metric("効率スコア",               f"{total_score} pts")

            if total_score > st.session_state["best_score_2"]:
                st.session_state["best_score_2"] = total_score

            # --- 攻撃タイプ別リーダーボードを更新 ---
            board = st.session_state["adv_leaderboard"]
            prev  = board.get(selected_attack_key)
            if prev is None or total_score > prev["score"]:
                board[selected_attack_key] = {
                    "score":      total_score,
                    "conf_drop":  conf_drop,
                    "visibility": visibility,
                }

            st.info(
                f"今回の最高スコア：**{st.session_state['best_score_2']} pts**",
            )

            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("lightbulb", color="#0284c7")} スコアの計算式</h3>
            <b>スコア = (確信度下落幅 ÷ ノイズ視認性) × 10 ＋ ラベル変更ボーナス（+200）</b><br><br>
            目標は「人間が気付かないほど小さいノイズで、AIに大きなダメージを与えること」。<br>
            これは本物の対抗的サンプル攻撃の目的と全く同じです。
            強度スライダーを小さくして、攻撃タイプを変えて試してみよう！
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("PyTorch が利用できないためAI確信度は計算できません。画像の変化を目で楽しんでください！")

    # ===== 攻撃タイプ別リーダーボード =====
    st.markdown(heading("攻撃タイプ別リーダーボード", "trophy", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        # このページ限定の小さなスコープ付きスタイル（assets/style.css は触らない方針）
        st.markdown("""
        <style>
        .adv-lb-card { background:var(--bg-card); border:2px solid var(--border-main);
                       border-left:6px solid var(--color-cyan); padding:12px 14px;
                       box-shadow:4px 4px 0px #000; min-height:150px; }
        .adv-lb-card.locked-card { border-left-color:var(--border-dim) !important; }
        .adv-lb-title { display:flex; align-items:center; gap:0.4em; font-weight:bold;
                        font-size:0.9rem; margin-bottom:8px; }
        .adv-lb-score { font-family:var(--font-mono); color:var(--color-yellow);
                        font-size:1.5rem; font-weight:900; line-height:1.1; }
        .adv-lb-row   { color:var(--text-dim); font-size:0.74rem; font-family:var(--font-mono);
                        margin-top:4px; }
        .adv-lb-locked-msg { color:var(--text-dim); font-size:0.8rem; margin-top:6px; }
        </style>
        """, unsafe_allow_html=True)

        board = st.session_state["adv_leaderboard"]
        lb_cols = st.columns(4)
        for col, (key, meta) in zip(lb_cols, ATTACK_BY_KEY.items()):
            entry = board.get(key)
            with col:
                if entry is not None:
                    st.markdown(f"""
                    <div class="adv-lb-card">
                        <div class="adv-lb-title" style="color:{meta['color']};">
                            {svg_icon(meta['icon'], size=16, color=meta['color'])}{meta['name']}
                        </div>
                        <div class="adv-lb-score">{entry['score']}<span style="font-size:0.7rem;"> pts</span></div>
                        <div class="adv-lb-row">確信度ダメージ {entry['conf_drop']:.1f}%</div>
                        <div class="adv-lb-row">ノイズ視認性 {entry['visibility']:.3f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="adv-lb-card locked-card">
                        <div class="adv-lb-title" style="color:var(--text-dim);">
                            {svg_icon("lock", size=16, color="#64748b")}{meta['name']}
                        </div>
                        <div class="adv-lb-locked-msg">未挑戦<br>この攻撃タイプをまだ試していません。</div>
                    </div>
                    """, unsafe_allow_html=True)

        tried    = len(board)
        remaining = [m["name"] for k, m in ATTACK_BY_KEY.items() if k not in board]
        st.progress(tried / len(ATTACK_BY_KEY))

        if tried >= len(ATTACK_BY_KEY):
            if not st.session_state["adv_full_clear_done_adv2"]:
                st.session_state["adv_full_clear_done_adv2"] = True
            st.success("完全制覇！4種類すべての攻撃タイプを記録しました。")
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("shield", color="#0284c7")} なぜ「多様な攻撃」で試すのか</h3>
            現実のAIロバストネス評価（adversarial robustness testing）では、
            <b>1種類の攻撃だけでなく、性質の異なる多数の攻撃手法</b>でモデルを検証します。
            ある攻撃には強くても別の攻撃には脆い、というのはよくあること。
            4タイプすべてを試したあなたは、まさに本物のセキュリティ評価者と同じ視点を持てました！
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(
                f"完全制覇まであと{len(remaining)}種類！未挑戦：{'、'.join(remaining)}",
            )

# ---------------------------------------------------------------
# TAB 3: 攻撃の仕組み
# ---------------------------------------------------------------
with tab3:
    st.markdown(heading("敵対的攻撃の仕組み", "zap", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("microscope", color="#0284c7")} FGSM とは何か？（Fast Gradient Sign Method）</h3>
        FGSMは2014年にGoogleのGoodfellow氏らが発見した、AIを最も効率よく騙す攻撃手法です。<br>
        その数式はシンプルです：

        <div style="background:#eef2fb; border:2px solid #b45309; padding:16px; margin:14px 0;
                    font-family:monospace; color:#b45309; font-size:1.1rem; text-align:center;
                    letter-spacing:0.05em;">
            x_adv = x + ε × sign( ∇ₓ J(θ, x, y) )
        </div>

        <ul>
        <li><b>x</b>：元の画像（数値の集合）</li>
        <li><b>ε（イプシロン）</b>：ノイズの強さ。人間の目に見えないほど小さい値</li>
        <li><b>∇ₓJ</b>：「この方向に変えると最も損失が増える」という損失関数の勾配</li>
        <li><b>sign()</b>：プラスかマイナスかだけを取り出す関数</li>
        </ul>
        つまり、<b>「AIが一番間違えやすい方向に、ほんの少しだけ画像を動かす」</b>のが FGSM の本質です。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("攻撃手法の分類図鑑", "map", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        attack_cards = [
            {
                "name": "ランダムノイズ",
                "icon": "wind",
                "badge": "ブラックボックス攻撃",
                "desc": "モデルの内部情報が不要なシンプルな攻撃。誰でも実行できるが効率は低い。",
                "color": "#0284c7",
            },
            {
                "name": "FGSM",
                "icon": "zap",
                "badge": "ホワイトボックス攻撃",
                "desc": "モデルの勾配情報を活用した高効率攻撃。少ないノイズで最大のダメージ。研究の基準手法。",
                "color": "#c026d3",
            },
            {
                "name": "標的型攻撃（Targeted）",
                "icon": "target",
                "badge": "高度ホワイトボックス",
                "desc": "「パンダ→テナガザル」のように特定の誤分類を狙う攻撃。より多くの計算が必要。",
                "color": "#b45309",
            },
            {
                "name": "物理的攻撃",
                "icon": "printer",
                "badge": "リアルワールド攻撃",
                "desc": "プリントしたパターンをカメラに見せる攻撃。自動運転の標識認識や顔認証を物理的に騙せる。",
                "color": "#dc2626",
            },
        ]

        card_col_l, card_col_r = st.columns(2)
        for i, card in enumerate(attack_cards):
            target_col = card_col_l if i % 2 == 0 else card_col_r
            with target_col:
                st.markdown(f"""
                <div style="background:#ffffff; border:2px solid {card['color']};
                            border-left:6px solid {card['color']}; padding:16px;
                            margin-bottom:16px; box-shadow:4px 4px 0px #000;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="color:{card['color']}; font-weight:bold; font-size:1rem;
                                     display:inline-flex; align-items:center; gap:0.4em;">
                            {svg_icon(card['icon'], size=16, color=card['color'])}{card['name']}
                        </span>
                        <span style="background:#e7ecf5; color:#64748b; padding:2px 8px;
                                     font-size:0.72rem; font-family:monospace;">
                            {card['badge']}
                        </span>
                    </div>
                    <p style="color:#000000; font-size:0.88rem; margin:0; line-height:1.7;">
                        {card['desc']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown(heading("現実世界への脅威", "alert-triangle", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        threat_c1, threat_c2, threat_c3 = st.columns(3)
        with threat_c1:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("alert-triangle", color="#0284c7")} 自動運転</h4>
                <p>道路の「一時停止」標識に特殊なシールを貼ると、AIが「時速45マイル制限」と
                誤認識する実験が報告されています。高速での事故につながりかねない深刻な問題です。</p>
            </div>
            """, unsafe_allow_html=True)
        with threat_c2:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("glasses", color="#0284c7")} 顔認証</h4>
                <p>特殊なパターンを印刷した眼鏡をかけることで、スマートフォンやセキュリティカメラの
                顔認証システムを別人として突破できることが研究で示されています。</p>
            </div>
            """, unsafe_allow_html=True)
        with threat_c3:
            st.markdown(f"""
            <div class="guide-box">
                <h4>{svg_icon("stethoscope", color="#0284c7")} 医療診断AI</h4>
                <p>X線画像に微小なノイズを加えることで、正常な肺をAIに「肺炎」と誤診断させる
                攻撃が理論的に可能です。医療現場でのAI活用には特に慎重な対策が必要です。</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 防御と未来
# ---------------------------------------------------------------
with tab4:
    st.markdown(heading("AIを守るために — 防御手法と倫理", "shield", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        defenses = [
            {
                "name": "敵対的訓練（Adversarial Training）",
                "icon": "dumbbell",
                "desc": "攻撃された画像もデータセットに含めてAIを再学習させる方法。最も効果的な防御の一つですが、"
                        "学習コストが高く、見たことのない新しい攻撃手法には対応できないことがあります。",
                "level": 82, "color": "#059669",
            },
            {
                "name": "攻撃検知モデル",
                "icon": "search",
                "desc": "推論の前に「この画像は攻撃されているか？」を別のモデルで判定する方法。"
                        "検知した場合に警告・拒否できますが、検知モデル自体が攻撃される可能性もあります。",
                "level": 68, "color": "#0284c7",
            },
            {
                "name": "入力平滑化（Input Smoothing）",
                "icon": "droplet",
                "desc": "入力画像にぼかし処理をかけてノイズを除去してから推論する方法。"
                        "シンプルで高速ですが、画像の細部も失われるため精度が低下することがあります。",
                "level": 53, "color": "#b45309",
            },
            {
                "name": "ランダム化防御",
                "icon": "shuffle",
                "desc": "推論のたびにランダムな変換を加えることで、攻撃者が狙える方向を予測不能にする方法。"
                        "同じ画像でも毎回少し異なる推論結果になります。",
                "level": 62, "color": "#c026d3",
            },
        ]

        for d in defenses:
            d_col_l, d_col_r = st.columns([4, 1])
            with d_col_l:
                st.markdown(f"""
                <div style="background:#ffffff; border:2px solid {d['color']};
                            border-left:6px solid {d['color']}; padding:14px; margin-bottom:6px;">
                    <strong style="color:{d['color']}; display:inline-flex;
                                   align-items:center; gap:0.4em;">
                        {svg_icon(d['icon'], size=16, color=d['color'])}{d['name']}
                    </strong>
                    <p style="color:#000000; margin:8px 0 0 0; font-size:0.88rem;">{d['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
            with d_col_r:
                st.markdown(f"<br>**有効度: {d['level']}%**", unsafe_allow_html=True)
                st.progress(d["level"] / 100)

    st.markdown(heading("あなたはどう考える？— 倫理クイズ", "globe", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>AIセキュリティは社会全体の問題</h3>
        AI騙しは「ハッカーの遊び」ではありません。自動運転・医療診断・セキュリティカメラ…
        AIが社会インフラになった今、<b>AIの脆弱性を理解することは社会全体の安全に直結</b>します。
        </div>
        """, unsafe_allow_html=True)

        ethics_q = st.radio(
            "もし自動運転AIの重大な脆弱性を発見したら、あなたはどうしますか？",
            [
                "自動車メーカーに非公開で報告し、修正後に発表を許可する",
                "すぐにSNSで公開して広く知らせる",
                "脆弱性情報を買い取る業者に売る",
                "修正を待たずにすぐ学術論文として発表する",
            ],
            key="ethics_radio"
        )

        ethics_responses = {
            "自動車メーカーに非公開で報告し、修正後に発表を許可する": (
                "正解です！これが「Responsible Disclosure（責任ある開示）」と呼ばれる国際的に推奨されている倫理的アプローチです。"
                "メーカーが修正する時間を確保しながら、最終的に公開することで社会全体に貢献できます。"
                "多くの大企業は「バグバウンティプログラム」という報奨金制度を設けています。",
                "success", ":material/check_circle:"
            ),
            "すぐにSNSで公開して広く知らせる": (
                "気持ちは分かりますが、即座の公開は悪意ある攻撃者にも情報が届くリスクがあります。"
                "修正が完了する前に実害が出る可能性があるため、まず開発者への通知が優先されます。",
                "warning", ":material/warning:"
            ),
            "脆弱性情報を買い取る業者に売る": (
                "正規のバグバウンティプログラム以外での脆弱性売買は、法的・倫理的に問題があります。"
                "悪意ある攻撃者への販売は刑事責任につながる可能性があります。",
                "error", ":material/cancel:"
            ),
            "修正を待たずにすぐ学術論文として発表する": (
                "学術発表は重要ですが、一般的には事前に開発者へ通知し、修正完了後に発表する"
                "「Coordinated Disclosure（調整された開示）」が推奨されます。多くの学術誌もこの手順を求めています。",
                "info", ":material/lightbulb:"
            ),
        }

        if ethics_q in ethics_responses:
            msg, level, _msg_icon = ethics_responses[ethics_q]
            getattr(st, level)(msg)

    st.markdown(heading("AIセキュリティの未来", "rocket", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explanation-box">
        <h3>AIを「騙しにくく」する研究の最前線</h3>
        <ul>
        <li><b>形状バイアスの訓練</b>：テクスチャではなく「形」で判断するよう意図的に訓練する研究</li>
        <li><b>証明可能な堅牢性</b>：「εの範囲内では絶対に騙されない」と数学的に証明できるAIの開発</li>
        <li><b>人間と同じ知覚</b>：人間の視覚システムを真似た、よりロバストなアーキテクチャの研究</li>
        <li><b>量子機械学習</b>：量子計算を使った、古典的攻撃が通用しないAIシステムの理論研究</li>
        </ul>

        <h3>あなたが学べること</h3>
        AIの弱点を知ることは、AIを「正しく使う」ための第一歩です。<br>
        騙されるAIをただ批判するのではなく、<b>なぜ騙されるのか・どう防ぐのかを理解する人</b>こそ、
        これからのAI時代をリードできる人材です。
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# SECTION 11: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。掌握する。</p>
    <p>MISSION 02: AI騙し — 対抗的サンプルの世界へようこそ</p>
</div>
""", unsafe_allow_html=True)
