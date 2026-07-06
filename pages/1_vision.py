import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import base64
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
    page_title="ミッション01: AIの目 | AI Inquiry Lab",
    page_icon=":material/visibility:",
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

@st.cache_data
def get_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def create_dummy_image(text: str, color: tuple) -> Image.Image:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, text, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return Image.fromarray(img)

# ================================================================
# SECTION 4: AIモデル（推論エンジン）
# ================================================================
if TORCH_AVAILABLE:
    @st.cache_resource
    def load_model():
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        m = tv_models.mobilenet_v2(weights=weights)
        m.eval()
        return m, weights.meta["categories"], weights.transforms()
    _model, _categories, _preprocess = load_model()
else:
    _model, _categories, _preprocess = None, [], None

@st.cache_data(show_spinner=False)
def predict_image(img_array: np.ndarray) -> tuple:
    """画像を推論して (ラベル, 確信度%) を返す。
    同じ画像に対する推論はキャッシュし、無関係な操作での再実行のたびに
    重いモデル推論をやり直さないようにする（_model/_preprocessはロード後不変）。"""
    if not TORCH_AVAILABLE or _model is None:
        return "AI機能は現在利用できません", 0.0
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    input_tensor = _preprocess(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = _model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probs, 0)
    return _categories[top_idx.item()], top_prob.item() * 100

# ================================================================
# SECTION 5: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)

render_top_nav("vision")

st.markdown(f"""
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon('eye', size=30)} AIの目</h1>
    <p class="sub-title-text">MISSION 01 — AIは画像をどうやって「見て」いるのか？</p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# SECTION 7: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#3b82f6; font-weight:bold; font-size:0.9rem;">{svg_icon('eye', size=15, color='#3b82f6')} VISION MODE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.info("**ヒント** AIはピクセルごとの「数値」を見ています。")
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
# SECTION 8: 画像ロード（固定: 鳥の画像を使用）
# ================================================================
bird_path = os.path.join(DATA_DIR, "bird.jpg")
image = Image.open(bird_path).convert("RGB")

# ================================================================
# SECTION 9: STEP 2 — タブコンテンツ
# ================================================================
st.markdown(heading("タブを選んでAIの見ている世界を体験しよう！", "microscope", level=2), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 数字の世界 (RGB)",
    "② 輪郭の世界 (Edge)",
    "③ フィルタの世界 (CNN)",
    "④ AI vs 人間チャレンジ",
])

image_resized = image.resize((600, int(600 * image.height / image.width)))
img_array     = np.array(image_resized)

# ---------------------------------------------------------------
# TAB 1: RGBの世界
# ---------------------------------------------------------------
with tab1:
    st.markdown(heading("画像の正体は『数字の集まり』！", "cpu", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_main1, col_main2 = st.columns([1, 2])

        with col_main1:
            rgb_b64 = get_image_as_base64(os.path.join(DATA_DIR, "rgb.jpg"))
            if rgb_b64:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:8px; overflow:hidden; background:white;">
                    <img src="data:image/jpeg;base64,{rgb_b64}" style="width:100%; display:block;">
                </div>
                <p style="text-align:center; color:#666; font-size:0.8em; margin-top:5px;">
                    【図：加法混色】R・G・Bの光の強さで色が決まる
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info("rgb.jpg 図説なし")

        with col_main2:
            st.markdown("""
            <div class="explanation-box">
            <h3>1. 画像は「小さな点」の集合体</h3>
            スマホで見ている綺麗な写真は、実は「ピクセル」という超小さな点の集まりです。
            ひとつひとつの点は、<b>赤(R)・緑(G)・青(B)</b>という3つの光の強さで作られています。

            <h3>2. コンピュータが見ているのは「数字」</h3>
            コンピュータは「色」として画像を認識できません。
            その代わり、各ピクセルの光の強さを<b>0〜255の数字</b>として処理しています。<br>
            R・G・Bがそれぞれ256通りなので、組み合わせは
            <b>256×256×256 ＝ 約1,677万通り</b>！

            <h3>3. AIはどうやって物体を見つける？</h3>
            AIはこの膨大な数字の並びをスキャンして、
            「この数字のパターンは猫の耳だ！」と判断します。
            人間には単なる色に見えるものでも、AIには<b>計算可能なデータの塊</b>です。
            </div>
            """, unsafe_allow_html=True)

        st.divider()

    st.markdown(heading("AIに見えているデータを覗いてみよう！", "search", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_img, col_data = st.columns([3, 2])

        with col_img:
            st.subheader("RGBレイヤーを分解してみよう")
            col_orig_disp, col_rgb_disp = st.columns(2)
            with col_orig_disp:
                st.markdown("▼ 元画像")
                st.image(image_resized, caption="3色の組み合わせ画像", use_container_width=True)
            with col_rgb_disp:
                st.markdown("▼ 成分画像")
                target_vis_placeholder = st.empty()

        with col_data:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("AIはこの数字の変化（グラデーション）を計算して、物体の形を判断します。")

            channel = st.radio(
                "分解する色を選択",
                ["Red", "Green", "Blue"],
                horizontal=True,
                key="rgb_selector_fixed"
            )

            st.markdown(heading("調査ポイントを動かす", "search", level=5), unsafe_allow_html=True)
            col_x, col_y = st.columns(2)
            pick_x = col_x.slider("横の位置 (X)", 0, image_resized.width  - 1, int(image_resized.width  / 2), key="slider_x")
            pick_y = col_y.slider("縦の位置 (Y)", 0, image_resized.height - 1, int(image_resized.height / 2), key="slider_y")

            r_ch  = img_array[:, :, 0]
            g_ch  = img_array[:, :, 1]
            b_ch  = img_array[:, :, 2]
            zeros = np.zeros_like(r_ch)

            if channel == "Red":
                target_vis = np.stack([r_ch, zeros, zeros], axis=2)
                cmap_style = "Reds"
            elif channel == "Green":
                target_vis = np.stack([zeros, g_ch, zeros], axis=2)
                cmap_style = "Greens"
            else:
                target_vis = np.stack([zeros, zeros, b_ch], axis=2)
                cmap_style = "Blues"

            target_vis_placeholder.image(target_vis, caption=f"明るい場所＝{channel}が強い", use_container_width=True)

            pix_r, pix_g, pix_b = img_array[pick_y, pick_x, :3]
            m1, m2, m3 = st.columns(3)
            m1.metric("Red",   int(pix_r))
            m2.metric("Green", int(pix_g))
            m3.metric("Blue",  int(pix_b))

            st.markdown(f"**▼ 周辺の数値データ** ({channel}チャンネル)")

            zoom_radius = 4
            y_start = max(0, pick_y - zoom_radius)
            y_end   = min(target_vis.shape[0], pick_y + zoom_radius + 1)
            x_start = max(0, pick_x - zoom_radius)
            x_end   = min(target_vis.shape[1], pick_x + zoom_radius + 1)
            zoom_area = target_vis[y_start:y_end, x_start:x_end]
            zoom_disp = cv2.resize(zoom_area, (250, 250), interpolation=cv2.INTER_NEAREST)

        col_zoom1, col_zoom2 = st.columns([3, 2])
        zoom_area_orig = img_array[y_start:y_end, x_start:x_end]
        zoom_disp_orig = cv2.resize(zoom_area_orig, (250, 250), interpolation=cv2.INTER_NEAREST)

        with col_zoom1:
            c1, c2 = st.columns(2)
            c1.image(zoom_disp_orig, caption="選択範囲(元)",  use_container_width=True)
            c2.image(zoom_disp,      caption="選択範囲(成分)", use_container_width=True)

        with col_zoom2:
            ch_idx      = 0 if channel == "Red" else 1 if channel == "Green" else 2
            zoom_single = zoom_area[:, :, ch_idx].astype(np.int32)
            max_val     = int(np.max(zoom_single))
            max_pos_local = np.unravel_index(np.argmax(zoom_single), zoom_single.shape)
            col_labels  = list(range(x_start, x_end))
            row_labels  = list(range(y_start, y_end))
            max_x = col_labels[max_pos_local[1]]
            max_y = row_labels[max_pos_local[0]]
            df_subset = pd.DataFrame(zoom_single, columns=col_labels, index=row_labels)
            st.table(
                df_subset.style
                    .background_gradient(cmap=cmap_style, axis=None, vmin=0, vmax=255)
                    .highlight_max(axis=None, props="color:white; font-weight:bold; background-color:#FF4B4B;")
                    .format("{:d}")
            )
            st.write(f"**中心**:({pick_x},{pick_y}) | **最大輝度**:({max_x},{max_y}) [値:{max_val}]")
            if max_val == 255:
                st.success("ビンゴ！一番明るい点を発見！")

# ---------------------------------------------------------------
# TAB 2: エッジ検出
# ---------------------------------------------------------------
with tab2:
    st.markdown(heading("輪郭を取り出す（エッジ検出）", "ruler", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
        <p>AIは物体の「輪郭（エッジ）」を、とても重要な手がかりとして使います。
        エッジとは、画像の中で色や明るさが急激に変わる境界線のことです。<br>
        AIはエッジを探す前に、細かいノイズを消すために画像を<b>「ぼかす」</b>処理をします。<br>
        2つの有名なアルゴリズム（CannyとLaplacian）を切り替えて、その違いを体験しましょう！</p>
        </div>
        """, unsafe_allow_html=True)

        edge_mode = st.radio("アルゴリズムを選択", ["Canny法", "ラプラシアンフィルタ"], key="edge_mode_radio")

        st.markdown(f"{svg_icon('wrench', size=16)} **AIフィルタ設定**", unsafe_allow_html=True)
        c_set1, c_set2, c_set3 = st.columns(3)
        blur_val = c_set1.slider("ガウシアンフィルタ（ぼかし）", 1, 15, 3, step=2)

        if edge_mode == "Canny法":
            th1 = c_set2.slider("感度:Min", 0, 255, 100)
            th2 = c_set3.slider("感度:Max", 0, 255, 200)
        else:
            lap_ksize = c_set2.slider("ラプラシアンフィルタ（エッジ検出）", 1, 7, 3, step=2)
            st.caption("※ラプラシアンは境界の『変化の大きさ』を計算します。")

        gray    = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

        if edge_mode == "Canny法":
            processed_edges = cv2.Canny(blurred, th1, th2)
        else:
            lap_raw = cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_ksize)
            processed_edges = cv2.convertScaleAbs(lap_raw)

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.image(image_resized,   caption="1. 元画像",            use_container_width=True)
        with col_res2:
            st.image(blurred,         caption="2. ぼかし後",           use_container_width=True)
        with col_res3:
            st.image(processed_edges, caption=f"3. {edge_mode} 結果", use_container_width=True)

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon('wrench', size=18)} ガウシアンフィルタ（ぼかし）</h3>
        値を大きくすると画像がよりぼやけます。ぼかすことで細かいノイズが消え、
        物体の大きな輪郭だけが際立ちます。AIはこのぼかしで「重要な輪郭だけ」を見つけやすくします。
        </div>
        """, unsafe_allow_html=True)

        if edge_mode == "Canny法":
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon('lightbulb', size=18)} Cannyエッジ検出の仕組み</h3>
            「感度」は色の変化をどれくらい厳しくチェックして「線」と認めるかの基準です。<br>
            理論上、<b>Min : Max = 1 : 2 または 1 : 3</b> の比率が最も綺麗な線を描けます。<br>
            <ul>
                <li>● <b>Max以上</b>：確実なエッジ → 無条件採用</li>
                <li>● <b>Min〜Maxの間</b>：迷いエッジ → 確実なエッジと繋がっていれば採用</li>
                <li>● <b>Min以下</b>：ノイズ → 無視</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon('sparkles', size=18)} ラプラシアンフィルタの仕組み</h3>
            色の明るさが急激に切り替わる地点を数学の「2階微分」で捉えるフィルタです。<br>
            全方向360度の変化を一度に計算するため、輪郭の点・線を強調します。<br>
            ただしノイズに敏感なため、ぼかし処理とセットで使うことが多いです。
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("ギャラリー：AIの視点プロセス完全図解", "image", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

        def show_row_arrow():
            st.markdown(f'<div class="row-arrow">{svg_icon("arrow-right", size=24)}</div>', unsafe_allow_html=True)

        steps_row1 = [
            {"file": os.path.join(DATA_DIR, "ori.png"),  "badge": "STEP 1", "title": "入力：元画像"},
            {"file": os.path.join(DATA_DIR, "mono.png"), "badge": "STEP 2", "title": "変換：モノクロ"},
            {"file": os.path.join(DATA_DIR, "blur.png"), "badge": "STEP 3", "title": "除去：ぼかし"},
        ]
        steps_row2 = [
            {"file": os.path.join(DATA_DIR, "edge.png"), "badge": "STEP 4", "title": "抽出：エッジ"},
            {"file": os.path.join(DATA_DIR, "ans.png"),  "badge": "TARGET", "title": "理想：正解データ"},
            {"file": os.path.join(DATA_DIR, "dif.png"),  "badge": "RESULT", "title": "判定：一致率"},
        ]

        st.subheader("Phase 1: 情報を削ぎ落とす")
        st.markdown("<br>", unsafe_allow_html=True)
        cols_1 = st.columns([10, 2, 10, 2, 10])
        for i, col_index in enumerate([0, 2, 4]):
            with cols_1[col_index]:
                item = steps_row1[i]
                st.markdown(f"""
                <div class="img-frame">
                    <span class="step-badge">{item['badge']}</span>
                    <div class="caption-text">{item['title']}</div>
                </div>""", unsafe_allow_html=True)
                if os.path.exists(item["file"]):
                    st.image(item["file"], use_container_width=True)
                else:
                    st.warning(f"画像なし: {os.path.basename(item['file'])}")
            if col_index != 4:
                with cols_1[col_index + 1]:
                    show_row_arrow()

        st.markdown(f"""
        <div style="margin-top:30px; display:flex; flex-direction:column;">
            <div style="text-align:right; padding-right:5%;">
                <svg width="60" height="60" viewBox="0 0 100 100">
                    <path d="M 20,10 Q 80,10 80,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                    <polygon points="65,55 80,85 95,55" fill="#D32F2F"/>
                </svg>
            </div>
            <div style="text-align:center; color:black; font-weight:bold; background-color:#c0c0c0;
                        padding:15px; border-radius:10px; margin:10px 0;">
                {svg_icon('wind', size=18, color='black')} 情報を整理したので、ここから「形」を取り出します {svg_icon('wind', size=18, color='black')}
            </div>
            <div style="text-align:left; padding-left:5%;">
                <svg width="60" height="60" viewBox="0 0 100 100">
                    <path d="M 80,10 Q 20,10 20,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                    <polygon points="5,55 20,85 35,55" fill="#D32F2F"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Phase 2: 形を見極める")
        st.markdown("<br>", unsafe_allow_html=True)
        cols_2 = st.columns([10, 2, 10, 2, 10])

        with cols_2[0]:
            item = steps_row2[0]
            st.markdown(f"""
            <div class="img-frame frame-phase2">
                <span class="step-badge badge-phase2">{item['badge']}</span>
                <div class="caption-text">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
        with cols_2[1]: show_row_arrow()

        with cols_2[2]:
            item = steps_row2[1]
            st.markdown(f"""
            <div class="img-frame frame-phase2">
                <span class="step-badge badge-phase2">{item['badge']}</span>
                <div class="caption-text">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
        with cols_2[3]: show_row_arrow()

        with cols_2[4]:
            item = steps_row2[2]
            st.markdown(f"""
            <div class="img-frame frame-result">
                <span class="step-badge badge-result">{item['badge']}</span>
                <div class="caption-text" style="color:#c0392b;">{item['title']}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(item["file"]):
                st.image(item["file"], use_container_width=True)
            else:
                st.warning(f"画像なし: {os.path.basename(item['file'])}")
            st.caption("※一致率が95%を超えると、AIは「完全な認識」として学習を完了します。")

    st.markdown(heading("AIの認識精度を実際に測ってみよう", "bird", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        karasu_path = os.path.join(DATA_DIR, "karasu.png")
        pro_path    = os.path.join(DATA_DIR, "pro.jpg")

        if not os.path.exists(karasu_path) or not os.path.exists(pro_path):
            st.warning("サンプル画像（karasu.png / pro.jpg）が見つかりません。")
        else:
            target_color  = cv2.cvtColor(cv2.imread(karasu_path), cv2.COLOR_BGR2RGB)
            target_gray   = cv2.cvtColor(target_color, cv2.COLOR_RGB2GRAY)
            edge_insert   = cv2.imread(pro_path)
            height, width = edge_insert.shape[:2]
            target_color  = cv2.resize(target_color, (width, height))
            target_gray   = cv2.resize(target_gray,  (width, height))

            y_true_edges = cv2.Canny(target_gray, 200, 300)
            kernel       = np.ones((3, 3), np.uint8)
            y_true_zone  = cv2.dilate(y_true_edges, kernel, iterations=1)

            proc_gray         = cv2.cvtColor(edge_insert, cv2.COLOR_BGR2GRAY)
            _, y_pred_binary  = cv2.threshold(proc_gray, 100, 255, cv2.THRESH_BINARY)
            y_pred_vis        = cv2.dilate(y_pred_binary, kernel, iterations=1)
            y_true_vis        = cv2.dilate(y_true_edges,  kernel, iterations=1)

            vis_img           = np.zeros((height, width, 3), dtype=np.uint8)
            vis_img[:, :, 1]  = (y_true_zone > 0) * 255
            vis_img[:, :, 0]  = (y_pred_vis  > 0) * 255

            ideal_img         = np.zeros((height, width, 3), dtype=np.uint8)
            ideal_img[:, :, 1] = (y_true_zone > 0) * 255
            ideal_img[:, :, 0] = (y_true_vis  > 0) * 255

            intersection = np.logical_and(y_pred_binary > 0, y_true_zone > 0)
            union        = np.logical_or (y_pred_binary > 0, y_true_zone > 0)
            iou_score    = float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0.0

            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.markdown("**元画像**")
                st.image(karasu_path, caption="処理前のカラス", use_container_width=True)
            with col_a2:
                st.markdown(f"{svg_icon('search', size=16)} **重ね合わせ検証**", unsafe_allow_html=True)
                st.image(vis_img, caption="赤=エッジ検出 / 緑=正解領域 / 黄=的中", use_container_width=True)
            with col_a3:
                st.markdown(f"{svg_icon('target', size=16)} **理論上の100%**", unsafe_allow_html=True)
                st.image(ideal_img, caption="理想的な重なり", use_container_width=True)

            st.warning(
                f"参考：エッジ的中率（IoU）は約{iou_score*100:.1f}%でした。"
                "同じカラスに見えても、デジタルデータとしては全く別物だとわかります。"
            )
            st.markdown("""
            <div class="explanation-box">
            <h3>この実験が示すこと</h3>
            <p>人間には「同じカラス」に見えても、AIにとっては画像ごとに全く別のデータです。
            角度・背景・光の当たり方が少し違うだけで、エッジの出る位置がガラッと変わります。
            AIが安定して物体を認識するためには、非常に多くの「似た画像」で学習する必要があります。
            これが「AIには膨大なデータが必要」と言われる理由です。</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: 畳み込みフィルタ
# ---------------------------------------------------------------
with tab3:
    st.markdown(heading("特徴を見つける眼鏡（畳み込みフィルタ）", "glasses", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        lasta_b64 = get_image_as_base64(os.path.join(DATA_DIR, "lasta.png"))
        col_view, col_text = st.columns([2, 3])

        with col_view:
            if lasta_b64:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:8px; overflow:hidden">
                    <img src="data:image/png;base64,{lasta_b64}" style="width:100%; display:block;">
                </div>
                <p style="text-align:center; color:#666; font-size:0.8em; margin-top:5px;">
                    【図：ラスタスキャン】左上から1マスずつ計算
                </p>
                """, unsafe_allow_html=True)
            else:
                st.warning("解説画像が見つかりません（lasta.png）。")

        with col_text:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon('search', size=18)} AIはどうやって画像を見る？</h3>
            <p>AIは画像を「意味」として一瞬で理解しているわけではありません。<br>
            左の図のように、<b>「3×3マスの小さな窓（フィルタ）」</b>を
            左上から1マスずつスライドさせながら計算しています。</p>
            <p>各ピクセル周辺の「色の変化」を数値として測るこの方式を、
            専門用語で<b>「ラスタスキャン」</b>と呼びます。</p>
            <h3>{svg_icon('lightbulb', size=18)} ここがポイント！</h3>
            <p>窓の中の9個の数字が、縦線・横線・角などの特徴に反応するよう設定されています。
            これを画像全体で<b>何万回も繰り返す</b>ことで、
            AIは「これは猫の耳だ！」と気づくことができるのです。</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(heading("有名なフィルタ係数を見てみよう", "layers", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            filter_type = st.selectbox("かけてみるフィルタを選択", [
                "恒等 (何もしない)",
                "ぼかし (平均化/Mean)",
                "シャープ化 (Sharpen)",
                "輪郭抽出 (Laplacian)",
                "縦の輪郭 (Sobel X)",
                "横の輪郭 (Sobel Y)",
                "エンボス (Emboss)",
            ])

            kernel_map = {
                "恒等 (何もしない)":     (np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),       "中央が1で他が0。元の画素をそのまま出力します。", False),
                "ぼかし (平均化/Mean)":  (np.ones((3, 3), np.float32) / 9,                             "周囲9マスの平均をとります。ノイズ低減に有効です。", False),
                "シャープ化 (Sharpen)": (np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),    "中央を強く・周囲を引くことで輪郭をクッキリさせます。", False),
                "輪郭抽出 (Laplacian)": (np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),       "周囲との差分を計算してエッジを検出します。", True),
                "縦の輪郭 (Sobel X)":  (np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),    "左右の差を計算。縦線がある場所だけ光ります。", True),
                "横の輪郭 (Sobel Y)":  (np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),     "上下の差を計算。横線がある場所だけ光ります。", True),
                "エンボス (Emboss)":    (np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),    "斜めの光と影を作り出し、立体的に見せます。", False),
            }

            kernel_val, desc, is_edge = kernel_map[filter_type]
            if is_edge:
                processed_raw = cv2.filter2D(img_array, cv2.CV_64F, kernel_val)
                processed     = cv2.convertScaleAbs(processed_raw)
            else:
                processed = cv2.filter2D(img_array, -1, kernel_val)

            st.markdown(f"<h3>{svg_icon('wrench', size=18)} カーネル（計算式）</h3>", unsafe_allow_html=True)
            df_kernel = pd.DataFrame(kernel_val)
            fmt = "{:.2f}" if filter_type == "ぼかし (平均化/Mean)" else "{:.0f}"
            st.write(df_kernel.style.format(fmt))
            st.info(desc)

        with c2:
            _, sub_right = st.columns([0.2, 1])
            with sub_right:
                st.image(processed, caption=f"【変換後】{filter_type} の世界", use_container_width=True)

        st.divider()
        st.warning("""
        プロ豆知識：なぜ「左上」から？
        昔のWindows画像（BMP）は「左下から右上」に向かって走査していました（数学グラフの名残）。
        現在主流のJPEG・PNG・OpenCVは「左上から右下」（本を読む順序）に走査します。
        """)

    st.markdown(heading("DIYラボ：自分だけのフィルタを作ろう！", "flask", level=2), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        col_lab_input, col_lab_result = st.columns(2)

        with col_lab_input:
            st.markdown(f"<h3>{svg_icon('wrench', size=18)} カーネル行列の入力</h3>", unsafe_allow_html=True)
            k = np.zeros((3, 3))
            r1c1, r1c2, r1c3 = st.columns(3)
            r2c1, r2c2, r2c3 = st.columns(3)
            r3c1, r3c2, r3c3 = st.columns(3)
            with r1c1: k[0,0] = st.number_input("0,0", value=0.0,  step=1.0, key="k00", label_visibility="collapsed")
            with r1c2: k[0,1] = st.number_input("0,1", value=-1.0, step=1.0, key="k01", label_visibility="collapsed")
            with r1c3: k[0,2] = st.number_input("0,2", value=0.0,  step=1.0, key="k02", label_visibility="collapsed")
            with r2c1: k[1,0] = st.number_input("1,0", value=-1.0, step=1.0, key="k10", label_visibility="collapsed")
            with r2c2: k[1,1] = st.number_input("1,1", value=5.0,  step=1.0, key="k11", label_visibility="collapsed")
            with r2c3: k[1,2] = st.number_input("1,2", value=-1.0, step=1.0, key="k12", label_visibility="collapsed")
            with r3c1: k[2,0] = st.number_input("2,0", value=0.0,  step=1.0, key="k20", label_visibility="collapsed")
            with r3c2: k[2,1] = st.number_input("2,1", value=-1.0, step=1.0, key="k21", label_visibility="collapsed")
            with r3c3: k[2,2] = st.number_input("2,2", value=0.0,  step=1.0, key="k22", label_visibility="collapsed")
            st.caption("この数字を変えてみてください！")
            use_abs = st.checkbox("結果を絶対値にする（境目を白く光らせる）", value=False)

        with col_lab_result:
            if use_abs:
                custom_raw = cv2.filter2D(img_array, cv2.CV_64F, k)
                custom_out = cv2.convertScaleAbs(custom_raw)
            else:
                custom_out = cv2.filter2D(img_array, -1, k)
            _, sub_right2 = st.columns([0.2, 1])
            with sub_right2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.image(custom_out, caption="あなたの実験結果", use_container_width=True)

        st.success("""
        実験のヒント:
        ・中央だけ大きい値で他を0 → 元画像のまま（恒等変換）
        ・中央を大きく＋周囲を負 → コントラスト強化（シャープ）
        ・全マス同じ値 → 平均化（ぼかし）
        ・正と負の値を混ぜる → 差分計算（エッジ検出）
        """)

# ================================================================
# SECTION 11: TAB 4 — AI vs 人間チャレンジ
# ================================================================
with tab4:
    # このタブ専用のスコープCSS（vision- プレフィックスで衝突回避）
    st.markdown("""
    <style>
    .vision-vch-stage-label {
        font-size: 0.72rem; color: var(--color-cyan, #22d3ee);
        text-align: center; margin-top: 4px; font-weight: bold;
        letter-spacing: 0.5px;
    }
    .vision-vch-crossline {
        display: flex; flex-wrap: wrap; gap: 0.5em; align-items: center;
        margin: 0.4em 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        heading("AIと人間、どちらが先に「わかる」？", "target", level=2),
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="explanation-box">
    <p>写真をどんどん<b>モザイク（低解像度）</b>にしていくと、ある段階で「何の写真か」が
    わからなくなります。人間のあなたと、AI（MobileNet）は、それぞれ
    <b>どのくらいの粗さ</b>まで正体を見抜けるでしょうか？<br>
    ぼやけた写真でも、人間もAIも「見分けるための手がかり（ピクセルの模様）」が
    十分あって初めて自信を持てます。その「わかる境界線」を比べてみましょう。</p>
    </div>
    """, unsafe_allow_html=True)

    # --- セッション状態（このファイル専用のユニークキー） ---
    if "vision_stage_attempts" not in st.session_state:
        st.session_state["vision_stage_attempts"] = 0
    if "vision_tried_images_vch" not in st.session_state:
        st.session_state["vision_tried_images_vch"] = set()

    # --- 被写体の候補（data/ に実在するものだけ） ---
    _vch_candidates = [
        ("bird.jpg",       "鳥"),
        ("cat.jpg",        "猫"),
        ("building.jpg",   "建物"),
        ("mountain.jpg",   "山"),
        ("soccerball.jpg", "サッカーボール"),
    ]
    _vch_available = [
        (fn, jp) for fn, jp in _vch_candidates
        if os.path.exists(os.path.join(DATA_DIR, fn))
    ]

    if not _vch_available:
        st.warning("チャレンジ用の画像が data/ に見つかりませんでした。")
    else:
        with st.container():
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

            sel_label = st.selectbox(
                "被写体の画像を選ぼう",
                options=[jp for _, jp in _vch_available],
                key="vch_subject_select_vch",
            )
            sel_file = next(fn for fn, jp in _vch_available if jp == sel_label)
            sel_path = os.path.join(DATA_DIR, sel_file)

            # 元画像を読み込み（正方形に寄せてモザイク比較しやすくする）
            _vch_pil = Image.open(sel_path).convert("RGB").resize((256, 256))
            _vch_full = np.array(_vch_pil)

            # 段階的に劣化させた版を生成（最も粗い→最も鮮明）
            _vch_res_steps = [4, 8, 16, 32, 64, 256]  # 256 = フル解像度
            _vch_stages = []
            for res in _vch_res_steps:
                if res >= 256:
                    stage_img = _vch_full.copy()
                else:
                    small = cv2.resize(_vch_full, (res, res), interpolation=cv2.INTER_AREA)
                    stage_img = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
                _vch_stages.append(stage_img)

            n_stages = len(_vch_stages)
            stage_names = [f"{r}px" if r < 256 else "フル" for r in _vch_res_steps]

            # --- モザイク・ギャラリー（粗い→鮮明） ---
            st.markdown(
                heading("モザイク・ギャラリー（粗い → 鮮明）", "image", level=3),
                unsafe_allow_html=True,
            )
            gallery_cols = st.columns(n_stages)
            for idx, (gcol, stage_img, sname) in enumerate(
                zip(gallery_cols, _vch_stages, stage_names)
            ):
                with gcol:
                    st.markdown(
                        f'<div class="img-frame"><span class="step-badge">STAGE {idx+1}</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.image(stage_img, use_container_width=True)
                    st.markdown(
                        f'<div class="vision-vch-stage-label">{sname}</div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # --- 人間の直感チェック（スコアなし・自己申告） ---
            st.markdown(
                heading("あなたの直感：何段階目で「わかった」？", "eye", level=3),
                unsafe_allow_html=True,
            )
            human_stage = st.slider(
                "この段階なら自信を持って当てられる、という所を選ぼう（正解プレッシャーなし・素直な直感でOK）",
                min_value=1,
                max_value=n_stages,
                value=(n_stages // 2),
                key="vch_human_stage_slider_vch",
            )
            st.caption(
                f"あなたの回答: STAGE {human_stage}（{stage_names[human_stage-1]}）で確信"
            )

            run_challenge = st.button(
                "AIの判定を見る",
                key="vch_run_button_vch",
                use_container_width=True,
            )

            if run_challenge:
                # 試行回数カウント（同じ画像の重複は1回扱いにしつつ、総試行も加算）
                st.session_state["vision_stage_attempts"] += 1
                st.session_state["vision_tried_images_vch"].add(sel_file)

                if not TORCH_AVAILABLE:
                    st.warning(
                        "AI（MobileNet）が利用できない環境のため、AIの判定は表示できません。"
                        "人間側の直感チェックだけ体験できます。",
                    )
                else:
                    # 各段階でAIを推論し、確信度が約50%を超える最初の段階を探す
                    ai_confs = []
                    ai_labels = []
                    ai_confident_stage = None
                    for idx, stage_img in enumerate(_vch_stages):
                        label, conf = predict_image(stage_img)
                        ai_confs.append(conf)
                        ai_labels.append(label)
                        if ai_confident_stage is None and conf >= 50.0:
                            ai_confident_stage = idx + 1

                    # 確信度の推移グラフ
                    st.markdown(
                        heading("AIの確信度は解像度とともにどう伸びる？", "trending-up", level=3),
                        unsafe_allow_html=True,
                    )
                    st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x = list(range(1, n_stages + 1))
                    ax.plot(x, ai_confs, marker="o", color="#22d3ee", linewidth=2, label="AI confidence")
                    ax.axhline(50, color="#f472b6", linestyle="--", linewidth=1.2, label="50% threshold")
                    ax.axvline(human_stage, color="#facc15", linestyle=":", linewidth=1.5, label="Human confident")
                    ax.set_xticks(x)
                    ax.set_xticklabels(stage_names)
                    ax.set_ylim(0, 100)
                    ax.set_xlabel("Stage (resolution)")
                    ax.set_ylabel("AI confidence (%)")
                    ax.legend(fontsize=8, loc="upper left")
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # 判定サマリ
                    col_h, col_a = st.columns(2)
                    with col_h:
                        st.metric(
                            "人間が確信した段階",
                            f"STAGE {human_stage}",
                            help=f"解像度 {stage_names[human_stage-1]}",
                        )
                    with col_a:
                        if ai_confident_stage is not None:
                            st.metric(
                                "AIが確信した段階 (≥50%)",
                                f"STAGE {ai_confident_stage}",
                                help=f"解像度 {stage_names[ai_confident_stage-1]} / "
                                     f"予測: {ai_labels[ai_confident_stage-1]}",
                            )
                        else:
                            st.metric("AIが確信した段階 (≥50%)", "到達せず")

                    # 最終予測
                    st.info(
                        f"フル解像度でのAIの予測: 「{ai_labels[-1]}」"
                        f"（確信度 {ai_confs[-1]:.1f}%）",
                    )

                    # クロスオーバーの考察
                    if ai_confident_stage is None:
                        verdict = ("この画像ではAIは最後まで50%の確信に届きませんでした。"
                                   "モザイクや被写体がAIの学習データと噛み合わなかった可能性があります。")
                    elif ai_confident_stage < human_stage:
                        verdict = ("AIの方が粗い段階で確信に達しました。"
                                   "AIは人間が意味を感じ取る前から、ピクセルの模様（テクスチャ）を"
                                   "手がかりにパターン照合できることがあります。")
                    elif ai_confident_stage > human_stage:
                        verdict = ("あなたの方が粗い段階で見抜きました。"
                                   "人間は少ない手がかりから文脈で補って推測できますが、"
                                   "AIは十分な識別ピクセルが揃うまで自信を持てませんでした。")
                    else:
                        verdict = "偶然にも、人間とAIが同じ段階で確信に達しました。"

                    st.markdown(f"""
                    <div class="explanation-box">
                    <h3>{svg_icon('lightbulb', size=18)} この結果が教えてくれること</h3>
                    <p>{verdict}</p>
                    <p>ぼやけた写真を人間が認識するのに十分なディテールが要るのと同じで、
                    AIも「他と見分けるためのピクセル・模様データ」が一定量そろわないと確信できません。
                    ただし<b>その境界線（クロスオーバー）は人間とAIで大きくズレる</b>ことがあります。
                    これは、AIの「認識」が概念的な理解ではなく、あくまで
                    <b>ピクセル模様のパターンマッチング</b>である、という本質を示しています。</p>
                    </div>
                    """, unsafe_allow_html=True)

                # 達成度（3枚以上ためしたらお祝い）
                tried_count = len(st.session_state["vision_tried_images_vch"])
                st.caption(
                    f"チャレンジした画像の種類: {tried_count} / {len(_vch_available)}　"
                    f"（総試行回数: {st.session_state['vision_stage_attempts']}）"
                )
                if tried_count >= 3:
                    st.success(
                        f"達成！ {tried_count}種類の被写体でAIと直感を比べました。"
                        "被写体によって『わかる境界』が変わるのを体感できましたか？",
                    )

# ================================================================
# SECTION 12: フッター
# ================================================================
st.markdown("""
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 01: AIの目 — 画像認識の仕組みを解き明かせ</p>
</div>
""", unsafe_allow_html=True)
