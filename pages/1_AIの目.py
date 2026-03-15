import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import base64
import hashlib
# --- セキュリティバイデザイン：アクセス制御と保護 ---
def protect_data(data_string):
    """
    入力データをSHA-256でハッシュ化し、元の値を推測不能にする
    """
    # ソルト（調味料）を加えて、レインボーテーブル攻撃（辞書攻撃）を防ぐ
    salt = "ai_vision_lab_2026_secure_key"
    hash_object = hashlib.sha256((data_string + salt).encode())
    return hash_object.hexdigest()

# 使用例：ユーザーが入力したIDや名前をハッシュ化して保存する
if "user_id" in st.session_state:
    # そのまま保存せず、ハッシュ化した値のみをシステムで扱う
    secure_id = protect_data(st.session_state.user_id)
# 1. ユーザー権限を「閲覧者」に完全固定（セッション乗っ取り対策）
if "user_role" not in st.session_state:
    st.session_state.user_role = "viewer"

# 2. URLパラメータによる不正な操作（管理者昇格など）を遮断
if st.query_params.to_dict():
    # パラメータが一つでもある場合は警告を出して停止（または無視）
    # 特定の安全なパラメータ以外を拒否する仕様
    st.error("不正なアクセスを検知しました。URLを直接書き換えないでください。")
    st.stop()

# 3. JavaScriptによるフロントエンド保護（クリックジャッキング対策）
st.markdown("""
    <script>
        // 他サイトのiframe内での表示を強制解除（サイトのなりすまし防止）
        if (window.top !== window.self) {
            window.top.location = window.self.location;
        }
        // 全ての外部リンクに noopener/noreferrer を付与（リンク先からの情報漏洩防止）
        document.querySelectorAll('a').forEach(link => {
            link.setAttribute('rel', 'noopener noreferrer');
        });
    </script>
""", unsafe_allow_html=True)
def get_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

def judge_with_confidence(score):
    if score > 85:
        return "高い確信があります。", "success"
    elif score > 60:
        return "おそらくそうだと思います。", "info"
    elif score > 35:
        return "自信がありません。", "warning"
    else:
        return "分かりません。", "error"

def predict_image(img_array):
    img_resized = cv2.resize(img_array, (224,224))
    img_pre = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))
    preds = model.predict(img_pre, verbose=0)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = float(decoded[2]) * 100
    return label, confidence

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSSファイルが見つかりません: {file_name}")

st.set_page_config(layout="wide")
# =============================
# 1. ページ基本設定
# =============================
st.markdown(f"""
    <div class="main-title-container">
        <h1 class="main-title-text">AIの目</h1>
        <p class="sub-title-text">画像認識の仕組みからAIの思考を覗いてみよう</p>
    </div>
""", unsafe_allow_html=True)
# =============================
# 2. 関数・パス設定
# =============================
def create_dummy_image(text, color):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, text, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    return Image.fromarray(img)

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "data")
assets_dir = os.path.join(parent_dir, "assets")
css_file = os.path.join(assets_dir, "style.css")

# CSSファイルを読み込む
local_css(css_file)

label_map = {"bird.jpg": "🐦 とり", "cat.jpg": "🐈 ねこ", "hyousiki.jpg": "🚦 標識","building.jpg": "🏢 建物","mountain.jpg": "🏔 山","soccerball.jpg": "⚽ ボール"}
samples = {}
try:
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        for fn, dn in label_map.items():
            if fn in files: samples[dn] = os.path.join(data_dir, fn)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f not in label_map:
                samples[f"📂 {f}"] = os.path.join(data_dir, f)
    if not samples: samples["🐦 とり"] = os.path.join(data_dir, "bird.jpg")
except: pass

# --- セッション状態の初期化 ---
if "img_path" not in st.session_state:
    st.session_state["img_path"] = list(samples.values())[0]
if "use_upload" not in st.session_state:
    st.session_state["use_upload"] = False

# =============================
# 3. サイドバー（同期型）
# =============================
with st.sidebar:
    st.header("⚙️ 設定・画像選択")
    # ソース選択をセッション連動
    src_idx = 1 if st.session_state["use_upload"] else 0
    img_source = st.radio("画像のソース", ["サンプルから選ぶ", "画像をアップロード"], index=src_idx)
    
    if img_source == "サンプルから選ぶ":
        st.session_state["use_upload"] = False
        # 現在のパスから名前を逆引きして初期値にする
        current_name = [k for k, v in samples.items() if v == st.session_state["img_path"]]
        selected_name = st.selectbox("画像を選択", list(samples.keys()), 
                                     index=list(samples.keys()).index(current_name[0]) if current_name else 0)
        st.session_state["img_path"] = samples[selected_name]
    else:
        st.session_state["use_upload"] = True
    
    st.divider()
    st.info("💡 **ヒント**　AIはピクセルごとの「数値」を見ています。")
# =============================
# 4. メインコンテンツ
# =============================
st.header("👁️ AIの目：多層可視化ラボ")
st.markdown("""
<div class="explanation-box">
AIが画像を認識するプロセスは、魔法ではありません。
<b>「光の分解 (RGB)」→「輪郭の抽出 (Edge)」→「特徴の発見 (Filter)」</b> という数理的な手順を踏みます。
さあ、AIの網膜（Retina）の裏側を覗いてみましょう。
</div>
""", unsafe_allow_html=True)

with st.container():
    st.header("STEP 1: AIに見せる画像を選ぼう！")
    
    # 画像1に対して、右側の操作エリアを2の比率で分割
    col_img_view, col_controls = st.columns([1, 2])
    
    # --- 最終的な画像決定ロジック (先に計算) ---
    if st.session_state["use_upload"] and st.session_state.get("uploaded_file"):
        image = Image.open(st.session_state.get("uploaded_file"))
    else:
        image = Image.open(st.session_state["img_path"]) if os.path.exists(st.session_state["img_path"]) else create_dummy_image("No Image", (100,100,100))
    
    image = image.convert("RGB")
    
    # 左側：画像表示
    with col_img_view:
        # アスペクト比を維持して表示
        st.image(image, caption="現在の入力画像", use_container_width=True)
        st.caption(f"サイズ: {image.width}x{image.height}px")

    # 右側：ボタンとアップローダー
    with col_controls:
        tab_sample, tab_upload = st.tabs(["サンプルから選ぶ", "自分の画像を使う"])
        
        with tab_sample:
            # 2列×3行のグリッドを作成
            labels = list(samples.keys())
            # ボタンを最大6個まで表示
            btn_grid = [st.columns(2) for _ in range(3)] 
            
            for idx in range(min(6, len(labels))):
                row = idx // 2
                col = idx % 2
                if btn_grid[row][col].button(labels[idx], key=f"main_btn_{idx}", use_container_width=True):
                    st.session_state["img_path"] = samples[labels[idx]]
                    st.session_state["use_upload"] = False
                    st.rerun()
        
        with tab_upload:
            uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "png", "jpeg"], key="main_up")
            if uploaded_file:
                st.session_state["uploaded_file"] = uploaded_file
                st.session_state["use_upload"] = True
                # アップロード直後に反映させるための処理
                image = Image.open(uploaded_file).convert("RGB")
# =============================
# メイン処理エリア
# =============================
if image:
    st.header("STEP 2: AIの視覚を体験しよう！興味のあるタブを選んでみよう！")
    # タブ設定
    tab1, tab2, tab3, tab4 = st.tabs([
        "① 数字の世界 (RGB)", 
        "② 輪郭の世界 (Edge)", 
        "③ フィルタの世界 (CNN)",
        "④ AIの弱点 (Noise)"
    ])
    image = image.resize((600, int(600 * image.height / image.width)))
    img_array = np.array(image)
    # --------------------------
    # TAB 1: 画像は数字である 
    # --------------------------
    with tab1:
        st.header("🧮 画像の正体は『数字の集まり？』")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            col_main1, col_main2 = st.columns([1, 2])
            with col_main1:
                img_path = "data/rgb.jpg"
                img_base64 = ""
                if os.path.exists(img_path):
                    img_base64 = get_image_as_base64(img_path)
                    if img_base64:
                        st.markdown(f"""
                            <div style="border: 1px solid #ccc; border-radius: 8px; overflow: hidden; background-color: white;">
                                <img class="html-zoom-img" src="data:image/png;base64,{img_base64}" style="width: 100%; display: block;">
                            </div>
                            <p style="text-align: center; color: #666; font-size: 0.8em; margin-top: 5px;">
                                【図：加法混色】RGBの光の強さの組み合わせで色が決まる
                            </p>
                        """, unsafe_allow_html=True)
            with col_main2:
                st.markdown("""
                <div class="explanation-box">
                <h3 style="font-size: 1.05rem;">1. 画像は「小さな点」の集合体</h3>
                私たちがスマホで見る綺麗な写真は、実は「ピクセル」と呼ばれる気が遠くなるほど小さな点の集まりです。
                その一つ一つの点は、さらに<b>「赤(R)・緑(G)・青(B)」</b>という3つの光の強さを組み合わせて作られています。

                <h3 style="font-size: 1.05rem;">2. コンピュータが見ているのは「光」ではなく「数字」</h3>
                コンピュータは人間のように「色」として画像を見ることができません。
                その代わり、各ピクセルの光の強さを<b>「0から255までの数字」</b>として処理しています。
                
                R・G・Bの3つがそれぞれ256通り（0〜255）の強さを持つため、その組み合わせは 
                <b>256 × 256 × 256 ＝ 16,777,216通り</b> にも及びます。
                デジタル画像とは、この約1,677万色の中から選ばれた数字が、タイル状に並んだ「巨大な数字の表」なのです。

                <h3 style="font-size: 1.05rem;">3. AIはどうやって物体を見つけている？</h3>
                AIはこの膨大な数字の羅列をスキャンし、「この数字の並び方はリンゴの形だ」「この色の変化は猫の耳だ」といった<b>特定のパターン</b>を瞬時に見つけ出します。
                
                人間にはただの色の重なりに見えるものでも、AIにとっては<b>「計算可能なデータの塊」</b>だからこそ、複雑な画像の中から特定の物体を正確に、そして人間以上のスピードで識別することができるのです。
                </div>
                """, unsafe_allow_html=True)
            st.divider()
        # ---------------------------------------------------------
        # 画像とデータの表示エリア
        # ---------------------------------------------------------
        st.header("STEP 3: 実際にAIに見えているデータを覗いてみよう！")
        # 枠で囲みたい範囲を st.container で包む
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            col_img, col_data = st.columns([3,2])
            with col_img:
                st.header("RGBレイヤー分解して拡大してみよう")
                st.markdown("画像を「赤・緑・青」の成分だけに分けてみましょう。")
                original, rgb_layer = st.columns([1,1])
                with original:
                    st.markdown(f"▼ 元画像 ")
                    st.image(image, caption="三色の組み合わせ画像", width=250)
                with rgb_layer:
                    st.markdown(f"▼ 成分画像")
                    target_vis_placeholder = st.empty()

            with col_data:
                st.markdown("<br>" * 1, unsafe_allow_html=True)
                st.info("AIはこの数字の変化（グラデーション）を計算して、物体の形を判断します。")
                # --- 2. 分解する色の選択 ---
                channel = st.radio(
                    "✂️ 分解する色を選択",
                    ["Red", "Green", "Blue"],
                    horizontal=True,
                    key="rgb_selector_fixed"
                )

                # --- 3. 調査ポイントの操作 ---
                st.markdown("##### 🔍 調査ポイントを動かす")
                col_x, col_y = st.columns(2)
                pick_x = col_x.slider("横の位置 (X)", 0, image.width - 1, int(image.width/2), key="slider_x")
                pick_y = col_y.slider("縦の位置 (Y)", 0, image.height - 1, int(image.height/2), key="slider_y")

                # --- 4. データの準備と計算 ---
                r_ch = img_array[:,:,0]
                g_ch = img_array[:,:,1]
                b_ch = img_array[:,:,2]
                zeros = np.zeros_like(r_ch)

                if channel == "Red":
                    target_data = r_ch
                    target_vis = np.stack([r_ch, zeros, zeros], axis=2)
                    cmap_style = "Reds"
                elif channel == "Green":
                    target_data = g_ch
                    target_vis = np.stack([zeros, g_ch, zeros], axis=2)
                    cmap_style = "Greens"
                else:
                    target_data = b_ch
                    target_vis = np.stack([zeros, zeros, b_ch], axis=2)
                    cmap_style = "Blues"
                
                target_vis_placeholder.image(target_vis, caption=f"明るい場所＝{channel}が強い", width=250)

                # --- 6. 数値の取得とメトリック表示 ---
                if len(img_array.shape) == 3:
                    # RGBA画像などの場合でも最初の3チャンネル(RGB)のみ取得
                    pix_r, pix_g, pix_b = img_array[pick_y, pick_x, :3]
                else:
                    pix_r = pix_g = pix_b = img_array[pick_y, pick_x]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 Red", pix_r)
                m2.metric("🟢 Green", pix_g)
                m3.metric("🔵 Blue", pix_b)

                st.markdown(f"**▼ 周辺の数値データ** ({channel}チャンネル)")
                
                zoom_radius = 4
                y_start = max(0, pick_y - zoom_radius)
                y_end = min(target_vis.shape[0], pick_y + zoom_radius + 1)
                x_start = max(0, pick_x - zoom_radius)
                x_end = min(target_vis.shape[1], pick_x + zoom_radius + 1)
                
                zoom_area = target_vis[y_start:y_end, x_start:x_end]
                zoom_disp = cv2.resize(zoom_area, (250, 250), interpolation=cv2.INTER_NEAREST)

            # --- 7. 下段のズーム表示エリア（ここもコンテナの中なので枠に入ります） ---
            col_zoom1, col_zoom2 = st.columns([3,2])
            zoom_area_orig = img_array[y_start:y_end, x_start:x_end]
            zoom_disp_orig = cv2.resize(zoom_area_orig, (250, 250), interpolation=cv2.INTER_NEAREST)

            with col_zoom1:
                c1, c2 = st.columns(2)
                c1.image(zoom_disp_orig, caption="選択範囲(元)", width=250)
                c2.image(zoom_disp, caption="選択範囲(成分)", width=250)

            with col_zoom2:
                if zoom_area.ndim == 3:
                    ch_idx = 0 if channel == "Red" else 1 if channel == "Green" else 2
                    zoom_single = zoom_area[:, :, ch_idx]
                else:
                    zoom_single = zoom_area

                zoom_single = zoom_single.astype(np.int32)
                max_val = np.max(zoom_single)
                max_pos_local = np.unravel_index(np.argmax(zoom_single), zoom_single.shape)
                
                col_labels = list(range(x_start, x_end))
                row_labels = list(range(y_start, y_end))
                max_x, max_y = col_labels[max_pos_local[1]], row_labels[max_pos_local[0]]
                
                df_subset = pd.DataFrame(zoom_single, columns=col_labels, index=row_labels)

                st.table(
                    df_subset.style.background_gradient(cmap=cmap_style, axis=None, vmin=0, vmax=255)
                    .highlight_max(axis=None, props='color: white; font-weight: bold; background-color: #FF4B4B;')
                    .format("{:d}")
                )
                
                st.write(f"🎯 **中心**:({pick_x}, {pick_y}) | 🌟 **最大輝度**:({max_x}, {max_y}) [値: {max_val}]")
                if max_val == 255:
                    st.success("✨ ビンゴ！一番明るい点を発見！")
    # --------------------------
    # TAB 2: エッジ検出 & ノイズ除去
    # --------------------------
    with tab2:
        st.header("📐 形を理解する（エッジ検出）")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="explanation-box">
            <p>本プログラムで「エッジ（輪郭）」の抽出を重視した理由は、それが<b>画像認識における『情報の圧縮と抽象化』の第一歩</b>だからです。</p>
            AIはエッジを探す前に、ノイズを消す(画像の中から重要な物体を探すために周りの情報量を減らす)ために画像を<b>「ぼかす」</b>ことがよくあります。<br>
            2つの有名なアルゴリズム（CannyとLaplacian）を切り替えて、その違いを体験しましょう！
            </div>
            """, unsafe_allow_html=True)
            edge_mode = st.radio(
                "アルゴリズムを選択",
                ["Canny法", "ラプラシアンフィルタ"],
                key="edge_mode_radio"
            )

            # 2. 設定エリア
            st.write("🔧 **AIフィルタ設定**")
            c_set1, c_set2, c_set3 = st.columns(3)
            
            # ガウシアンフィルタは共通
            blur_val = c_set1.slider("ガウシアンフィルタ(ぼかし)", 1, 15, 3, step=2)
            
            # 【重要】条件分岐の文字列をラジオボタンの選択肢と完全に一致させました
            if edge_mode == "Canny法":
                th1 = c_set2.slider("感度:Min", 0, 255, 100)
                th2 = c_set3.slider("感度:Max", 0, 255, 200)
            else:
                # Laplacianモード
                lap_ksize = c_set2.slider("ラプラシアンフィルタ(エッジ検出)", 1, 7, 3, step=2)
                st.caption("※ラプラシアンは境界の『変化の大きさ』を計算します。")

            # --- 画像処理の実装 ---
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # 前処理：共通のガウスぼかし
            blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
            
            if edge_mode == "Canny法":
                # Cannyアルゴリズム
                processed_edges = cv2.Canny(blurred, th1, th2)
            else:
                # Laplacianアルゴリズム
                lap_raw = cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_ksize)
                processed_edges = cv2.convertScaleAbs(lap_raw)

            # 3. 画像表示エリア
            col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
            
            with col_res1:
                st.image(image, caption="1. 元画像", width=300)
            with col_res2:
                st.image(blurred, caption="2. ぼかし後", width=300)
            with col_res3:
                st.image(processed_edges, caption=f"3. {edge_mode} 結果", width=300)

            # 教材メッセージ
            st.markdown(
                    """
                <div class="explanation-box">
                    <h3>🔧 ガウシアンフィルタ(ぼかし)</h3>
                    値を大きくすると、画像がよりぼやけます。画像がぼやけると、<br>はっきりした線や物体の境目を見つけやすく、逆に細かい線や薄い線、背景などは消えてしまいます。<br>
                    AIはこのぼかしを使うことでその画像において重要な「物体」を発見しやすくなるというわけです。その後に線を残すか削除するかの処理をします。Cannyはノイズに強く綺麗な線を、Laplacianは色の変化をダイレクトに抽出します。下の解説も見てみましょう。
                </div>
                    """, 
                    unsafe_allow_html=True
            )
            if edge_mode == "Canny法":
                st.markdown(
                    """
                    <div class="explanation-box">
                        <h3>💡 Cannyエッジ検出の仕組み</h3>
                        「感度」とは、画像の色の変化をどれくらい厳しくチェックして「線」と認めるかの基準です。<br>
                        理論上、<b>Min : Max = 1 : 2 または 1 : 3</b> の比率にすると、最も綺麗に線がつながりやすいと言われています。
                        <div style="margin-top: 15px; border-top: 1px solid #e0e0e0; padding-top: 10px;">
                            <p>● <b>確実なエッジ（Max以上）</b>：変化が激しい場所。誰が見ても「線」なので、無条件で採用されます。</p>
                            <p>● <b>迷い中のエッジ（Min 〜 Maxの間）</b>：変化が中くらい。「確実なエッジ」と繋がっていれば採用され、孤立していれば無視されます。</p>
                            <p>● <b>ノイズ（Min以下）</b>：変化が緩やか。線ではないと見なされ、無視されます。</p>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="explanation-box">
                        <h3>🔮 ラプラシアンフィルタの仕組み</h3>
                        「エッジ（境界線）」を見つけるための、数学的な「2階微分」を利用したフィルタです。<br>
                        色の明るさが急激に切り替わる地点を、プラスとマイナスの大きな変化として捉えます。
                        <div style="margin-top: 15px; border-top: 1px solid #e0e0e0; padding-top: 10px;">
                            <p>● <b>一方向ではなく「全方向」</b>：<br>横や縦だけでなく、周囲360度の変化を一度に計算するため、輪郭の「点」や「線」が強調されます。</p>
                            <p>● <b>ノイズに敏感</b>：<br>非常に鋭い感度を持っているため、画像にザラつき（ノイズ）があると、それもエッジとして拾いすぎてしまう性質があります。</p>
                            <p>● <b>AIの「下書き」</b>：<br>そのままでは線が細すぎることもありますが、画像をくっきりさせる「シャープネス処理」のベースとしてよく使われます。</p>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        # --- ギャラリー設定 ---
        st.header("🖼️ ギャラリー：AIの視点プロセス完全図解")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)# UI layout consistency check
            gallery_size = 350  # 画像サイズを350に固定（スライダー削除）
            if os.path.exists(r"C:\Python practice\Antigravity"):
                path = r"C:\Python practice\Antigravity"
            else:
                path = "data"

            steps_row1 = [
                {"file": f"{path}/ori.png",  "badge": "STEP 1", "title": "入力：元画像"},
                {"file": f"{path}/mono.png", "badge": "STEP 2", "title": "変換：モノクロ"},
                {"file": f"{path}/blur.png", "badge": "STEP 3", "title": "除去：ぼかし"}
            ]

            steps_row2 = [
                {"file": f"{path}/edge.png", "badge": "STEP 4", "title": "抽出：エッジ"},
                {"file": f"{path}/ans.png",  "badge": "TARGET", "title": "理想：正解データ"},
                {"file": f"{path}/dif.png",  "badge": "RESULT", "title": "判定：一致率"}
            ]

            # --- 横方向の矢印表示ヘルパー ---
            def show_row_arrow():
                st.markdown('<div class="row-arrow">➡</div>', unsafe_allow_html=True)


            # ==========================================
            # 1段目：前処理フェーズ
            # ==========================================
            st.subheader("Phase 1: 情報を削ぎ落とす")
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            cols_1 = st.columns([10, 2, 10, 2, 10])

            for i, col_index in enumerate([0, 2, 4]):
                with cols_1[col_index]:
                    item = steps_row1[i]
                    st.markdown(f"""
                    <div class="img-frame">
                        <span class="step-badge">{item['badge']}</span>
                        <div class="caption-text">{item['title']}</div>
                    </div>""", unsafe_allow_html=True)
                    try: st.image(item["file"], width=gallery_size)
                    except: st.error("画像なし")

                # 矢印の配置 (画像と画像の間にのみ配置)
                if col_index != 4:
                    with cols_1[col_index + 1]:
                        show_row_arrow()


            # ==========================================
            # 折り返し（3枚目 → 4枚目の接続）
            # ==========================================
            # 太く赤いSVG矢印を使用して、1本目は右側、2本目は左側に下にずらして配置
            st.markdown("""
            <div style="margin-top: 30px; display: flex; flex-direction: column;">
                <div style="text-align: right; padding-right: 5%;">
                    <div class="show-arrow-svg">
                        <svg width="60" height="60" viewBox="0 0 100 100">
                            <path d="M 20,10 Q 80,10 80,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                            <polygon points="65,55 80,85 95,55" fill="#D32F2F"/>
                        </svg>
                    </div>
                </div>
                <div style="text-align: center; color: black; font-weight: bold; background-color: #c0c0c0; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    🌀 情報を整理したので、ここから「形」を取り出します 🌀
                </div>
                <div style="text-align: left; padding-left: 5%;">
                    <div class="show-arrow-svg">
                        <svg width="60" height="60" viewBox="0 0 100 100">
                            <path d="M 80,10 Q 20,10 20,60" fill="none" stroke="#D32F2F" stroke-width="12" stroke-linecap="round"/>
                            <polygon points="5,55 20,85 35,55" fill="#D32F2F"/>
                        </svg>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # ==========================================
            # 2段目：認識・評価フェーズ
            # ==========================================
            st.subheader("Phase 2: 形を見極める")
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            cols_2 = st.columns([10, 2, 10, 2, 10])

            # 画像4 (STEP 4)
            with cols_2[0]:
                item = steps_row2[0]
                st.markdown(f"""
                <div class="img-frame frame-phase2">
                    <span class="step-badge badge-phase2">{item['badge']}</span>
                    <div class="caption-text">{item['title']}</div>
                </div>""", unsafe_allow_html=True)
                try: st.image(item["file"], width=gallery_size)
                except: st.error("画像なし")

            with cols_2[1]: show_row_arrow()

            # 画像5 (TARGET)
            with cols_2[2]:
                item = steps_row2[1]
                st.markdown(f"""
                <div class="img-frame frame-phase2">
                    <span class="step-badge badge-phase2">{item['badge']}</span>
                    <div class="caption-text">{item['title']}</div>
                </div>""", unsafe_allow_html=True)
                try: st.image(item["file"], width=gallery_size)
                except: st.error("画像なし")

            with cols_2[3]: show_row_arrow()

            # 画像6 (RESULT)
            with cols_2[4]:
                item = steps_row2[2]
                st.markdown(f"""
                <div class="img-frame frame-result">
                    <span class="step-badge badge-result">{item['badge']}</span>
                    <div class="caption-text" style="color: #c0392b;">{item['title']}</div>
                </div>""", unsafe_allow_html=True)
                try: st.image(item["file"], width=gallery_size)
                except: st.error("画像なし")        
                st.caption("※一致率が95%を超えると、AIは「完全な認識」として学習を完了します。")
        st.header("画像認識の基礎")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            ans_path = os.path.join(path, "karasu.png")

            try:
                if not os.path.exists(ans_path):
                    st.error(f"ファイルが見つかりません: {ans_path}")
                else:
                    # 1. 画像の準備
                    target_color = cv2.imread(ans_path)
                    target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB)
                    target_gray = cv2.cvtColor(target_color, cv2.COLOR_RGB2GRAY)

                    # 【修正ポイント】pro.jpg を読み込み、サイズ取得のエラーを回避
                    edge_insert = os.path.join(path, "pro.jpg")
                    processed_edges = cv2.imread(edge_insert)
                    
                    if processed_edges is None:
                        st.error(f"ファイルが読み込めません: {edge_insert}")
                    else:
                        # .shape[:2] とすることで、カラー画像でも(height, width)だけを確実に取得
                        height, width = processed_edges.shape[:2]
                        
                        target_color = cv2.resize(target_color, (width, height))
                        target_gray = cv2.resize(target_gray, (width, height))

                        # 2. 【緑：正解のターゲット】の作成
                        y_true_edges = cv2.Canny(target_gray, 200, 300)
                        kernel = np.ones((3, 3), np.uint8)
                        # ターゲット（緑）の許容範囲
                        y_true_zone = cv2.dilate(y_true_edges, kernel, iterations=1) 

                        # 3. 【赤：AIエッジ抽出】の作成
                        # カラーで読み込んでいる可能性を考慮してグレースケール変換してから二値化
                        processed_gray = cv2.cvtColor(processed_edges, cv2.COLOR_BGR2GRAY)
                        _, y_pred_binary = cv2.threshold(processed_gray, 100, 255, cv2.THRESH_BINARY)
                        
                        # ★黄色と赤を強く見せるための「表示用」膨張処理
                        y_pred_for_vis = cv2.dilate(y_pred_binary, kernel, iterations=1)
                        y_true_edges_for_vis = cv2.dilate(y_true_edges, kernel, iterations=1)

                        # 4. 【可視化レイヤーの合成】
                        # --- 左：AIエッジ検証画像 ---
                        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
                        vis_img[:, :, 1] = (y_true_zone > 0) * 255   # 緑：ターゲット
                        vis_img[:, :, 0] = (y_pred_for_vis > 0) * 255 # 赤：AIエッジ（太らせて黄色を強化）
                        
                        # --- 右：理想の一致画像 ---
                        ideal_img = np.zeros((height, width, 3), dtype=np.uint8)
                        ideal_img[:, :, 1] = (y_true_zone > 0) * 255 # 緑：ターゲット
                        ideal_img[:, :, 0] = (y_true_edges_for_vis > 0) * 255 # 赤：理想の芯（太らせて視認性UP）

                        # --- 追加：純粋な赤エッジ画像（pro.jpg由来） ---
                        red_only_img = np.zeros((height, width, 3), dtype=np.uint8)
                        red_only_img[:, :, 0] = (y_pred_for_vis > 0) * 255

                        # 5. IoU（一致率）の計算
                        y_pred_bool = y_pred_binary > 0
                        y_true_bool = y_true_zone > 0
                        intersection = np.logical_and(y_pred_bool, y_true_bool)
                        union = np.logical_or(y_pred_bool, y_true_bool)
                        new_iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0
                        col_ans1, col_ans2,col_ans3 = st.columns(3)
                        with col_ans1:
                            st.markdown("**元画像**")
                            st.image(ans_path, caption="処理前画像", width=300)
                        with col_ans2:
                            st.markdown("**🔍 重ね合わせ検証**")
                            st.image(vis_img, caption="🔴エッジ検出 🟢正解領域（甘め） 🟡的中", width=300)
                        with col_ans3:
                            st.markdown("**🎯 理論上の100%**")
                            st.image(ideal_img, caption="理想的な重なり",width=300)
                        st.warning("""参考：中央画像のエッジ的中率（黄色の重なりの割合）は「2.7%」でした。これにより、赤と緑の画像は同じではないと分かります。私たち人間にとっては同じに見えても、コンピュータにとってはこのような手順を踏む必要があります。""")
                        st.markdown("<br>" * 2, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="explanation-box">
                            <h3>1. 情報を「削ぎ落とす」という知性</h3>
                            <p>コンピュータにとって、カラー画像は膨大な数値データの集まりであり、そのままでは情報量が多すぎて「何が写っているか」を判断できません。本コードで行っている処理は、色や質感といった副次的な情報をあえて捨て去り、<b>物体の「構造（形）」だけを取り出す作業</b>です。この「形を定義する」プロセスこそが、あらゆるコンピュータビジョンの基盤となります。</p>
                            <h3>2. 自動運転技術への繋がりと、本実装の「位置づけ」</h3>
                            <p>自動運転における映像認識も、根本はこの「エッジ抽出」から始まります。走行中の車載カメラは、白線、ガードレール、歩行者の輪郭を一瞬で捉え、自車の位置や障害物を識別します。ただし、今回のコードはあくまで<b>「基礎中の基礎（基盤）」</b>です。実際の自動運転や高度なAI認識には、ここからさらに以下のステップが必要となります。</p>
                            <ul>
                                <li><b>特徴量の抽出：</b> 抽出したエッジ(物の輪郭のようなイメージ)から、それが「円形（タイヤ）」なのか「直線（白線）」なのかを幾何学的に特定する。</li>
                                <li><b>パターンマッチングと学習：</b> 膨大なデータベースと照らし合わせ、そのエッジの集合体が「歩行者」である確率を推論する（ディープラーニング）。</li>
                                <li><b>時系列解析：</b> 静止画ではなく動画として、エッジがどちらの方向に動いているか（オプティカルフロー）を計算する。</li>
                            </ul>
                            <h3>3. この実験が示す「本質的な価値」</h3>
                            <p>実装では、異なる画像同士を重ねた際の一致率が極めて低くなることを示しました。これは、<b>「人間には同じ動物や物体に見えても、デジタルデータとしては全くの別物である」</b>という客観的な事実を突きつけています。この「基礎の基礎」を理解せずして、高度なAIモデルを扱うことはできません。エッジという最小単位のデータを正確に制御し、比較し、その差異を定義する。この地味ながらも厳格なプロセスこそが、世界を正しく認識しようとするコンピュータ・サイエンスの誠実な入り口なのです。</p>
                            <p>自動運転というのは非常に難しいことが分かります。映像を判別することは私たちが想像している以上に「計算量」が多く、「慎重かつ正確」でなければなりません。
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
    #---------------------------        
        # TAB 3: 畳み込みフィルタ
    # --------------------------
    with tab3:
        st.header("🕶️ 特徴を見つける眼鏡（畳み込みフィルタ）")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            # --- 1. データの準備 ---
            img_path = "data/lasta.png"
            img_base64 = ""
            if os.path.exists(img_path):
                img_base64 = get_image_as_base64(img_path)

            # --- 2. カラムを使って「左に画像」「右にテキスト」を配置 ---
            col_view, col_text = st.columns([2, 3])  

            with col_view:
                if img_base64:
                    # 画像をHTMLで表示（サイズをカラムに合わせる）
                    st.markdown(f"""
                        <div style="border: 1px solid #ccc; border-radius: 8px; overflow: hidden; background-color: white;">
                            <img class="html-zoom-img" src="data:image/png;base64,{img_base64}" style="width: 100%; display: block;">
                        </div>
                        <p style="text-align: center; color: #666; font-size: 0.8em; margin-top: 5px;">
                            【図：ラスタスキャン】左上から1マスずつ計算
                        </p>

                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ 解説画像が見つかりません。")
            with col_text:
                st.markdown("""
                    <div class="explanation-box">
                        <h3>🔍 AIはどうやって画像を見る？</h3>
                        <p>AIは画像を一瞬で「意味」として理解しているわけではありません。<br>
                        左の図のように、<span class="highlight-text">「3×3マスの小さな窓（フィルタ）」</span>を、左上から1マスずつスライドさせながら計算しています。</p>
                        <p>この処理では、各ピクセル周辺の「色の変化」や「差分」を数値として測っています。このように画像を順番に走査していく方式を、専門用語で<b>「ラスタスキャン」</b>と呼びます。</p></p>
                        <h3>💡 ここがポイント！</h3>
                        <p>窓の中にある「9個の数字」が、特定の形（縦線、横線、角など）に反応するように設定されています。
                        この計算を<span class="highlight-text">画像全体で何万回も繰り返す</span>ことで、AIは「これは猫の耳だ！」などと気づくことができるのです。</p>
                    </div>
                """, unsafe_allow_html=True)
        # ----------------------------
        # 1. プリセットフィルタ体験
        # ----------------------------
        st.header("1️⃣有名なフィルタ係数を見てみよう ")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                filter_type = st.selectbox(
                "かけてみる眼鏡（フィルタ）を選択",
                [
                    "恒等 (何もしない)",
                    "ぼかし (平均化/Mean)", 
                    "シャープ化 (Sharpen)", 
                    "輪郭抽出 (Laplacian)", 
                    "縦の輪郭 (Sobel X)", 
                    "横の輪郭 (Sobel Y)",
                    "エンボス (Emboss)"
                ]
                )
                # カーネル定義
                kernel = None
                desc = ""
                is_edge_filter = False # エッジ系は絶対値処理をするフラグ

                if filter_type == "恒等 (何もしない)":
                    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                    desc = "中央が1で他が0。つまり元の画素をそのまま出力します。「基本のキ」です。"

                elif filter_type == "ぼかし (平均化/Mean)":
                    # 5x5ではなく3x3で統一してわかりやすくします
                    kernel = np.ones((3, 3), np.float32) / 9
                    desc = "周囲9マスの色の平均をとります。CNNでは『プーリング（情報を間引く）』に近い役割を果たします。"

                elif filter_type == "シャープ化 (Sharpen)":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    desc = "真ん中を強く(5)、周囲を引く(-1)ことで、色の落差を強調し、クッキリさせます。"

                elif filter_type == "輪郭抽出 (Laplacian)":
                    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                    desc = "周囲との差分を計算します。CNNが『エッジ（形）』を学習する際、これに近い数値を自動で獲得することがよくあります。"
                    is_edge_filter = True

                elif filter_type == "縦の輪郭 (Sobel X)":
                    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    desc = "左右の差を計算します。つまり『縦線』がある場所だけが反応して光ります。"
                    is_edge_filter = True

                elif filter_type == "横の輪郭 (Sobel Y)":
                    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                    desc = "上下の差を計算します。つまり『横線』がある場所だけが反応して光ります。"
                    is_edge_filter = True

                elif filter_type == "エンボス (Emboss)":
                    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
                    desc = "斜めの光と影を作り出し、立体的な凸凹に見せます。"

                # フィルタ処理
                if is_edge_filter:
                    processed_raw = cv2.filter2D(img_array, cv2.CV_64F, kernel)
                    processed = cv2.convertScaleAbs(processed_raw)
                else:
                    processed = cv2.filter2D(img_array, -1, kernel)   
                
                st.markdown("<h3>⚙️ カーネル（計算式）</h3>", unsafe_allow_html=True)
                df_kernel = pd.DataFrame(kernel)
                # 見やすくフォーマット
                styled_kernel = df_kernel.style.format("{:.2f}" if filter_type == "ぼかし (平均化/Mean)" else "{:.0f}")
                st.write(styled_kernel)
                st.info(f"💡 {desc}")
            with c2:
                sub_col_left, sub_col_right = st.columns([0.2, 1]) 
                with sub_col_right:
                    st.image(processed, caption=f"【変換後】{filter_type} の世界", width=400)
            st.divider()
            st.warning("""
            🎓 **プロフェッショナルな豆知識：なぜ「左上」から？**
            
            実は、昔のWindows画像形式（BMP）は**「左下から右上」**に向かって走査していました。
            これは数学のグラフ（X軸・Y軸）が左下を原点(0,0)にしていた名残です。
            
            しかし、現在主流のJPEGやPNG、そしてこのアプリで使っているOpenCVは、本を読む順序と同じ**「左上から右下」**に走査します。
            歴史的な経緯で「スタート地点」が違うことがあるなんて、面白いですよね！
            """)
        # ==========================================
        st.header("🧪 **DIYラボ：自分だけのフィルタを作ろう！**")
        Lab_area = st.container()
        with Lab_area:
            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
            col_lab_input, col_lab_result = st.columns([1, 1])

            with col_lab_input:
                st.markdown("<h3>⚙️ カーネル行列の入力</h3>", unsafe_allow_html=True)
                # 3x3の入力フォームを作成
                k_custom = np.zeros((3, 3))
                
                # レイアウトを整えて配置
                c_r1_1, c_r1_2, c_r1_3 = st.columns(3)
                c_r2_1, c_r2_2, c_r2_3 = st.columns(3)
                c_r3_1, c_r3_2, c_r3_3 = st.columns(3)

                # 行ごとの入力
                with c_r1_1: k_custom[0,0] = st.number_input("0,0", value=0.0, step=1.0, key="k00", label_visibility="collapsed")
                with c_r1_2: k_custom[0,1] = st.number_input("0,1", value=-1.0, step=1.0, key="k01", label_visibility="collapsed")
                with c_r1_3: k_custom[0,2] = st.number_input("0,2", value=0.0, step=1.0, key="k02", label_visibility="collapsed")

                with c_r2_1: k_custom[1,0] = st.number_input("1,0", value=-1.0, step=1.0, key="k10", label_visibility="collapsed")
                with c_r2_2: k_custom[1,1] = st.number_input("1,1", value=5.0, step=1.0, key="k11", label_visibility="collapsed")
                with c_r2_3: k_custom[1,2] = st.number_input("1,2", value=-1.0, step=1.0, key="k12", label_visibility="collapsed")

                with c_r3_1: k_custom[2,0] = st.number_input("2,0", value=0.0, step=1.0, key="k20", label_visibility="collapsed")
                with c_r3_2: k_custom[2,1] = st.number_input("2,1", value=-1.0, step=1.0, key="k21", label_visibility="collapsed")
                with c_r3_3: k_custom[2,2] = st.number_input("2,2", value=0.0, step=1.0, key="k22", label_visibility="collapsed")

                st.caption("☝️ この数字を変えてみてください！")
                
                # オプション：絶対値処理をするかどうか
                use_abs = st.checkbox("結果を絶対値にする（チェックすると、「境目」を白く光らせることができる。）", value=False)

            with col_lab_result:
                # カスタム処理実行
                if use_abs:
                    processed_custom_raw = cv2.filter2D(img_array, cv2.CV_64F, k_custom)
                    processed_custom = cv2.convertScaleAbs(processed_custom_raw)
                else:
                    processed_custom = cv2.filter2D(img_array, -1, k_custom)
                
                sub_col_left, sub_col_right = st.columns([0.2, 1]) 
                with sub_col_right:
                    st.markdown("<br>" * 2, unsafe_allow_html=True)
                    st.image(processed_custom, caption="あなたの実験結果", width=400)

                
            st.success("""
            💡 **実験のヒント**: 
            ・中央だけ大きい値で他を0にすると、元画像のまま
            ・中央を大きく＋周囲を負にすると、周囲との差が拡大されるためコントラストが強まり、くっきりした画像になる（シャープ化）。
            ・9マスすべてを同じ値にすると平均化になり、ノイズが減る代わりに輪郭もなだらかになる（ぼかし）。
            ・十字方向（上下左右）だけ値を持たせると、その方向の変化に強く反応する。縦成分を強めれば横線が目立つ、横成分を強めれば縦線が目立つ。
            ・正と負の値を混ぜると「差分」を計算する形になり、隣接画素との変化量＝エッジ（輪郭）を強調できる。
            """)
    # --------------------------
    # TAB 4: AI弱点ラボ - 強化完全版（信頼度追跡修正版）
    # --------------------------
    with tab4:
        st.header("🧠 AI弱点ラボ")

        Lab_area = st.container()
        with Lab_area:

            st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)

            experiment = st.radio(
                "🔬 実験する弱点を選択してください：",
                ["🌪️ ノイズ耐性", "🔄 回転耐性", "🌗 コントラスト依存"],
                horizontal=True
            )

            base_label, base_conf = predict_image(img_array)

            # ==============================
            # 画像処理
            # ==============================
            test_img = img_array.copy()
            param_value = 0

            if "🌪️" in experiment:
                noise_level = st.slider("ノイズ強度 (標準偏差)", 0, 128, 0)
                param_value = noise_level
                if noise_level > 0:
                    img_float = img_array.astype(np.float32)
                    noise = np.random.normal(0, noise_level, img_array.shape)
                    test_img = np.clip(img_float + noise, 0, 255).astype(np.uint8)
                st.caption("高周波ノイズはCNNの特徴抽出を破壊します。")

            elif "🔄" in experiment:
                angle = st.slider("回転角度", 0, 180, 0)
                param_value = angle
                if angle > 0:
                    h, w = img_array.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    test_img = cv2.warpAffine(img_array, M, (w, h))
                st.caption("多くのCNNは回転不変性を完全には持ちません。")

            else:
                contrast = st.slider("コントラスト倍率", 0.1, 2.0, 1.0)
                param_value = contrast
                test_img = cv2.convertScaleAbs(img_array, alpha=contrast, beta=0)
                st.caption("低コントラストはエッジ情報を弱めます。")

            # ==============================
            # AI推論
            # ==============================
            img_resized = cv2.resize(test_img, (224,224))
            img_pre = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))
            preds = model.predict(img_pre, verbose=0)
            decoded = decode_predictions(preds, top=5)[0]

            current_top_label = decoded[0][1]

            target_conf = 0.0
            for d in decoded:
                if d[1] == base_label:
                    target_conf = float(d[2]) * 100
                    break

            display_conf = target_conf
            delta_conf = display_conf - base_conf

            message, level_msg = judge_with_confidence(display_conf)

            # ==============================
            # レイアウト
            # ==============================
            col_main_img, col_bar, col_trend = st.columns([1.3, 1, 1])

            # ------------------
            # 左：入力と結果
            # ------------------
            with col_main_img:
                st.subheader("👁️ AIへの入力画像")
                st.image(test_img, use_container_width=True)

                st.metric("予測クラス", current_top_label)
                st.metric(
                    "正解クラス維持率",
                    f"{display_conf:.2f}%",
                    f"{delta_conf:+.2f}%"
                )

                getattr(st, level_msg)(message)

                if display_conf < 10 and current_top_label != base_label:
                    st.error(f"⚠️ AIは「{current_top_label}」と誤認しています。")

            # ------------------
            # 中央：分布
            # ------------------
            with col_bar:
                st.subheader("📊 正解クラスの維持率")

                fig_bar, ax_bar = plt.subplots(figsize=(4,2))
                ax_bar.barh([base_label], [display_conf], color="#3a7bd5")
                ax_bar.set_xlim(0, 100)
                ax_bar.text(display_conf + 1, 0, f"{display_conf:.1f}%", va='center', fontweight='bold')
                fig_bar.patch.set_alpha(0)
                ax_bar.patch.set_alpha(0)
                st.pyplot(fig_bar)

                st.subheader("🧠 Top-5 分布")

                top5_labels = [d[1] for d in decoded][::-1]
                top5_scores = [float(d[2])*100 for d in decoded][::-1]

                fig_top5, ax_top5 = plt.subplots(figsize=(4, 4))
                bars = ax_top5.barh(top5_labels, top5_scores, color="#c3cfe2")

                for i, label_name in enumerate(top5_labels):
                    if label_name == base_label:
                        bars[i].set_color('#3a7bd5')

                ax_top5.set_xlim(0, 100)
                fig_top5.patch.set_alpha(0)
                ax_top5.patch.set_alpha(0)
                st.pyplot(fig_top5)

            # ------------------
            # 右：履歴
            # ------------------
            with col_trend:
                st.subheader("📈 信頼度推移")

                if "history" not in st.session_state:
                    st.session_state.history = []

                if "last_exp" not in st.session_state or st.session_state.last_exp != experiment:
                    st.session_state.history = []
                    st.session_state.last_exp = experiment

                if not st.session_state.history or st.session_state.history[-1]["param"] != param_value:
                    st.session_state.history.append({
                        "param": param_value,
                        "confidence": display_conf
                    })

                recent = st.session_state.history[-20:]
                conf_values = [h["confidence"] for h in recent]

                fig_line, ax_line = plt.subplots(figsize=(4, 4))
                ax_line.plot(conf_values, marker='o', color='#3a7bd5')
                ax_line.set_ylim(0, 105)
                ax_line.set_title(f"Target: {base_label}")
                fig_line.patch.set_alpha(0)
                ax_line.patch.set_alpha(0)
                st.pyplot(fig_line)

                st.markdown("**📝 現在の状態**")
                st.write(f"- 正解クラス: {base_label}")
                st.write(f"- 予測クラス: {current_top_label}")
                st.write(f"- 維持率: {display_conf:.2f}%")

            # ==============================
            # 教育解説
            # ==============================
            st.info(f"""
            🎓 **実験のポイント**

            AIが別の物体に確信を持つと、元の正解（{base_label}）の信頼度は相対的に低下します。

            - ノイズは「模様」と誤認されやすい
            - 回転は特徴の位置関係を崩す
            - 低コントラストはエッジ情報を弱める

            グラフが下がるほど、AIが対象を見失っている状態です。
            """)

st.markdown(f"""
    <div class="custom-footer">
        <p>
            © 2026 <strong>AI Vision Lab.</strong> 
            | 画像認識の仕組みを楽しく学ぼう。
        </p>
        <p>
            Developed by <a href="#" target="_blank" rel="noopener noreferrer">My name</a> 
            | <a href="#">Contact</a>
        </p>
    </div>
""", unsafe_allow_html=True)

