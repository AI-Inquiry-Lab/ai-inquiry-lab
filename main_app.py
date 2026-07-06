import streamlit as st
import numpy as np
from datetime import datetime
import hashlib
import os
from utils.nav import render_top_nav
from utils.icons import svg_icon, icon_html, heading

# =====================================
# 1. ページ設定（※一番最初に記述必須！）
# =====================================
st.set_page_config(
    page_title="AI Inquiry Lab | トップ",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# 2. CSS読み込み処理（一本化＆絶対パス化）
# =====================================
@st.cache_data
def _read_css_text(file_path: str) -> str:
    # ファイル読み込みをキャッシュし、再実行のたびにディスクI/Oが発生しないようにする
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_css(file_name):
    # 実行ファイル(main_app.py)があるディレクトリを取得
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "assets", file_name)

    if os.path.exists(file_path):
        st.markdown(f'<style>{_read_css_text(file_path)}</style>', unsafe_allow_html=True)
    else:
        # 見つからない場合のデバッグ表示
        st.error(f"{file_name} が見つかりません。探索パス: {file_path}")

# assets/style.css を読み込む
load_css("style.css")

# =====================================
# 2b. トップナビゲーション
# =====================================
render_top_nav("home")

# =====================================
# 3. セキュリティバイデザイン・アーマー
# =====================================
# URLパラメータの監視
if st.query_params.to_dict():
    st.error("【Security Alert】 不正なパラメータを検知しました。")
    st.stop()

# セッション・セキュリティ
if "user_role" not in st.session_state:
    st.session_state.user_role = hashlib.sha256(b"viewer_role_secure_salt").hexdigest()

# ミッション踏破状況（このブラウザセッション内でのみ保持する簡易プログレス）
MISSION_KEYS = [
    "visited_vision", "visited_adversarial", "visited_training",
    "visited_llm", "visited_hardware", "visited_physical", "visited_rlgame",
    "visited_ml",
]
for k in MISSION_KEYS:
    if k not in st.session_state:
        st.session_state[k] = False

# フロントエンド保護
st.markdown("""
    <script>
        if (window.top !== window.self) { window.top.location = window.self.location; }
        document.querySelectorAll('a').forEach(link => { link.setAttribute('rel', 'noopener noreferrer'); });
    </script>
""", unsafe_allow_html=True)
# =====================================
# 4. カスタム・ナビゲーション（サイドバー）
# =====================================
with st.sidebar:
    st.markdown(
        f'<h2 style="color: #3b82f6; font-size: 1.2rem;">{svg_icon("zap", size=20)} AI CORE CONTROL</h2>',
        unsafe_allow_html=True,
    )

    # リアルタイム感のあるプログレスバー
    st.markdown(icon_html("dna", "**Neural Syncing...**", size=14), unsafe_allow_html=True)
    sync_rate = np.random.randint(88, 100)
    st.progress(sync_rate / 100)
    st.caption(f"Sync Rate: {sync_rate}% (STABLE)")

    st.markdown("---")

    # ハッシュ化されたアクセスキーの表示（視覚的インパクト）
    user_hash = hashlib.sha256(st.session_state.user_role.encode()).hexdigest()[:10].upper()
    st.markdown(f"""
        <div class="access-key-box">
            <span style="font-size: 0.7rem; color: #64748b;">ENCRYPTED ID</span><br>
            <span style="color: #3b82f6; font-weight: bold;">{user_hash}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<h3>{svg_icon("compass", size=16)} Navigation</h3>', unsafe_allow_html=True)
    st.page_link("main_app.py",                      label="司令室 (Home)")
    st.page_link("pages/1_vision.py",                label="M01: AIの目")
    st.page_link("pages/2_adversarial.py",           label="M02: AI騙し")
    st.page_link("pages/3_training.py",              label="M03: AI育成")
    st.page_link("pages/4_llm_mechanism.py",         label="M04: LLMの脳内")
    st.page_link("pages/5_cpu_gpu.py",               label="M05: CPU対GPU")
    st.page_link("pages/6_physical_digital_ai.py",   label="M06: AIの二形態")
    st.page_link("pages/7_rl_game.py",                label="M07: 育成ゲーム")
    st.page_link("pages/8_machine_learning.py",      label="M08: 機械学習の仕組み")

    st.markdown("---")
    completed = sum(st.session_state[k] for k in MISSION_KEYS)
    st.markdown(f"**{svg_icon('trophy', size=14, color='var(--color-yellow)')} 踏破ミッション: {completed} / {len(MISSION_KEYS)}**", unsafe_allow_html=True)
    st.progress(completed / len(MISSION_KEYS) if MISSION_KEYS else 0)

    st.success("SHIELD: ONLINE")
    # システム時刻の秒まで動的に表示（再読み込みのたびに更新）
    st.caption(f"Last Ping: {datetime.now().strftime('%H:%M:%S')}")
# =====================================
# 5. ヒーローセクション & ラボ・ガイド
# =====================================
st.markdown(f'<div class="hero-title">{svg_icon("brain", size=44)} AI Inquiry Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">数学 × データ × 思考実験で、AIのブラックボックスを解体する。</div>', unsafe_allow_html=True)

# セレクトボックスの代わりに、ラボの「提供価値」を3つのフェーズで紹介
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="guide-box">
            <h4>{svg_icon("search", size=18)} 解体</h4>
            <p>「魔法」に見えるAIを、数式とコードのレベルまでバラバラにして、その中身を覗き見ます。</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="guide-box">
            <h4>{svg_icon("flask", size=18)} 実験</h4>
            <p>理論を学ぶだけではありません。実際にパラメータをいじり、AIの挙動がどう変わるかを手元で体験します。</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="guide-box">
            <h4>{svg_icon("shield", size=18)} 掌握</h4>
            <p>ブラックボックスの正体を知ることで、AIを恐れるのではなく、使いこなすための「知の武器」を手に入れます。</p>
        </div>
    """, unsafe_allow_html=True)
# =====================================
# 6. ミッションボード（実験コンテンツ）
# =====================================
st.markdown(heading("Active Missions", "rocket", level=2), unsafe_allow_html=True)

MISSIONS = [
    {
        "icon": "eye", "title": "ミッション01: AIの目",
        "desc": "画像は単なる「数字の配列」に過ぎない。AIが世界を認識する最初のステップを体験せよ。",
        "badge": "RGB = (255, 128, 64)", "page": "pages/1_vision.py", "key": "btn_mission_1",
    },
    {
        "icon": "mask", "title": "ミッション02: AI騙し",
        "desc": "人間の目には見えない微小なノイズ。AIの脆弱性の仕組みを暴き、対抗的サンプルを体験せよ。",
        "badge": "f(x + ε) ≠ y", "page": "pages/2_adversarial.py", "key": "btn_mission_2",
    },
    {
        "icon": "dna", "title": "ミッション03: AI育成",
        "desc": "ニューロンを育て、ネットワークを組む。学習・過学習・勾配降下法をインタラクティブに体験せよ。",
        "badge": "Loss ↓ Accuracy ↑", "page": "pages/3_training.py", "key": "btn_mission_3",
    },
    {
        "icon": "message-circle", "title": "ミッション04: LLMの脳内",
        "desc": "ChatGPTのようなAIは「次の一語」を予測しているだけ。トークン化・注意機構・確率生成を体験せよ。",
        "badge": "P(次の単語 | 文脈)", "page": "pages/4_llm_mechanism.py", "key": "btn_mission_4",
    },
    {
        "icon": "cpu", "title": "ミッション05: CPU対GPU",
        "desc": "AIはなぜ「GPU」で学習するのか？ 1人の天才 vs 1000人の作業員、並列計算の威力を体感せよ。",
        "badge": "並列 ≫ 直列", "page": "pages/5_cpu_gpu.py", "key": "btn_mission_5",
    },
    {
        "icon": "footprints", "title": "ミッション06: AIの二形態",
        "desc": "画面の中のAIと、身体を持つAI。「強いAI」と「弱いAI」の違いを、対話とロボットの視点で理解せよ。",
        "badge": "弱いAI ≠ 強いAI", "page": "pages/6_physical_digital_ai.py", "key": "btn_mission_6",
    },
    {
        "icon": "gamepad", "title": "ミッション07: 育成ゲーム",
        "desc": "報酬と罰でAIエージェントを迷路の達人に育てろ。強化学習の試行錯誤をゲーム感覚で体験せよ。",
        "badge": "試行錯誤 → 賢さ", "page": "pages/7_rl_game.py", "key": "btn_mission_7",
    },
    {
        "icon": "cluster", "title": "ミッション08: 機械学習の仕組み",
        "desc": "「教えて学ばせる」教師あり学習と、「自分で気づかせる」教師なし学習。2つの学び方の違いを体感せよ。",
        "badge": "ラベルあり ⇔ ラベルなし", "page": "pages/8_machine_learning.py", "key": "btn_mission_8",
    },
]

mission_rows = [MISSIONS[i:i + 3] for i in range(0, len(MISSIONS), 3)]
for row in mission_rows:
    cols = st.columns(3)
    for col, m in zip(cols, row):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{svg_icon(m['icon'], size=20)} {m['title']}</h3>
                <p>{m['desc']}</p>
                <div class="math-badge">{m['badge']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="invisible-button">', unsafe_allow_html=True)
                if st.button(f"{m['title']}を体験するにはここをクリック！", use_container_width=True, key=m["key"]):
                    st.switch_page(m["page"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
# =====================================
# 7. 今日のInsight（ハッカー風デザイン）
# =====================================
st.markdown(heading("Decrypted Insight", "sparkles", level=2), unsafe_allow_html=True)

facts = [
    "AIにとって「猫」は数百万次元の空間に浮かぶ、一つのベクトル点に過ぎない。",
    "AIの出力する『確率0.51』は“ほぼ確実”ではなく、単なる“コイントスの結果”である。",
    "画像にわずか1%のノイズを混ぜるだけで、自動運転AIは標識を見誤る可能性がある。",
    "人間の「直感」もまた、過去の膨大なデータから導き出された確率推定器である。",
    "ChatGPTのようなLLMは「意味」を理解しているのではなく、次に来る単語の確率を計算しているだけである。",
    "GPUが1000個の単純な計算を同時に行う一方、CPUは1つの複雑な計算を高速に行う——AIの学習にはGPUが向いている。",
    "「強いAI（汎用人工知能）」はまだ存在しない。現在のAIはすべて特定の作業に特化した「弱いAI」である。",
]

# ランダムなインサイトを表示
st.markdown(f"""
<div class="insight-card">
    <strong>[ SYSTEM MSG ] 傍受したAIの思考ログ:</strong><br><br>
    > {np.random.choice(facts)}
</div>
""", unsafe_allow_html=True)

# =====================================
# 8. フルワイド・フッター
# =====================================
st.markdown(f"""
    <div class="full-width-footer">
        <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
        <p style="margin-top: 8px;">
            <a href="#" style="color: #3b82f6; text-decoration: none;">Security Policy</a> |
            <a href="#" style="color: #3b82f6; text-decoration: none;">Contact Admin</a>
        </p>
    </div>
""", unsafe_allow_html=True)
