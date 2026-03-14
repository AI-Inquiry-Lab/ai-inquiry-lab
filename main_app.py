import streamlit as st
import numpy as np
from datetime import datetime
import hashlib
import os
import time

# =====================================
# 1. ページ設定（※一番最初に記述必須！）
# =====================================
st.set_page_config(
    page_title="AI Inquiry Lab | トップ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# 2. CSS読み込み処理（一本化＆絶対パス化）
# =====================================
def load_css(file_name):
    # 実行ファイル(main_app.py)があるディレクトリを取得
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "assets", file_name)
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # 見つからない場合のデバッグ表示
        st.error(f"⚠️ {file_name} が見つかりません。\n探索パス: {file_path}")

# assets/style.css を読み込む
load_css("style.css")

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
    st.markdown('<h2 style="color: #3b82f6; font-size: 1.2rem;">⚡ AI CORE CONTROL</h2>', unsafe_allow_html=True)
    
    # リアルタイム感のあるプログレスバー
    st.write("🧬 **Neural Syncing...**")
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
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧭 Navigation")
    st.page_link("main_app.py", label="司令室 (Home)", icon="🏠")
    st.page_link("pages/1_AIの目.py", label="ミッション01: AIの目", icon="👁️")
    st.success("🛡️ SHIELD: ONLINE")
    # システム時刻の秒まで動的に表示（再読み込みのたびに更新）
    st.caption(f"Last Ping: {datetime.now().strftime('%H:%M:%S')}")
# =====================================
# 5. ヒーローセクション & ラボ・ガイド
# =====================================
st.markdown('<div class="hero-title">AI Inquiry Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">数学 × データ × 思考実験で、AIのブラックボックスを解体する。</div>', unsafe_allow_html=True)

# セレクトボックスの代わりに、ラボの「提供価値」を3つのフェーズで紹介
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="guide-box">
            <h4>🔍 解体</h4>
            <p>「魔法」に見えるAIを、数式とコードのレベルまでバラバラにして、その中身を覗き見ます。</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="guide-box">
            <h4>🧪 実験</h4>
            <p>理論を学ぶだけではありません。実際にパラメータをいじり、AIの挙動がどう変わるかを手元で体験します。</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="guide-box">
            <h4>🛡️ 掌握</h4>
            <p>ブラックボックスの正体を知ることで、AIを恐れるのではなく、使いこなすための「知の武器」を手に入れます。</p>
        </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# =====================================
# 6. ミッションボード（実験コンテンツ）
# =====================================
st.markdown("### 🚀 Active Missions")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feature-card">
        <h3>👁️ ミッション01: AIの目</h3>
        <p>画像は単なる「数字の配列」に過ぎない。AIが世界を認識する最初のステップを体験せよ。</p>
        <div class="math-badge">RGB = (255, 128, 64)</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="invisible-button">', unsafe_allow_html=True)
        if st.button("AIの目を体験するにはここをクリック！",use_container_width=True,key="btn_mission_1"):
            st.switch_page("pages/1_AIの目.py")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- ミッション2：ロック中 ---
with c2:
    st.markdown("""
    <div class="feature-card locked-card">
        <h3>🎭 ミッション02: AI騙し</h3>
        <p>人間の目には見えない微小なノイズ。AIの脆弱性の仕組みを暴く。（現在ロック中）</p>
        <div class="math-badge">f(x + ε) ≠ y</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="invisible-button">', unsafe_allow_html=True)
        if st.button("🔒 アクセス権限不足", disabled=True, use_container_width=True, key="lock1"):
            st.switch_page("pages/2_AI騙し.py")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
# --- ミッション3：ロック中 ---
with c3:
    st.markdown("""
    <div class="feature-card locked-card">
        <h3>🧠 ミッション03: 育成</h3>
        <p>AIの脳ネットワークを構築する。学習プロセスを可視化せよ。（現在ロック中）</p>
        <div class="math-badge">Loss ↓ Accuracy ↑</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="invisible-button">', unsafe_allow_html=True)
        if st.button("🔒 アクセス権限不足", disabled=True, use_container_width=True, key="lock2"):
            st.switch_page("pages/3_AI育成.py")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
# =====================================
# 7. 今日のInsight（ハッカー風デザイン）
# =====================================
st.markdown("### 💡 Decrypted Insight")

facts = [
    "AIにとって「猫」は数百万次元の空間に浮かぶ、一つのベクトル点に過ぎない。",
    "AIの出力する『確率0.51』は“ほぼ確実”ではなく、単なる“コイントスの結果”である。",
    "画像にわずか1%のノイズを混ぜるだけで、自動運転AIは標識を見誤る可能性がある。",
    "人間の「直感」もまた、過去の膨大なデータから導き出された確率推定器である。"
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