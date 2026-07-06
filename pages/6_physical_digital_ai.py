import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from utils.nav import render_top_nav
from utils.icons import svg_icon, icon_html, heading

# 日本語グリフ対応: CJKフォントを優先し、無ければ順にフォールバックする
# (Windows: Yu Gothic/Meiryo, Linux/Docker: fonts-noto-cjk 導入時の Noto Sans CJK JP)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP', 'Noto Sans JP', 'Yu Gothic', 'Meiryo', 'MS Gothic',
    'IPAexGothic', 'TakaoPGothic', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ================================================================
# SECTION 0: ページ設定
# ================================================================
st.set_page_config(
    page_title="ミッション06: AIの二形態 | AI Inquiry Lab",
    page_icon=":material/psychology:",
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
CSS_FILE   = os.path.join(PARENT_DIR, "assets", "style.css")

# カラーパレット
C_BG     = "#ffffff"
C_SPINE  = "#cbd5e1"
C_TEXT   = "#000000"
C_CYAN   = "#0284c7"
C_PINK   = "#c026d3"
C_GREEN  = "#059669"
C_YELLOW = "#b45309"
C_RED    = "#dc2626"
C_PURPLE = "#7c3aed"
C_ORANGE = "#c2410c"
C_MUTED  = "#64748b"

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


def style_axes(ax):
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_color(C_SPINE)
    ax.tick_params(colors=C_TEXT)


# ================================================================
# SECTION 4: データ定義
# ================================================================
# 4象限マップに配置する実在のAI例。
# x = 物理度 (0=完全デジタル … 100=完全に身体を持つ)
# y = 汎用度 (0=一つの作業に特化 … 100=あらゆる課題に対応)
AI_EXAMPLES = [
    # (名前, x物理度, y汎用度, 色, 一言説明)
    ("チャットAI\n(ChatGPT等)", 8,  35, C_CYAN,   "画面の中で文章を作る。会話は上手だが身体は持たない。"),
    ("将棋・囲碁AI",           12, 10, C_CYAN,   "盤上の一手を極める。世界王者に勝つが雑談はできない。"),
    ("おすすめ機能AI",         6,  14, C_CYAN,   "動画や商品を推薦。あなたの好みだけを当てる特化型。"),
    ("画像生成AI",             10, 30, C_CYAN,   "絵を描くのは得意。だが自分で歩くことはできない。"),
    ("お掃除ロボット",         78, 12, C_GREEN,  "部屋を動き回る身体を持つ。仕事は『掃除』だけ。"),
    ("工場ロボットアーム",     88, 8,  C_GREEN,  "決められた溶接や組立を正確に繰り返す特化型ロボット。"),
    ("自動運転支援",           72, 42, C_ORANGE, "多様な道路状況に対応。特化型の中では汎用度が高め。"),
    ("ヒューマノイド試作機",   82, 30, C_GREEN,  "人型の身体を持つが、中身は特定作業向けの弱いAI。"),
]

# クイズのシナリオ (問題文, 正解キー, 解説)
# 正解キー: "narrow" = 弱いAI, "strong" = 強いAI(現実には未存在)
QUIZ = [
    {
        "q": "将棋で世界チャンピオンに勝つが、雑談は一切できないAI。",
        "answer": "narrow",
        "exp": "これは典型的な<b>弱いAI（特化型）</b>です。将棋という1つの領域では人間を超えますが、"
               "その力を会話や料理など別の課題に応用することは一切できません。"
               "『ある1点だけ超人的』は弱いAIの見分け方の代表例です。",
    },
    {
        "q": "映画に出てくる、初めて出会うどんな問題も人間のように考えて自分で解決するロボット。",
        "answer": "strong",
        "exp": "これは<b>強いAI（汎用AI・AGI）</b>の説明です。未知の課題にも自力で応用が利く——"
               "しかし<b>現実の世界にはまだ存在しません</b>。研究中の目標であり、SF映画が描く姿です。"
               "『どんな問題でも』という言葉が出たら強いAIを疑いましょう。",
    },
    {
        "q": "写真に写っているのが猫か犬かを判定するだけのAI。",
        "answer": "narrow",
        "exp": "<b>弱いAI</b>です。画像分類という決まった1タスクの専門家。"
               "犬猫は当てられても、その写真について会話したり、写っている料理のレシピを考えたりはできません。",
    },
    {
        "q": "料理のレシピを考えてくれるが、実際にキッチンで調理はできないAIアシスタント。",
        "answer": "narrow",
        "exp": "<b>弱いAI（しかもデジタル側）</b>です。文章としてレシピは作れますが、"
               "身体を持たないので鍋も振れません。『賢そうな会話』ができても、それは汎用知能の証拠にはなりません。",
    },
    {
        "q": "人型ロボットの身体に、おしゃべりが得意なチャットAIを『頭脳』として載せたもの。",
        "answer": "narrow",
        "exp": "ひっかけ問題です。身体があり会話も達者でも、これは<b>弱いAI</b>のままです。"
               "チャットAIは言語という1領域の特化型。身体を与えても『あらゆる未知の課題に自力で対応する』"
               "汎用知能にはなりません。<b>身体＋おしゃべり＝強いAI、ではない</b>——ここが最重要ポイントです。",
    },
    {
        "q": "自分の目標を勝手に設定し直し、人間に指示されていない新しい分野を独学でマスターしていくAI。",
        "answer": "strong",
        "exp": "これは<b>強いAI（AGI）</b>の特徴です。目標を自ら立て、未知の分野へ自律的に一般化する——"
               "現実のAIにはできません。ニュースで『AIが自我に目覚めた』的な話を見たら、"
               "まずこのレベルの主張なのかを冷静に確かめましょう。ほぼ誇張です。",
    },
]

TOTAL_Q = len(QUIZ)


# ================================================================
# SECTION 5: 図の生成関数
# ================================================================
def plot_quadrant_map(user_x=None, user_y=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(C_BG)
    style_axes(ax)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 上半分（強いAIの領域）＝まだ存在しない：グレーで塗る
    ax.axhspan(50, 100, color="#2a2f3a", alpha=0.55, zorder=0)
    # 中央の分割線
    ax.axvline(50, color=C_SPINE, lw=1.4, ls="--", zorder=1)
    ax.axhline(50, color=C_SPINE, lw=1.4, ls="--", zorder=1)

    # 象限ラベル
    ax.text(25, 95, "digital x strong AI", ha="center", va="top",
            color=C_MUTED, fontsize=10, fontweight="bold")
    ax.text(25, 89, "(mada sonzai shinai)", ha="center", va="top",
            color=C_MUTED, fontsize=8, style="italic")
    ax.text(75, 95, "physical × strong AI", ha="center", va="top",
            color=C_MUTED, fontsize=10, fontweight="bold")
    ax.text(75, 89, "(mada sonzai shinai)", ha="center", va="top",
            color=C_MUTED, fontsize=8, style="italic")
    ax.text(50, 70, "AGI = kenkyuu chuu no mokuhyou",
            ha="center", va="center", color=C_MUTED, fontsize=11,
            style="italic", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1e26", ec=C_MUTED, lw=1, ls="--"))

    ax.text(25, 4, "digital × narrow AI", ha="center", va="bottom",
            color=C_CYAN, fontsize=10, fontweight="bold", alpha=0.75)
    ax.text(75, 4, "physical × narrow AI", ha="center", va="bottom",
            color=C_GREEN, fontsize=10, fontweight="bold", alpha=0.75)

    # 実在AIの点をプロット
    for name, x, y, col, _ in AI_EXAMPLES:
        ax.scatter([x], [y], s=180, c=col, edgecolors="white", lw=0.8, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=C_TEXT, fontsize=8, zorder=6)

    # ユーザーの点
    if user_x is not None and user_y is not None:
        ax.scatter([user_x], [user_y], s=420, marker="*", c=C_YELLOW,
                   edgecolors="white", lw=1.4, zorder=8)
        ax.annotate("Your AI", (user_x, user_y), textcoords="offset points",
                    xytext=(0, 14), ha="center", color=C_YELLOW,
                    fontsize=10, fontweight="bold", zorder=9)

    ax.set_xlabel("<- digital / gamen no naka        sumu basho (butsuri-do)        physical / karada wo motsu ->",
                  color=C_TEXT, fontsize=9)
    ax.set_ylabel("<- narrow / tokka-gata        kashikosa no han-i        strong / hanyou-gata ->",
                  color=C_TEXT, fontsize=9)
    ax.set_title("AI no 4-shougen map", color=C_YELLOW, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.12, color=C_SPINE)
    plt.tight_layout()
    return fig


@st.cache_data
def simulate_robot(noise, latency, steps=120, seed=0):
    """
    直線(中央=0)を追従しようとするロボットの比例制御シミュレーション。
    noise   : センサーノイズの大きさ (0=完璧)
    latency : 反応の遅れ（ステップ数）
    戻り値  : ideal(=0のライン), actual(横位置の履歴)
    """
    rng = np.random.default_rng(seed)
    Kp = 0.45          # 比例ゲイン（操舵の強さ）
    drift = 0.03       # 常に少し右へ押される外乱（摩擦・傾きの比喩）
    pos = 1.2          # 初期位置（少しずれた所からスタート）
    actual = np.zeros(steps)
    sensor_buffer = []
    for t in range(steps):
        actual[t] = pos
        # センサーは「今の横位置」を測るが、ノイズが乗る
        reading = pos + rng.normal(0, noise)
        sensor_buffer.append(reading)
        # 反応の遅れ：latencyステップ前の測定値に基づいて操舵する
        idx = max(0, len(sensor_buffer) - 1 - int(latency))
        delayed = sensor_buffer[idx]
        # 比例制御：測定したズレを打ち消す向きに動く
        steer = -Kp * delayed
        pos = pos + steer + drift + rng.normal(0, noise * 0.15)
    ideal = np.zeros(steps)
    return ideal, actual


def plot_robot_path(ideal, actual, noise):
    steps = len(actual)
    t = np.arange(steps)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(C_BG)
    style_axes(ax)
    # 走行レーン（許容幅）の帯
    ax.axhspan(-0.5, 0.5, color=C_GREEN, alpha=0.10, zorder=0)
    ax.plot(t, ideal, color=C_CYAN, lw=2, ls="--", label="ideal michi (chuuou)")
    line_col = C_GREEN if noise < 0.35 else (C_ORANGE if noise < 0.8 else C_RED)
    ax.plot(t, actual, color=line_col, lw=2, label="robot no jissai no michi")
    ax.axhline(0, color=C_SPINE, lw=0.8)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("time step", color=C_TEXT, fontsize=9)
    ax.set_ylabel("yoko-ichi (zure)", color=C_TEXT, fontsize=9)
    ax.set_title("robot no chokusen tsuiju", color=C_TEXT, fontsize=11)
    ax.legend(labelcolor=C_TEXT, facecolor=C_BG, edgecolor=C_SPINE, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.12, color=C_SPINE)
    plt.tight_layout()
    return fig


# ================================================================
# SECTION 6: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)
render_top_nav("physical")

# ページ専用スコープCSS（クラス名は pd- 接頭辞）
st.markdown("""
<style>
.pd-quad-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 8px 0 4px 0;
}
.pd-quad-card {
    border: 2px solid #cbd5e1;
    border-radius: 6px;
    padding: 12px 14px;
    background: #eef2fb;
}
.pd-quad-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
.pd-quad-card p  { margin: 0; font-size: 0.8rem; color: #334155; line-height: 1.5; }
.pd-axis-chip {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: bold; margin: 2px 4px 2px 0;
    border: 1px solid #cbd5e1;
}
@media (max-width: 640px) {
    .pd-quad-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("footprints", size=32)} AIの二形態</h1>
    <p class="sub-title-text">MISSION 06 — 「賢さの種類」と「住む場所」でAIを見分ける</p>
</div>
''', unsafe_allow_html=True)

# ================================================================
# SECTION 7: セッション状態の初期化
# ================================================================
if "quiz_idx_pd"     not in st.session_state: st.session_state["quiz_idx_pd"]     = 0
if "quiz_score_pd"   not in st.session_state: st.session_state["quiz_score_pd"]   = 0
if "quiz_answered_pd" not in st.session_state: st.session_state["quiz_answered_pd"] = False
if "quiz_last_ok_pd" not in st.session_state: st.session_state["quiz_last_ok_pd"] = None

# ================================================================
# SECTION 8: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#b45309; font-weight:bold; font-size:0.9rem;">
        {svg_icon("footprints", size=15, color="#b45309")} FORM ANALYSIS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('compass', '2つの軸', size=16, color=C_CYAN)}", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#334155; line-height:1.7;">
    {svg_icon("cloud", size=14, color=C_CYAN)} <b>住む場所</b>：デジタル ⟷ 物理<br>
    {svg_icon("brain", size=14, color=C_PINK)} <b>賢さの範囲</b>：弱いAI ⟷ 強いAI
    </div>
    """, unsafe_allow_html=True)

    st.info("現実のAIは、例外なく「弱いAI（特化型）」です。強いAI（AGI）はまだ存在しません。")

    st.divider()
    st.markdown(f"### {icon_html('compass', 'Navigation', size=16)}", unsafe_allow_html=True)
    st.page_link("main_app.py", label="司令室 (Home)")
    st.page_link("pages/1_vision.py", label="M01: AIの目")
    st.page_link("pages/2_adversarial.py", label="M02: AI騙し")
    st.page_link("pages/3_training.py", label="M03: AI育成")
    st.page_link("pages/4_llm_mechanism.py", label="M04: LLMの脳内")
    st.page_link("pages/5_cpu_gpu.py", label="M05: CPU対GPU")
    st.page_link("pages/6_physical_digital_ai.py", label="M06: AIの二形態")
    st.page_link("pages/7_rl_game.py", label="M07: 育成ゲーム")
    st.page_link("pages/8_machine_learning.py", label="M08: 機械学習")

    st.divider()
    st.success("SHIELD: ONLINE")

# ================================================================
# SECTION 9: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{svg_icon("footprints", size=18)} 「AI」とひとくちに言うけれど…</h3>
チャットAI、お掃除ロボット、自動運転、SF映画の意識を持ったロボット——
どれも「AI」と呼ばれますが、その正体はまったく違います。<br><br>
このミッションでは、世の中のあらゆるAIを<b>たった2つの軸</b>で整理します。<br>
{svg_icon("cloud", size=14, color="#0284c7")} <b>住む場所</b>（デジタル空間か、物理世界に身体を持つか）と、
{svg_icon("brain", size=14, color="#c026d3")} <b>賢さの範囲</b>（1つの作業だけ得意か、どんな課題にも応用が利くか）。<br>
この2軸で作る<b>4象限マップ</b>を手に入れれば、
ニュースやCMで飛び交う「すごいAI」の話も、冷静に位置づけられるようになります。
</div>
""", unsafe_allow_html=True)

st.markdown(heading("タブを選んで、AIの2つの顔を見分けよう", "target", level=3), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 4象限マップ",
    "② デジタル vs フィジカル 実験",
    "③ 強いAI vs 弱いAI クイズ",
    "④ AIの未来をどう考えるか",
])

# ================================================================
# TAB 1: 4象限マップ
# ================================================================
with tab1:
    st.markdown(heading("すべてのAIを『2つの軸』で地図にする", "map", level=3),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("compass", size=18)} 2つの軸とは？</h3>
        AIを見分けるとき、多くの人は「賢いか / 賢くないか」の1本のものさしで考えがちです。
        でも、それだと<b>チャットAIとお掃除ロボットの違い</b>を説明できません。
        大事なのは、独立した2本の軸で見ることです。<br><br>

        <b style="color:#0284c7;">▶ X軸：住む場所（物理度）</b><br>
        <b>デジタル空間</b>＝画面の中だけで動く。文章や画像、おすすめを出す。身体は持たない
        （例：チャットAI、将棋AI、レコメンドAI）。<br>
        ⟷ <b>物理世界</b>＝実際のセンサー・モーター・身体を持ち、現実空間で動く
        （例：お掃除ロボット、自動運転車、ヒューマノイド）。<br><br>

        <b style="color:#c026d3;">▶ Y軸：賢さの範囲（汎用度）</b><br>
        <b>弱いAI（特化型 / Narrow AI）</b>＝決まった1つの作業だけが得意。
        <b>今この世に存在するAIは、すべてこれ</b>です。<br>
        ⟷ <b>強いAI（汎用 / AGI）</b>＝人間のように、初めて出会う未知の課題にも自力で応用が利く。
        <b>現時点では存在しない、研究中の目標</b>です。<br>

        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">要点：</b> 世の中の実在AIは、地図の<b>下半分（弱いAI）</b>にしか置けません。
        上半分（強いAI）は、まだSFと研究の中だけの世界です。
        </div>
        """, unsafe_allow_html=True)

    # 4象限の説明カード
    st.markdown(f"""
    <div class="pd-quad-grid">
      <div class="pd-quad-card" style="border-color:#64748b;">
        <h4 style="color:#64748b;">{svg_icon("cloud", size=15, color="#64748b")} デジタル × 強いAI</h4>
        <p><b>まだ存在しない。</b>SF映画の「意識を持ったコンピュータAI」が想像しているのはここ。</p>
      </div>
      <div class="pd-quad-card" style="border-color:#64748b;">
        <h4 style="color:#64748b;">{svg_icon("bot", size=15, color="#64748b")} 物理 × 強いAI</h4>
        <p><b>まだ存在しない。</b>アニメや映画の、人間のように何でもこなす人型AGIロボットはここ。</p>
      </div>
      <div class="pd-quad-card" style="border-color:#0284c7;">
        <h4 style="color:#0284c7;">{svg_icon("cloud", size=15, color="#0284c7")} デジタル × 弱いAI</h4>
        <p>ChatGPTなどのチャットAI、将棋・囲碁AI、おすすめ機能AI。画面の中で1つの仕事を極める。</p>
      </div>
      <div class="pd-quad-card" style="border-color:#059669;">
        <h4 style="color:#059669;">{svg_icon("bot", size=15, color="#059669")} 物理 × 弱いAI</h4>
        <p>お掃除ロボット、工場の溶接ロボット、自動運転の車線維持支援。身体を持つが仕事は特化。</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(heading("あなたのAIアイデアを地図に置いてみよう", "sliders", level=3),
                unsafe_allow_html=True)

    map_ctrl, map_graph = st.columns([2, 3])

    with map_ctrl:
        st.markdown("**思いついた「こんなAIがあったら」を、2つのスライダーで位置づけてみましょう。**")
        user_phys = st.slider("物理度（0=完全デジタル … 100=完全に身体を持つ）",
                              0, 100, 20, 5, key="user_phys_pd")
        user_gen  = st.slider("汎用度（0=一つの作業に特化 … 100=あらゆる課題に対応）",
                              0, 100, 25, 5, key="user_gen_pd")

        st.caption("注：今の現実世界では、汎用度が40〜50を超えるAIは"
                   "まだ実現しておらず、想像上・未来のAI（研究段階）です。")

        # 象限判定
        phys_side = "物理世界のロボット型" if user_phys >= 50 else "デジタル空間型"
        if user_gen >= 50:
            gen_side = "強いAI（汎用）"
            verdict_color = C_MUTED
            reality = ("この位置は<b>まだ存在しない領域</b>です。もし本当に作れたら世界を変える発明ですが、"
                       "現時点では研究者の目標であり、実現の見通しは立っていません。")
        elif user_gen >= 40:
            gen_side = "弱いAIの上限ぎりぎり（かなり多才な特化型）"
            verdict_color = C_ORANGE
            reality = ("現実のAIで到達できる<b>最先端あたり</b>です。自動運転のように"
                       "多様な状況に対応しますが、それでも『決められた役割』の中での話です。")
        else:
            gen_side = "弱いAI（特化型）"
            verdict_color = C_GREEN
            reality = ("これは<b>今すぐ実現可能</b>な、現実的なAIです。1つの仕事に集中するからこそ"
                       "高い性能を出せます。世の中で活躍するAIの大半がこのタイプです。")

        st.markdown(f"""
        <div style="background:#eef2fb; border:1px solid {verdict_color};
                    border-left:6px solid {verdict_color}; padding:14px; margin-top:8px;">
            <div style="color:{verdict_color}; font-weight:bold; margin-bottom:6px;">
            {svg_icon("crosshair", size=15, color=verdict_color)} 判定：{phys_side} × {gen_side}</div>
            <div style="color:#000000; font-size:0.85rem; line-height:1.6;">{reality}</div>
        </div>
        """, unsafe_allow_html=True)

    with map_graph:
        fig_map = plot_quadrant_map(user_phys, user_gen)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_map)
        plt.close(fig_map)
        st.caption("色付きの丸＝実在するAIの例（すべて地図の下半分＝弱いAI）。"
                   "グレーの上半分＝強いAI＝まだ存在しない研究目標。星印＝あなたのAI。")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:8px;">
    <h3>{svg_icon("lightbulb", size=18)} 地図から読み取れること</h3>
    <ul>
    <li>実在AIの丸は<b>すべて下半分</b>に集まっています。これが「今のAIは全部、弱いAI」という事実の見える化です。</li>
    <li><b>自動運転支援</b>だけ少し上（汎用度が高め）にあります。多様な現実の道路状況に対応するためです。
    それでも『運転』という役割の中の話で、料理や会話はできません。<b>やはり弱いAI</b>です。</li>
    <li>横方向（物理度）は難しさとは無関係に見えて、実は<b>開発のしやすさ</b>を大きく左右します。
    次のタブでそれを実験します。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 2: デジタル vs フィジカル 実験
# ================================================================
with tab2:
    st.markdown(heading("同じ頭脳でも『身体を持つ』と急に難しくなる", "bot", level=3),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("contrast", size=18)} きれいな世界 vs 汚い世界</h3>
        <b>デジタルAI</b>（チャットAIなど）は、とてもきれいで整った世界に住んでいます。
        入力は「完璧な形の文章」として届き、ノイズも遅れもありません。計算した通りに物事が進みます。<br><br>
        いっぽう<b>フィジカルAI</b>（ロボット）が住むのは、現実というやっかいな世界です。
        センサーの値はブレる（ノイズ）、判断の反映は一瞬遅れる（遅延）、床には摩擦や傾きがあり、
        急に障害物も現れます。<br>
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">この実験で確かめること：</b>
        まったく同じ「まっすぐ進め」という単純な判断ロジック（比例制御）を積んだロボットでも、
        <b>センサーのノイズと反応の遅れ</b>を加えるだけで、走りがどれだけ乱れるか。<br>
        これが、<b>「身体を持つAI」の開発が桁違いに難しい</b>理由の正体です。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("ライン追従ロボット・シミュレーター", "footprints", level=3),
                unsafe_allow_html=True)

    sim_ctrl, sim_graph = st.columns([2, 3])

    with sim_ctrl:
        st.markdown("**ロボットは中央のライン（ズレ0）をまっすぐ追いかけようとします。"
                    "現実の厳しさを少しずつ足してみましょう。**")
        noise_pd = st.slider("センサーノイズ（0=完璧なセンサー＝デジタルAI的 … 高いほど現実的）",
                             0.0, 1.5, 0.0, 0.05, key="noise_pd")
        latency_pd = st.slider("反応の遅れ（0=即応 … 大きいほど判断が過去の情報に基づく）",
                               0, 12, 0, 1, key="latency_pd")

        st.caption("ノイズ0・遅れ0のときは、画面の中のデジタルAIと同じ『きれいな世界』。"
                   "スライダーを上げるほど、現実のロボットが直面する『汚い世界』になります。")

    ideal, actual = simulate_robot(noise_pd, latency_pd)
    # ズレの指標
    rmse = float(np.sqrt(np.mean(actual ** 2)))
    max_dev = float(np.max(np.abs(actual)))
    out_of_lane = float(np.mean(np.abs(actual) > 0.5) * 100)

    with sim_graph:
        fig_robot = plot_robot_path(ideal, actual, noise_pd)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_robot)
        plt.close(fig_robot)

    rm1, rm2, rm3 = st.columns(3)
    rm1.metric("平均ズレ (RMSE)", f"{rmse:.3f}")
    rm2.metric("最大ズレ", f"{max_dev:.3f}")
    rm3.metric("レーン逸脱率", f"{out_of_lane:.0f}%")

    if noise_pd < 0.2 and latency_pd <= 1:
        st.success("きれいな世界：ロボットはほぼ完璧にラインを追従しています。"
                   "これが『画面の中のデジタルAI』が動く環境のイメージです。")
    elif out_of_lane > 30 or max_dev > 1.5:
        st.error("汚い世界：ロボットが大きく蛇行し、レーンから外れています。"
                 "判断ロジックは同じなのに、ノイズと遅れだけでここまで乱れます。")
    else:
        st.warning("現実の世界：多少ふらつき始めました。実機ではこの揺れを抑えるために、"
                   "フィルタや制御工学など多くの追加エンジニアリングが必要になります。")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:8px;">
    <h3>{svg_icon("lightbulb", size=18)} だから『身体を持つAI』は難しい</h3>
    今のシミュレーションで、ロボットの「頭脳」＝判断ロジックは<b>まったく変えていません</b>。
    変えたのは、センサーのノイズと反応の遅れという<b>現実の物理条件だけ</b>です。
    それだけで、完璧だった走りがフラフラに乱れました。<br><br>
    <b style="color:#b45309;">だからこそ、同じ『AIの頭脳』を使っても、画面の中で動くチャットAIより、
    実際に歩くロボットの方がずっと開発が難しいのです。</b><br><br>
    チャットAIは「きれいな文章」を受け取れば済みますが、ロボットはブレる目・遅れる手・滑る足と
    戦い続けなければなりません。ニュースで「二足歩行ロボットがついに階段を登った」が
    大きく報じられるのは、<b>身体を現実世界で制御することが、それほど困難な挑戦</b>だからです。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 3: 強いAI vs 弱いAI クイズ
# ================================================================
with tab3:
    st.markdown(heading("強いAI・弱いAI 見分けクイズ", "help-circle", level=3),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("scale", size=18)} 定義をはっきりさせる</h3>
        <b style="color:#059669;">弱いAI（特化型 / Narrow AI）</b><br>
        決まった1つの作業だけが得意。例：将棋AI（将棋だけ）、顔認証（顔照合だけ）、
        翻訳AI（翻訳だけ）。<b>今この世にある全AIがこれ</b>です。ある1点では人間を超えることもあります。<br><br>
        <b style="color:#64748b;">強いAI（汎用 / AGI）</b><br>
        人間のように、初めて出会うどんな課題にも自力で応用が利く。例：SF映画の意識を持つAI、
        何でも自分で学ぶロボット。<b>現実にはまだ存在しません。</b><br>
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#dc2626;">よくある誤解を正す：</b>
        SF映画のAIは、ほぼ例外なく『強いAI』として描かれます。だから私たちは
        「AI＝何でもできる万能の知能」というイメージを持ちがちです。
        しかし<b>現実の世界にあるAIは、例外なく『弱いAI』</b>です。
        『すごく賢そうに会話する』ことは、汎用知能の証拠にはなりません。
        </div>
        """, unsafe_allow_html=True)

    idx = st.session_state["quiz_idx_pd"]

    if idx >= TOTAL_Q:
        # 結果画面
        score = st.session_state["quiz_score_pd"]
        pct = score / TOTAL_Q * 100
        st.markdown(f"""
        <div style="background:#eef2fb; border:2px solid #b45309; border-left:6px solid #b45309;
                    padding:20px; text-align:center;">
            <div style="color:#b45309; font-size:1.3rem; font-weight:bold;">
            {svg_icon("trophy", size=22, color="#b45309")} クイズ終了！</div>
            <div style="color:#000000; font-size:1.6rem; margin:10px 0;">
            {score} / {TOTAL_Q} 問正解（{pct:.0f}%）</div>
        </div>
        """, unsafe_allow_html=True)

        if pct >= 80:
            st.success("見事です！あなたはもう、SFのイメージと現実のAIをしっかり切り分けられます。")
            st.balloons()
        elif pct >= 50:
            st.info("いい線です。もう一度チャレンジして、『身体＋会話＝強いAIではない』を"
                    "定着させましょう。")
        else:
            st.warning("もう一度挑戦してみましょう。ポイントは『1つの作業だけ得意＝弱いAI』"
                       "という見分け方です。")

        st.markdown(f"""
        <div class="explanation-box" style="margin-top:8px;">
        <h3>{svg_icon("book-open", size=18)} このクイズの一番の学び</h3>
        <b>SF映画のAIはほぼ『強いAI』として描かれるが、現実の世界にあるAIは例外なく『弱いAI』である。</b><br><br>
        身体を与えても、上手におしゃべりができても、それは<b>特定の役割をこなす弱いAI</b>のまま。
        「どんな未知の問題も自力で解決する」汎用知能＝強いAI（AGI）は、
        いまだ研究の途上にあります。この見分ける目こそ、AIニュースに振り回されないための武器です。
        </div>
        """, unsafe_allow_html=True)

        if st.button("もう一度挑戦する", key="quiz_reset_pd"):
            st.session_state["quiz_idx_pd"] = 0
            st.session_state["quiz_score_pd"] = 0
            st.session_state["quiz_answered_pd"] = False
            st.session_state["quiz_last_ok_pd"] = None
            st.rerun()
    else:
        item = QUIZ[idx]
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin:6px 0;">
            <span style="color:#b45309; font-weight:bold;">第 {idx + 1} 問 / {TOTAL_Q}</span>
            <span style="color:#64748b;">現在のスコア：{st.session_state['quiz_score_pd']}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(idx / TOTAL_Q)

        st.markdown(f"""
        <div style="background:#eef2fb; border:2px solid #0284c7; border-left:6px solid #0284c7;
                    padding:18px; margin:8px 0; font-size:1.05rem; color:#000000;">
            {svg_icon("message-circle", size=18, color="#0284c7")} {item['q']}
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state["quiz_answered_pd"]:
            c_a, c_b = st.columns(2)
            with c_a:
                if st.button("弱いAI（特化型）", use_container_width=True, key=f"ans_narrow_{idx}"):
                    ok = (item["answer"] == "narrow")
                    st.session_state["quiz_answered_pd"] = True
                    st.session_state["quiz_last_ok_pd"] = ok
                    if ok:
                        st.session_state["quiz_score_pd"] += 1
                    st.rerun()
            with c_b:
                if st.button("強いAI（汎用・AGI）", use_container_width=True, key=f"ans_strong_{idx}"):
                    ok = (item["answer"] == "strong")
                    st.session_state["quiz_answered_pd"] = True
                    st.session_state["quiz_last_ok_pd"] = ok
                    if ok:
                        st.session_state["quiz_score_pd"] += 1
                    st.rerun()
        else:
            ok = st.session_state["quiz_last_ok_pd"]
            correct_label = "弱いAI（特化型）" if item["answer"] == "narrow" else "強いAI（汎用・AGI）"
            if ok:
                st.success(f"正解！ これは「{correct_label}」です。")
            else:
                st.error(f"おしい！ 正解は「{correct_label}」でした。")

            border = C_GREEN if ok else C_RED
            st.markdown(f"""
            <div class="explanation-box" style="border-color:{border};">
            <h3>{svg_icon("lightbulb", size=18)} 解説</h3>
            {item['exp']}
            </div>
            """, unsafe_allow_html=True)

            next_label = "次の問題へ" if idx + 1 < TOTAL_Q else "結果を見る"
            if st.button(next_label, key=f"next_{idx}"):
                st.session_state["quiz_idx_pd"] += 1
                st.session_state["quiz_answered_pd"] = False
                st.session_state["quiz_last_ok_pd"] = None
                st.rerun()

# ================================================================
# TAB 4: AIの未来をどう考えるか
# ================================================================
with tab4:
    st.markdown(heading("AIの未来を、冷静に・公平に考える", "compass", level=3),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("scale", size=18)} 恐れすぎず、侮りすぎず</h3>
        AIの未来を語るとき、世の中は2つの極端に振れがちです。
        「もうすぐAIが人類を超える」という<b>過度な期待・不安</b>と、
        「しょせん自動化ツール、たいしたことない」という<b>過度な軽視</b>。
        どちらも、現実を正しく見ることを妨げます。<br><br>
        このタブでは採点も正解もありません。事実を整理したうえで、
        <b>あなた自身がどう考えるか</b>を落ち着いて見つめるための材料を用意しました。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("強いAIに近づくには、本当は何が必要か", "puzzle", level=3),
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="explanation-box">
    「計算資源（GPU）をもっと増やせば、いずれ強いAIになる」——これはよくある単純化です。
    しかし多くの研究者は、それだけでは足りないと考えています。汎用知能に近づくには、
    計算量では埋まらない<b>未解決の研究課題</b>がいくつも横たわっています。
    <ul>
    <li><b>{icon_html('refresh-cw', '未知への一般化', size=14, color=C_CYAN)}</b>：
    訓練で見たことのない、まったく新しい状況に自力で対応する力。今のAIは苦手です。</li>
    <li><b>{icon_html('link', '常識と因果', size=14, color=C_CYAN)}</b>：
    「物を離せば落ちる」のような世界の仕組み（因果関係）を、本当の意味で理解しているわけではありません。</li>
    <li><b>{icon_html('target', '自律的な目標設定', size=14, color=C_CYAN)}</b>：
    人間が与えた課題は解けても、自分で意味のある目標を立てることはできません。</li>
    <li><b>{icon_html('footprints', '身体を通じた学習', size=14, color=C_CYAN)}</b>：
    人間は身体で世界に触れながら学びます（このミッションの第2の軸ですね）。
    これをどう実現するかは大きな研究テーマです。</li>
    </ul>
    これらは「もっと大きなモデル」を作れば自動的に解ける、という保証がありません。
    だからこそ、強いAIは<b>工学の問題であると同時に、未解明の科学の問題</b>なのです。
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("なぜ専門家でも予測が割れるのか", "users", level=3),
                unsafe_allow_html=True)
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        st.markdown(f"""
        <div class="pd-quad-card" style="border-color:#059669;">
        <h4 style="color:#059669;">{svg_icon("trending-up", size=15, color="#059669")} 「近い」と考える人の論拠</h4>
        <p>ここ数年の進歩の速さ、モデルを大きくするほど性能が伸びてきた実績、
        投資と人材の集中。これらを根拠に、数十年以内の実現もありうると見る。</p>
        </div>
        """, unsafe_allow_html=True)
    with exp_c2:
        st.markdown(f"""
        <div class="pd-quad-card" style="border-color:#c2410c;">
        <h4 style="color:#c2410c;">{svg_icon("trending-down", size=15, color="#c2410c")} 「まだ遠い」と考える人の論拠</h4>
        <p>上記の未解決課題が本質的に難しいこと、過去にもAIブームと冬が繰り返されてきた歴史、
        ベンチマークの高得点が現実の汎用性に直結しないこと。</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="explanation-box" style="margin-top:8px;">
    どちらの立場にも、まじめな研究者がいます。予測が割れる最大の理由は、
    <b>「汎用知能に必要なピースが全部そろっているのか、まだ誰にも分からない」</b>からです。
    誰かが自信満々に断定していたら、むしろ少し警戒したほうがよいくらいです。
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("ニュースの『AI突破』を見抜く4つの問い", "search", level=3),
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="explanation-box">
    「AIがついに〇〇を達成！」という見出しを見たとき、うのみにせず次を自分に問いかけてみましょう。
    <ol>
    <li><b>それは大きくなった弱いAIか、汎用への一歩か？</b>
    多くの「突破」は、既存の特化型AIを大規模化したもの。強いAIへの質的な飛躍とは別物です。</li>
    <li><b>誰が言っている？</b>
    製品を売りたい企業のマーケティングか、第三者が検証した査読付きの研究か。動機を見ましょう。</li>
    <li><b>どんな条件での成果？</b>
    特定のテストや限られた環境だけの結果を、あたかも万能の証拠のように語っていないか。</li>
    <li><b>失敗例やできないことに触れているか？</b>
    誠実な発表は必ず限界も語ります。良い面しか言わない話ほど、割り引いて聞くべきです。</li>
    </ol>
    この4つを習慣にするだけで、誇大広告と本物の進歩を、かなり見分けられるようになります。
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("あなたはどう考えますか？", "message-circle", level=3),
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#eef2fb; border:2px solid #7c3aed; border-left:6px solid #7c3aed;
                padding:18px;">
        <div style="color:#7c3aed; font-weight:bold; margin-bottom:10px;">
        {svg_icon("lightbulb", size=16, color="#7c3aed")} 3つの問い（正解はありません）</div>
        <div style="color:#000000; font-size:0.9rem; line-height:1.9;">
        <b>1.</b> あなたの身の回りにある「AI」は、この地図のどこに置けますか？
        それは本当に強いAIでしょうか、それとも上手な弱いAIでしょうか？<br>
        <b>2.</b> もし本物の強いAI（AGI）が実現したら、あなたはどんな仕事を任せたいですか？
        逆に、絶対に任せたくないことは何ですか？<br>
        <b>3.</b> 「AIが人間を超える」という言葉を聞いたとき、それは<b>どの軸で</b>超えるという意味でしょうか。
        1つの作業で（＝すでに起きている）？ それとも、あらゆる面で（＝まだ遠い）？
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:12px;">
    <h3>{svg_icon("check-circle", size=18)} このミッションのまとめ</h3>
    <ul>
    <li>すべてのAIは<b>「住む場所（デジタル⟷物理）」×「賢さの範囲（弱い⟷強い）」</b>の2軸で整理できる。</li>
    <li><b>今あるAIは、例外なく弱いAI（特化型）</b>。強いAI（AGI）はまだ存在しない研究目標。</li>
    <li><b>身体を持つAIは開発が桁違いに難しい</b>。同じ頭脳でも現実世界のノイズと遅れが壁になる。</li>
    <li><b>身体＋おしゃべり＝強いAI、ではない</b>。それは有用だが、あくまで身体を操る弱いAI。</li>
    <li>AIニュースは、<b>誇大広告と本物の進歩を切り分けて</b>受け取る。それが賢い付き合い方。</li>
    </ul>
    AIを恐れる必要も、侮る必要もありません。<b>正しく理解すること</b>が、いちばんの力になります。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown('''
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 06: AIの二形態 — フィジカル/デジタル × 弱いAI/強いAI</p>
</div>
''', unsafe_allow_html=True)
