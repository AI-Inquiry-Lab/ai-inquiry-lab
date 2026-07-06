import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    page_title="ミッション05: CPU対GPU | AI Inquiry Lab",
    page_icon=":material/memory:",
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

# テーマ配色（マット系）
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

# ラウンド色分け用の離散カラーマップ
ROUND_COLORS = [C_CYAN, C_GREEN, C_YELLOW, C_PINK, C_PURPLE, C_ORANGE, C_RED,
                "#4ac6ff", "#7dffc8", "#ffe466", "#ffa3e0", "#cfa8ff", "#ffbd7d"]


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


def style_ax(ax):
    ax.set_facecolor(C_BG)
    for sp in ax.spines.values():
        sp.set_color(C_SPINE)
    ax.tick_params(colors=C_TEXT)


def round_color(i: int) -> str:
    return ROUND_COLORS[i % len(ROUND_COLORS)]


# ================================================================
# SECTION 4: シミュレーション・描画関数群
# ================================================================
def simulate_time(tasks: int, cores: int, time_per_task: float) -> tuple:
    """整列バッチ処理モデル: ceil(tasks / cores) ラウンド × 1タスクの処理時間。"""
    rounds = int(np.ceil(tasks / cores))
    total  = rounds * time_per_task
    return rounds, total


def plot_time_bars(cpu_time: float, gpu_time: float):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(C_BG)
    style_ax(ax)
    labels = ["CPU", "GPU"]
    vals   = [cpu_time, gpu_time]
    colors = [C_ORANGE, C_GREEN]
    bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("Total simulated time (arbitrary units)", color=C_TEXT, fontsize=9)
    ax.set_title("Total time to finish all tasks (lower = faster)", color=C_YELLOW, fontsize=11)
    vmax = max(vals) if max(vals) > 0 else 1
    for b, v in zip(bars, vals):
        ax.text(v + vmax * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:,.0f}", va="center", ha="left", color=C_TEXT, fontsize=10, fontweight="bold")
    ax.set_xlim(0, vmax * 1.18)
    ax.grid(True, axis="x", alpha=0.15, color=C_SPINE)
    plt.tight_layout(pad=0.6)
    return fig


def plot_batch_grid(vis_tasks: int, cpu_cores: int, gpu_cores: int):
    """縮小した代表タスク数で、各プロセッサが何ラウンドで塗り終えるかを可視化。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.patch.set_facecolor(C_BG)

    ncol = 10
    nrow = int(np.ceil(vis_tasks / ncol))

    for ax, cores, name, color in [
        (axes[0], cpu_cores, "CPU", C_ORANGE),
        (axes[1], gpu_cores, "GPU", C_GREEN),
    ]:
        ax.set_facecolor(C_BG)
        ax.axis("off")
        ax.set_xlim(-0.5, ncol + 0.5)
        ax.set_ylim(-0.5, nrow + 0.5)
        ax.set_aspect("equal")
        rounds = int(np.ceil(vis_tasks / max(cores, 1)))
        for t in range(vis_tasks):
            r = t // ncol
            c = t % ncol
            batch = t // max(cores, 1)   # このタスクが処理されるラウンド番号
            rect = plt.Rectangle((c, nrow - 1 - r), 0.9, 0.9,
                                 facecolor=round_color(batch), edgecolor=C_BG, lw=1.0)
            ax.add_patch(rect)
        ax.set_title(f"{name}: {cores:,} cores -> {rounds} rounds",
                     color=color, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Rounds needed to clear all tasks (color = round #)",
                 color=C_TEXT, fontsize=10, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_matrix_rounds(N: int, cells_at_once: int):
    """N×N 出力行列の各マス目が、何ラウンド目に計算されるかを色分け表示。"""
    total_cells = N * N
    cells_at_once = max(1, min(cells_at_once, total_cells))
    rounds = int(np.ceil(total_cells / cells_at_once))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.axis("off")
    ax.set_xlim(-0.3, N + 0.3)
    ax.set_ylim(-0.3, N + 0.3)
    ax.set_aspect("equal")

    for idx in range(total_cells):
        r = idx // N
        c = idx % N
        batch = idx // cells_at_once
        rect = plt.Rectangle((c, N - 1 - r), 0.94, 0.94,
                             facecolor=round_color(batch), edgecolor=C_BG, lw=1.5)
        ax.add_patch(rect)
        ax.text(c + 0.47, N - 1 - r + 0.47, f"R{batch + 1}",
                ha="center", va="center", color="#eef2fb", fontsize=max(7, 14 - N),
                fontweight="bold")

    ax.set_title(f"Output cells colored by round  (rounds = {rounds})",
                 color=C_YELLOW, fontsize=10, pad=8)
    plt.tight_layout(pad=0.5)
    return fig, rounds


# ================================================================
# SECTION 5: CSSロード & スコープ付きスタイル
# ================================================================
load_css(CSS_FILE)

# このページ専用のコアグリッド用スタイル（style.cssは編集しない方針のためここに注入）
st.markdown("""
<style>
.hw-core-wrap { display:flex; flex-wrap:wrap; gap:16px; margin:8px 0 4px 0; }
.hw-core-card {
    flex: 1 1 260px;
    background:#ffffff; border:2px solid #cbd5e1; padding:14px;
    box-shadow:4px 4px 0px #eef2fb;
}
.hw-core-card.cpu { border-color:#c2410c; }
.hw-core-card.gpu { border-color:#059669; }
.hw-core-title { font-weight:900; font-size:1.0rem; margin-bottom:4px; }
.hw-core-title.cpu { color:#c2410c; }
.hw-core-title.gpu { color:#059669; }
.hw-core-sub { color:#64748b; font-size:0.78rem; margin-bottom:10px; }
.hw-grid-cpu {
    display:grid; grid-template-columns:repeat(4, 1fr); gap:8px;
    max-width:220px;
}
.hw-grid-cpu span {
    aspect-ratio:1/1; background:#c2410c; border:2px solid #ffd0a0;
    display:flex; align-items:center; justify-content:center;
    color:#eef2fb; font-weight:900; font-size:0.7rem;
}
.hw-grid-gpu {
    display:grid; grid-template-columns:repeat(20, 1fr); gap:2px;
}
.hw-grid-gpu span {
    aspect-ratio:1/1; background:#059669; border:1px solid #0d3a26;
}
.hw-cap { color:#000000; font-size:0.82rem; margin-top:10px; line-height:1.6; }
.hw-cap b { color:#b45309; }
@media (max-width: 640px) {
    .hw-core-card { flex:1 1 100%; }
    .hw-grid-cpu { grid-template-columns:repeat(4, 1fr); gap:5px; max-width:150px; }
    .hw-grid-gpu { grid-template-columns:repeat(16, 1fr); gap:1px; }
}
</style>
""", unsafe_allow_html=True)

render_top_nav("hardware")

st.markdown(f'''
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("cpu", size=32)} CPU対GPU</h1>
    <p class="sub-title-text">MISSION 05 — なぜAIの学習には「GPU」が必要なのか？</p>
</div>
''', unsafe_allow_html=True)

# ================================================================
# SECTION 6: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#b45309; font-weight:bold; font-size:0.9rem;">
        {svg_icon("cpu", size=15, color="#b45309")} HARDWARE MODE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('sliders', 'クイック設定', size=16, color=C_CYAN)}",
                unsafe_allow_html=True)
    st.caption("「② 並列計算レース」タブの初期値になります。")
    sb_tasks = st.slider("処理タスク総数", 100, 20000, 5000, 100, key="sb_tasks_hw")
    sb_cpu   = st.slider("CPUコア数",      1,    16,    8,    1,   key="sb_cpu_hw")
    sb_gpu   = st.slider("GPUコア数",      256,  8192,  2048, 256, key="sb_gpu_hw")

    st.divider()
    st.markdown(f"### {icon_html('compass', 'Navigation', size=16, color=C_CYAN)}",
                unsafe_allow_html=True)
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
# SECTION 7: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{icon_html('cpu', 'AIを動かす「頭脳」は2種類ある', size=20, color=C_CYAN)}</h3>
あなたのパソコンやスマホの中には、計算を担当する部品が入っています。
その代表格が<b>CPU</b>と<b>GPU</b>です。この2つは「どちらが偉い」わけではなく、
<b>得意な仕事がまったく違う</b>のです。<br><br>
<b>CPU</b>は<b>少数の非常に賢い処理係</b>——たとえるなら「1人の熟練シェフ」。
複雑な判断や順番の決まった作業を、1つずつ丁寧かつ高速にこなします。<br>
<b>GPU</b>は<b>大量の単純作業員</b>——たとえるなら「1000人のアルバイト」。
1人あたりの能力は控えめでも、<b>同じ単純作業を全員で一斉に</b>片づけます。<br><br>
そしてAIの学習の正体は、<b>「同じ掛け算・足し算を何百万回もくり返す」巨大な計算</b>。
だから少数の天才（CPU）より、大量の単純作業員（GPU）の方が圧倒的に速いのです。
このミッションで、その理由を手を動かして確かめましょう。<br>
<hr style="border-color:#cbd5e1; margin:10px 0;">
<b style="color:#059669;">注意：</b>このページの計算時間はすべて
<b>教育用の単純な数式モデル（近似）</b>です。実際の製品のベンチマーク値ではありません。
</div>
""", unsafe_allow_html=True)

st.markdown(f"## {icon_html('flask', 'タブを選んで、CPUとGPUの違いを体験しよう', size=22, color=C_YELLOW)}",
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① アーキテクチャ図鑑",
    "② 並列計算レース",
    "③ 行列演算を並列化する",
    "④ まとめ：AIとハードウェアの歴史",
])

# ================================================================
# TAB 1: アーキテクチャ図鑑
# ================================================================
with tab1:
    st.markdown(f"### {icon_html('layers', 'CPUとGPU、中身はこんなに違う', size=18, color=C_CYAN)}",
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{icon_html('brain', '「少数の天才」対「大量の作業員」', size=20, color=C_CYAN)}</h3>
        <b style="color:#c2410c;">CPU（中央処理装置）= 少数の超優秀なコア</b><br>
        コアの数はだいたい<b>4〜16個</b>ほど。1つ1つがとても速く、柔軟で、
        <b>複雑な判断や条件分岐</b>が得意です。プログラムの流れをコントロールする「司令塔」の役割。<br>
        たとえるなら<b>1人の熟練シェフが、一皿ずつ丁寧にコース料理を仕上げる</b>イメージ。<br><br>
        <b style="color:#059669;">GPU（画像処理装置）= 大量の単純なコア</b><br>
        コアの数は<b>数千〜1万個以上</b>。1つ1つはCPUのコアより遅くて単純ですが、
        <b>同じ種類の小さな計算を全員で同時に</b>処理できます。<br>
        たとえるなら<b>1000人のアルバイトが、同時に同じ「野菜切り」だけをひたすらこなす</b>イメージ。<br>
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">ポイント：</b>
        「1人が全部やる」より「1000人で手分けする」方が速いのは、
        <b>その作業が全部バラバラ（独立）で、同じ手順のとき</b>だけ。
        AIの学習はまさにこのパターンなので、GPUが輝きます。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('bar-chart', 'コアの数を目で見て比べる（イラスト値）', size=18, color=C_GREEN)}",
                unsafe_allow_html=True)

    # コアグリッド（CPU: 大きな8マス / GPU: 小さな多数マス）
    cpu_cells = "".join(f"<span>{i+1}</span>" for i in range(8))
    gpu_cells = "".join("<span></span>" for _ in range(320))  # 表示用のイラスト密度
    st.markdown(f"""
    <div class="hw-core-wrap">
        <div class="hw-core-card cpu">
            <div class="hw-core-title cpu">CPU</div>
            <div class="hw-core-sub">少数・大型・高性能なコア（イラスト：8コア）</div>
            <div class="hw-grid-cpu">{cpu_cells}</div>
            <div class="hw-cap">大きくて賢いコアが<b>数個</b>。
            1つずつが速く、複雑な指示もこなせる。</div>
        </div>
        <div class="hw-core-card gpu">
            <div class="hw-core-title gpu">GPU</div>
            <div class="hw-core-sub">多数・小型・単純なコア（イラスト：5,120コアの一部）</div>
            <div class="hw-grid-gpu">{gpu_cells}</div>
            <div class="hw-cap">小さなコアが<b>数千個</b>。
            1つの能力は控えめでも、全員で一斉に同じ計算を片づける。</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("※ コア数（8 / 5,120）はイメージをつかむための概略値です。特定の実在製品の仕様ではありません。")

    st.divider()
    st.markdown(f"### {icon_html('help-circle', 'ミニクイズ：どっちが得意？', size=18, color=C_YELLOW)}",
                unsafe_allow_html=True)
    st.caption("次の作業は CPU と GPU、どちらが向いているでしょう？ 選んでみよう。")

    quiz = [
        {
            "key": "q1_hw",
            "q": "メールの中身をチェックして「これは重要」「これは迷惑メール」と細かく条件分岐して振り分ける",
            "ans": "CPU",
            "why": "複雑な条件分岐と順番のある判断が中心。少数でも賢いCPU向きの仕事です。",
        },
        {
            "key": "q2_hw",
            "q": "3Dゲームで、画面上の何百万個ものピクセルの色を同時に計算して描画する",
            "ans": "GPU",
            "why": "全ピクセルがほぼ独立した同じ計算。大量並列が得意なGPUの本領（もともとGPUはこのために生まれました）。",
        },
        {
            "key": "q3_hw",
            "q": "AIモデルの学習で、巨大な行列どうしの掛け算を何百万回もくり返す",
            "ans": "GPU",
            "why": "独立した同じ掛け算・足し算の山。まさにGPUのためにあるような仕事です。",
        },
        {
            "key": "q4_hw",
            "q": "ワープロで文章を打ちながら、変換・レイアウト・保存を順番に処理する",
            "ans": "CPU",
            "why": "順番に進める処理が中心で、並列化の余地は小さい。CPUが得意な領域です。",
        },
    ]

    correct_count = 0
    answered = 0
    for i, item in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {item['q']}**")
        choice = st.radio(
            "この作業が得意なのは？",
            ["まだ選んでいない", "CPU", "GPU"],
            key=item["key"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if choice != "まだ選んでいない":
            answered += 1
            if choice == item["ans"]:
                correct_count += 1
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color:#059669;">
                <b style="color:#059669;">正解！ 答えは {item['ans']}。</b><br>{item['why']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="explanation-box" style="border-left-color:#dc2626;">
                <b style="color:#dc2626;">おしい！ 正解は {item['ans']} です。</b><br>{item['why']}
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if answered == len(quiz):
        if correct_count == len(quiz):
            st.success(f"全問正解！ {correct_count}/{len(quiz)}。CPUとGPUの役割分担、完璧に理解できています。")
        else:
            st.info(f"現在 {correct_count}/{len(quiz)} 問正解。上の解説を読んで、もう一度考えてみよう。")

# ================================================================
# TAB 2: 並列計算レース
# ================================================================
with tab2:
    st.markdown(f"### {icon_html('rocket', '同じ仕事を、CPUとGPUで競わせよう', size=18, color=C_CYAN)}",
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{icon_html('flask', 'レースのルール', size=20, color=C_CYAN)}</h3>
        たくさんの<b>独立した単純タスク</b>（たとえば「1マスずつ色を塗る」「1個ずつ掛け算する」）を、
        CPUチームとGPUチームのどちらが速く終わらせるかを競います。<br><br>
        コアは<b>同時に動く作業員</b>。1ラウンドで「コア数」だけタスクを片づけ、
        全タスクが終わるまでラウンドをくり返します。<br>
        <b style="color:#b45309;">大事な仕掛け：</b>
        GPUの1コアは、実はCPUの1コアより<b>1タスクあたりの処理が遅い</b>設定にしてあります
        （下のスライダーで確認できます）。それでもGPUが勝てるのは、<b>頭数（並列数）が桁違いに多い</b>から。
        「1人の速さ」ではなく「全体の処理量（スループット）」で勝負が決まるのです。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('sliders', 'レース設定', size=18, color=C_GREEN)}",
                unsafe_allow_html=True)

    cset1, cset2 = st.columns(2)
    with cset1:
        tasks = st.slider("処理タスクの総数（独立した単純作業）", 100, 20000,
                          int(st.session_state.get("sb_tasks_hw", 5000)), 100, key="race_tasks_hw")
        cpu_cores = st.slider("CPUコア数（少数・高性能）", 1, 16,
                              int(st.session_state.get("sb_cpu_hw", 8)), 1, key="race_cpu_hw")
        gpu_cores = st.slider("GPUコア数（多数・単純）", 256, 8192,
                              int(st.session_state.get("sb_gpu_hw", 2048)), 256, key="race_gpu_hw")
    with cset2:
        cpu_tpt = st.slider("CPU：1タスクあたりの処理時間", 0.5, 3.0, 1.0, 0.1,
                            format="%.1f", key="race_cpu_tpt_hw")
        gpu_tpt = st.slider("GPU：1タスクあたりの処理時間（あえてCPUより遅め）",
                            1.0, 6.0, 3.0, 0.1, format="%.1f", key="race_gpu_tpt_hw")
        st.caption("※ 時間の単位は説明用の「架空の単位」です。GPUの1コアはCPUより遅い＝単純、という前提を表しています。")

    cpu_rounds, cpu_time = simulate_time(tasks, cpu_cores, cpu_tpt)
    gpu_rounds, gpu_time = simulate_time(tasks, gpu_cores, gpu_tpt)
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0

    # 計算式を明示（ブラックボックスにしない）
    st.markdown(f"""
    <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:14px;
                font-family:monospace; font-size:0.9rem; margin:6px 0;">
        <span style="color:#b45309;">使っている計算式（切り上げ除算）：</span><br>
        total_time = <b style="color:#0284c7;">ceil(tasks / cores)</b> × time_per_task<br><br>
        <span style="color:#c2410c;">CPU:</span> ceil({tasks:,} / {cpu_cores}) = {cpu_rounds:,} ラウンド
        × {cpu_tpt:.1f} = <span style="color:#c2410c; font-weight:bold;">{cpu_time:,.0f}</span><br>
        <span style="color:#059669;">GPU:</span> ceil({tasks:,} / {gpu_cores:,}) = {gpu_rounds:,} ラウンド
        × {gpu_tpt:.1f} = <span style="color:#059669; font-weight:bold;">{gpu_time:,.0f}</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("CPU 合計時間", f"{cpu_time:,.0f}", f"{cpu_rounds:,} ラウンド")
    m2.metric("GPU 合計時間", f"{gpu_time:,.0f}", f"{gpu_rounds:,} ラウンド")
    m3.metric("GPUの速さ（CPU比）", f"{speedup:,.1f} 倍" if speedup >= 1 else f"{speedup:.2f} 倍")

    fig_bar = plot_time_bars(cpu_time, gpu_time)
    st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    if speedup >= 1.0:
        st.success(f"GPUの勝ち！ 同じ仕事を約 {speedup:,.1f} 倍の速さで終わらせました。"
                   f"1コアは遅くても、頭数の多さがすべてをひっくり返します。")
    else:
        st.warning("いまの設定ではCPUが勝っています。タスク数を増やすか、GPUコア数を増やしてみよう。"
                   "タスクが少ないと、大量の作業員がいても手持ち無沙汰になり並列の強みが出ません。")

    st.divider()
    st.markdown(f"### {icon_html('layers', 'ラウンド数を目で見る（縮小イラスト）', size=18, color=C_YELLOW)}",
                unsafe_allow_html=True)
    st.caption("実際の数字は上のメトリクス/グラフの通りです。ここでは仕組みを見るため、"
               "タスクを約40個に縮小し、コア数も同じ比率で縮めて『何ラウンドで塗り終わるか』を色分け表示します。")

    vis_tasks = 40
    scale = vis_tasks / tasks
    vis_cpu = max(1, int(round(cpu_cores * scale)))
    vis_gpu = max(1, int(round(gpu_cores * scale)))
    # GPUは縮小しても全タスクをほぼ1ラウンドで捌けることを見せる
    vis_gpu = max(vis_gpu, 1)

    st.markdown(f"""
    <div style="font-family:monospace; font-size:0.82rem; color:#64748b; margin-bottom:4px;">
    縮小イラスト条件： タスク {vis_tasks} 個 ／ CPU {vis_cpu} コア ／ GPU {vis_gpu} コア（各色 = 何ラウンド目に処理したか）
    </div>
    """, unsafe_allow_html=True)
    fig_grid = plot_batch_grid(vis_tasks, vis_cpu, vis_gpu)
    st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
    st.pyplot(fig_grid)
    plt.close(fig_grid)
    st.caption("CPUは少ないコアで何度もラウンドを重ねる（色がたくさん）。"
               "GPUはコアが多いので、ほぼ全部を1〜数ラウンドで塗り終える（ほぼ1色）。")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>{icon_html('brain', 'これがAIの学習とどう関係する？', size=20, color=C_CYAN)}</h3>
    ニューラルネットワークの学習の「1ステップ」は、
    数字を格子状に並べた<b>行列（マトリクス）</b>どうしの掛け算・足し算の集まりです。
    その数は1ステップで<b>何百万〜何十億回</b>。しかも、そのステップの中では
    <b>1つ1つの掛け算がすべて独立</b>（別々に計算してよい）。<br><br>
    これはまさに「大量の・同じ・独立した単純タスク」——
    このレースで見た、<b>GPUが圧勝するパターンそのもの</b>です。<br>
    <b style="color:#b45309;">だから、</b>もともと画像処理用だったGPUが計算用に転用されたことで、
    ディープラーニングが「現実的な時間」で学習できるようになりました。
    <hr style="border-color:#cbd5e1; margin:10px 0;">
    <b style="color:#059669;">おさらい：</b>この数字はすべて簡単な数式による教育用モデルで、
    実機の性能測定ではありません。大切なのは「なぜ並列が効くのか」という考え方です。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 3: 行列演算を並列化する
# ================================================================
with tab3:
    st.markdown(f"### {icon_html('atom', 'AIの学習の正体は「行列の掛け算」', size=18, color=C_CYAN)}",
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{icon_html('layers', '行列って何？ なぜ並列化できるの？', size=20, color=C_CYAN)}</h3>
        <b>行列</b>とは、数字を格子状（表のよう）に並べたもの。
        ミッション03で見た「ニューロンに重みを掛けて足す」計算は、
        まとめると<b>行列どうしの掛け算</b>になります。AIの学習は、これを何度もくり返すことです。<br><br>
        行列の掛け算で作られる「答えの表（出力行列）」の<b>1マス1マスは、それぞれ独立して計算できます</b>。
        あるマスを計算するのに、隣のマスの答えは必要ありません。<br><br>
        <b style="color:#c2410c;">CPU的なやり方：</b>マスを<b>1個ずつ順番に</b>計算する（少数のコアで丁寧に）。<br>
        <b style="color:#059669;">GPU的なやり方：</b>たくさんのマスを<b>同時に一気に</b>計算する（大量のコアで並列に）。<br>
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        下で実際に小さな行列を掛け算し、「一度に何マス計算するか」を変えて、
        <b>何ラウンドで全部終わるか</b>を見てみましょう。
        </div>
        """, unsafe_allow_html=True)

    # 再ロール用シード
    if "mat_seed_hw" not in st.session_state:
        st.session_state["mat_seed_hw"] = 7

    mc1, mc2 = st.columns([2, 1])
    with mc1:
        N = st.slider("行列のサイズ N（N×N 行列）", 2, 6, 3, 1, key="mat_n_hw")
    with mc2:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button("行列を作り直す", key="mat_reroll_hw", use_container_width=True):
            st.session_state["mat_seed_hw"] = int(np.random.randint(0, 100000))

    rng = np.random.default_rng(st.session_state["mat_seed_hw"])
    A = rng.integers(0, 10, size=(N, N))
    B = rng.integers(0, 10, size=(N, N))
    Cmat = A @ B

    def _df(mat):
        return pd.DataFrame(
            mat,
            index=[f"行{i+1}" for i in range(mat.shape[0])],
            columns=[f"列{j+1}" for j in range(mat.shape[1])],
        )

    st.markdown(f"#### {icon_html('bar-chart', '行列 A × B = C', size=16, color=C_GREEN)}",
                unsafe_allow_html=True)
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.caption("行列 A")
        st.dataframe(_df(A), use_container_width=True)
    with dcol2:
        st.caption("行列 B")
        st.dataframe(_df(B), use_container_width=True)
    with dcol3:
        st.caption("積 C = A × B（この表の全マスを計算する）")
        st.dataframe(_df(Cmat), use_container_width=True)

    st.markdown(f"""
    <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px;
                font-family:monospace; font-size:0.85rem;">
        <span style="color:#b45309;">計算するマスの総数：</span>
        N × N = {N} × {N} = <span style="color:#059669; font-weight:bold;">{N*N} マス</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {icon_html('sliders', '一度に何マス計算する？', size=18, color=C_YELLOW)}",
                unsafe_allow_html=True)
    st.caption("スライダーを 1（＝CPU：1マスずつ順番に）から N×N（＝GPU：全マス同時に）まで動かして、"
               "必要なラウンド数がどう変わるか見てみよう。")

    total_cells = N * N
    cells_at_once = st.slider("同時に計算するマス目の数", 1, total_cells, 1, 1, key="mat_par_hw")

    fig_mat, mat_rounds = plot_matrix_rounds(N, cells_at_once)

    gcol1, gcol2 = st.columns([3, 2])
    with gcol1:
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        st.pyplot(fig_mat)
        plt.close(fig_mat)
    with gcol2:
        mm1, mm2 = st.columns(2)
        mm1.metric("同時計算マス数", f"{cells_at_once}")
        mm2.metric("必要ラウンド数", f"{mat_rounds}")
        st.markdown(f"""
        <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px;
                    font-family:monospace; font-size:0.85rem;">
            rounds = ceil({total_cells} / {cells_at_once})
            = <span style="color:#059669; font-weight:bold;">{mat_rounds}</span>
        </div>
        """, unsafe_allow_html=True)
        if cells_at_once == 1:
            st.info("いまは CPU 的な動き：1マスずつ順番に計算するので、"
                    f"{total_cells} 回のラウンドが必要です。")
        elif cells_at_once >= total_cells:
            st.success("いまは GPU 的な動き：全マスを一度に計算するので、"
                       "たった1ラウンドで完了！ これが並列化の威力です。")
        else:
            st.warning(f"中間の状態：{cells_at_once} マスずつまとめて計算し、"
                       f"{mat_rounds} ラウンドで完了します。")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:16px;">
    <h3>{icon_html('lightbulb', 'ここから分かること', size=20, color=C_CYAN)}</h3>
    出力行列のマスは全部独立なので、<b>好きなだけ同時に計算してよい</b>——
    ただしCPUはコアが少ないので少しずつしか進められません。
    GPUは数千のコアで<b>ほぼ全マスを一気に</b>計算し、ラウンド数を劇的に減らします。<br><br>
    本物のAIでは、この行列が何千×何千という巨大サイズになり、マスの数は数百万〜数十億。
    その一つ一つが独立した単純な掛け算・足し算です。だからこそ、
    <b>大量の単純コアを持つGPU</b>が学習を現実的な時間に縮めてくれるのです。<br>
    <b style="color:#059669;">これも簡略化した教育モデル</b>で、
    実際のGPUはメモリ転送やスレッド管理などもっと複雑な要素を含みます。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 4: まとめ：AIとハードウェアの歴史
# ================================================================
with tab4:
    st.markdown(f"### {icon_html('book-open', 'なぜ「今」AIが花開いたのか', size=18, color=C_CYAN)}",
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{icon_html('clock', 'アイデアは古い、でも計算力が足りなかった', size=20, color=C_CYAN)}</h3>
        ニューラルネットワークの基本アイデアは、実はとても古く、
        何十年も前から研究されてきました。にもかかわらず長らく「使いもの」にならなかった大きな理由の一つが、
        <b>計算に時間がかかりすぎた</b>ことです。<br><br>
        転機は<b>2010年代</b>。研究者たちが、
        <b>ビデオゲームのグラフィックス描画のために作られたGPU</b>に注目しました。
        3D描画もまた「大量のピクセルに同じ単純計算をする」という<b>超並列・単純計算</b>の問題。
        つまりGPUは、AIの学習に必要な行列演算と<b>同じ形の仕事</b>が得意だったのです。<br><br>
        そこでGPUを<b>グラフィックス用から計算用に転用</b>したところ、学習が桁違いに速くなり、
        「ディープラーニング」が一気に実用段階へ進みました。
        よく知られた節目として、<b>2012年のAlexNet</b>が画像認識のコンテストでGPUを使って大きな成果を上げ、
        以後の流れを決定づけた、と語られます。<br>
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#059669;">誠実な補足：</b>
        ここで挙げた年代（2010年代／2012年AlexNet）は、広く知られている大まかな目安です。
        細かい数値の断定は避け、あくまで「なぜGPUがAIを後押ししたのか」という
        <b>大きな流れ</b>として理解してください。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('scale', 'CPUとGPU、結局どう使い分ける？', size=18, color=C_GREEN)}",
                unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""
        <div class="guide-box" style="border-color:#c2410c;">
        <h4 style="border-left-color:#c2410c; color:#c2410c;">
        {icon_html('cpu', 'CPUの強み', size=16, color='#c2410c')}</h4>
        <p>
        ・少数だが非常に賢いコア<br>
        ・複雑な条件分岐・順番のある処理が得意<br>
        ・プログラム全体の司令塔<br>
        ・臨機応変な判断が必要な作業に強い<br><br>
        <b style="color:#c2410c;">たとえ：</b>1人の熟練シェフ
        </p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="guide-box" style="border-color:#059669;">
        <h4 style="border-left-color:#059669; color:#059669;">
        {icon_html('gpu', 'GPUの強み', size=16, color='#059669')}</h4>
        <p>
        ・数千の単純なコア<br>
        ・同じ計算を大量に同時処理（並列）<br>
        ・行列演算・画像描画が大得意<br>
        ・AIの学習を現実的な時間に縮める<br><br>
        <b style="color:#059669;">たとえ：</b>1000人のアルバイト
        </p>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="guide-box" style="border-color:#0284c7;">
        <h4 style="border-left-color:#0284c7; color:#0284c7;">
        {icon_html('puzzle', 'AIには両方必要', size=16, color='#0284c7')}</h4>
        <p>
        ・CPUが全体の流れを指揮（段取り・制御）<br>
        ・GPUが重い並列計算を一手に引き受ける<br>
        ・二人三脚で初めてAIは動く<br>
        ・どちらが上でもなく、役割が違うだけ<br><br>
        <b style="color:#0284c7;">たとえ：</b>司令塔＋大量の作業員
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"### {icon_html('bar-chart', '早わかり比較表', size=18, color=C_YELLOW)}",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="explanation-box">
    <table style="color:#000000; width:100%; font-size:0.9rem; border-collapse:collapse;">
    <tr style="border-bottom:2px solid #cbd5e1;">
        <th style="text-align:left; padding:8px; color:#b45309;">観点</th>
        <th style="text-align:left; padding:8px; color:#c2410c;">CPU</th>
        <th style="text-align:left; padding:8px; color:#059669;">GPU</th>
    </tr>
    <tr style="border-bottom:1px solid #cbd5e1;">
        <td style="padding:8px; color:#64748b;">コアの数</td>
        <td style="padding:8px;">少ない（数個〜十数個）</td>
        <td style="padding:8px;">とても多い（数千〜）</td>
    </tr>
    <tr style="border-bottom:1px solid #cbd5e1;">
        <td style="padding:8px; color:#64748b;">1コアの性能</td>
        <td style="padding:8px;">高い・柔軟</td>
        <td style="padding:8px;">控えめ・単純</td>
    </tr>
    <tr style="border-bottom:1px solid #cbd5e1;">
        <td style="padding:8px; color:#64748b;">得意な仕事</td>
        <td style="padding:8px;">複雑な判断・順次処理</td>
        <td style="padding:8px;">同じ単純計算の大量並列</td>
    </tr>
    <tr style="border-bottom:1px solid #cbd5e1;">
        <td style="padding:8px; color:#64748b;">AIでの役割</td>
        <td style="padding:8px;">段取り・制御（司令塔）</td>
        <td style="padding:8px;">行列演算の主力（力仕事）</td>
    </tr>
    <tr>
        <td style="padding:8px; color:#64748b;">たとえ</td>
        <td style="padding:8px;">1人の熟練シェフ</td>
        <td style="padding:8px;">1000人のアルバイト</td>
    </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card" style="margin-top:16px;">
    &gt; MISSION 05 まとめ<br>
    &gt; CPU = 少数の天才。GPU = 大量の単純作業員。<br>
    &gt; AIの学習 = 同じ掛け算・足し算を何百万回もくり返す並列作業。<br>
    &gt; だから「1人の速さ」ではなく「全体の処理量」で勝つGPUが主役になる。<br>
    &gt; そしてCPUが指揮し、GPUが力を出す——両輪でAIは動く。
    </div>
    """, unsafe_allow_html=True)

    st.success("これでミッション05は完了！ 次は M06『AIの二形態』で、"
               "デジタルの中のAIと、体を持つAI（ロボット）の違いを探ろう。")

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown('''
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 05: CPU対GPU — 並列計算がAIを動かす</p>
</div>
''', unsafe_allow_html=True)
