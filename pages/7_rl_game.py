import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    page_title="ミッション07: 育成ゲーム | AI Inquiry Lab",
    page_icon=":material/sports_esports:",
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

# 配色（マップのマス種別）
COL_BG    = "#ffffff"   # 空きマスの背景
COL_START = "#0284c7"   # スタート（シアン）
COL_GOAL  = "#059669"   # ゴール（グリーン）
COL_TRAP  = "#dc2626"   # 罠（レッド）
COL_WALL  = "#2a3a52"   # 壁
COL_EMPTY = "#243447"   # 空きマス
COL_PATH  = "#b45309"   # エージェントの通った道（イエロー）
COL_HUMAN = "#c2410c"   # 人間の通った道（オレンジ）
COL_GRID  = "#cbd5e1"   # グリッド線
COL_TXT   = "#000000"

# 報酬設計（このページ全体で共有）
R_GOAL = 10.0     # ゴール到達（＋報酬・エピソード終了）
R_TRAP = -10.0    # 罠にはまる（－罰・エピソード終了）
R_STEP = -0.05    # 1歩あたりの小さなコスト（最短ルートを促す）
R_WALL = -0.1     # 壁にぶつかる（その場にとどまる）

# 行動: 0=上, 1=下, 2=左, 3=右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_LABELS_EN = ["Up", "Down", "Left", "Right"]
ACTION_LABELS_JP = ["上", "下", "左", "右"]

# ---------------------------------------------------------------
# プリセット迷路（左端の列＋最下段は必ず通れるため必ず解ける）
# S=スタート  G=ゴール  #=壁  .=空きマス
# ---------------------------------------------------------------
PRESET_MAZES = {
    "かんたん (5x5)": [
        "S..#.",
        ".#.#.",
        ".#...",
        ".###.",
        "....G",
    ],
    "ふつう (6x6)": [
        "S.....",
        ".####.",
        "...#..",
        ".#.#.#",
        ".#....",
        ".#..#G",
    ],
    "むずかしい (7x7)": [
        "S.....#",
        ".####.#",
        ".#..#..",
        ".#.##.#",
        ".#....#",
        ".####.#",
        "......G",
    ],
}

# ================================================================
# SECTION 3: ユーティリティ
# ================================================================
@st.cache_data
def _read_css_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def load_css(path: str) -> None:
    if os.path.exists(path):
        st.markdown(f"<style>{_read_css_text(path)}</style>", unsafe_allow_html=True)


def find_cell(maze, target):
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            if ch == target:
                return (r, c)
    return None


@st.cache_data
def apply_traps(base_rows, density, seed):
    """プリセット迷路の空きマスの一部を罠(T)に変える。S/G は変えない。"""
    grid = [list(row) for row in base_rows]
    empties = [
        (r, c)
        for r, row in enumerate(grid)
        for c, ch in enumerate(row)
        if ch == "."
    ]
    rng = np.random.RandomState(seed)
    n_traps = int(round(len(empties) * density))
    if n_traps > 0 and empties:
        idx = rng.choice(len(empties), size=min(n_traps, len(empties)), replace=False)
        for i in idx:
            r, c = empties[i]
            grid[r][c] = "T"
    return ["".join(row) for row in grid]


def step_env(maze, r, c, a):
    """1ステップ進める。戻り値: (nr, nc, reward, done, reached_goal)。"""
    nrows, ncols = len(maze), len(maze[0])
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    # 場外 or 壁 → その場にとどまる
    if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols or maze[nr][nc] == "#":
        return r, c, R_WALL, False, False
    cell = maze[nr][nc]
    if cell == "G":
        return nr, nc, R_GOAL, True, True
    if cell == "T":
        return nr, nc, R_TRAP, True, False
    return nr, nc, R_STEP, False, False


# ================================================================
# SECTION 4: Q学習エンジン（純numpy・外部RLライブラリ不使用）
# ================================================================
@st.cache_data(show_spinner=False)
def q_learning_train(maze, episodes, alpha, gamma, epsilon0,
                     epsilon_decay, max_steps=200, seed=0):
    """表形式Q学習。

    戻り値:
      q_table       : shape (nrows*ncols, 4)
      reward_history: 各エピソードの合計報酬 (list[float])
      snapshots     : {episode_index: [(r,c), ...]} 代表エピソードの全経路
      success_hist  : 各エピソードでゴール到達したか (list[0/1])
      epsilon_hist  : 各エピソードの探索率 (list[float])
    """
    nrows, ncols = len(maze), len(maze[0])
    n_states = nrows * ncols
    Q = np.zeros((n_states, 4))
    start = find_cell(maze, "S")
    rng = np.random.RandomState(seed)

    reward_history, success_hist, epsilon_hist = [], [], []
    # 代表エピソード（最初 / 25% / 50% / 最後）の経路を丸ごと記録
    snap_eps = sorted(set([
        0,
        max(0, episodes // 4),
        max(0, episodes // 2),
        episodes - 1,
    ]))
    snapshots = {}

    epsilon = float(epsilon0)
    eps_floor = 0.02

    for ep in range(episodes):
        r, c = start
        total = 0.0
        path = [(r, c)]
        reached_goal = False
        record = ep in snap_eps

        for _ in range(max_steps):
            s = r * ncols + c
            # ε-greedy: 確率εでランダム探索、それ以外は現時点のベスト行動
            if rng.rand() < epsilon:
                a = rng.randint(4)
            else:
                a = int(np.argmax(Q[s]))

            nr, nc, reward, done, goal = step_env(maze, r, c, a)
            s2 = nr * ncols + nc
            # ベルマン更新
            best_next = 0.0 if done else float(np.max(Q[s2]))
            Q[s, a] += alpha * (reward + gamma * best_next - Q[s, a])

            r, c = nr, nc
            total += reward
            if record:
                path.append((r, c))
            if done:
                reached_goal = goal
                break

        reward_history.append(total)
        success_hist.append(1 if reached_goal else 0)
        epsilon_hist.append(epsilon)
        if record:
            snapshots[ep] = path
        # エピソードごとに探索率を減衰（だんだん「経験」を信じるようになる）
        epsilon = max(eps_floor, epsilon * epsilon_decay)

    return Q, reward_history, snapshots, success_hist, epsilon_hist


@st.cache_data
def greedy_rollout(maze, Q, max_steps=200):
    """学習後の方策(argmax)だけで迷路を解いた経路と到達可否を返す。"""
    nrows, ncols = len(maze), len(maze[0])
    r, c = find_cell(maze, "S")
    path = [(r, c)]
    reached = False
    visited = set()
    for _ in range(max_steps):
        s = r * ncols + c
        if s in visited:  # 同じ状態に戻る＝ループ、打ち切り
            break
        visited.add(s)
        a = int(np.argmax(Q[s]))
        nr, nc, _, done, goal = step_env(maze, r, c, a)
        if (nr, nc) == (r, c):  # 壁で動けない
            break
        r, c = nr, nc
        path.append((r, c))
        if done:
            reached = goal
            break
    return path, reached


# ================================================================
# SECTION 5: 描画関数（matplotlib）
# ================================================================
def _new_maze_fig(nrows, ncols, figsize=None):
    if figsize is None:
        figsize = (0.72 * ncols + 1.2, 0.72 * nrows + 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)  # 行が下向きに増える（左上が原点）
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(COL_GRID)
    return fig, ax


def _cell_color(ch):
    return {
        "S": COL_START, "G": COL_GOAL, "T": COL_TRAP,
        "#": COL_WALL, ".": COL_EMPTY,
    }.get(ch, COL_EMPTY)


def draw_maze(maze, path=None, agent_pos=None, path_color=COL_PATH,
              title=None, show_labels=True):
    nrows, ncols = len(maze), len(maze[0])
    fig, ax = _new_maze_fig(nrows, ncols)
    for r, row in enumerate(maze):
        for c, ch in enumerate(row):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=_cell_color(ch),
                                   edgecolor=COL_GRID, linewidth=1.2))
            if show_labels and ch in ("S", "G", "T"):
                ax.text(c + 0.5, r + 0.5, ch, ha="center", va="center",
                        color="#eef2fb", fontsize=13, fontweight="bold")
    # 経路
    if path and len(path) > 1:
        xs = [c + 0.5 for (r, c) in path]
        ys = [r + 0.5 for (r, c) in path]
        ax.plot(xs, ys, color=path_color, lw=2.4, alpha=0.9, zorder=4)
        ax.scatter(xs, ys, color=path_color, s=16, zorder=5, alpha=0.8)
        ax.scatter([xs[0]], [ys[0]], color=path_color, s=90,
                   marker="o", zorder=6, edgecolors="#eef2fb")
    # 現在位置（人間プレイ用）
    if agent_pos is not None:
        r, c = agent_pos
        ax.scatter([c + 0.5], [r + 0.5], color=path_color, s=260,
                   marker="o", zorder=7, edgecolors="#eef2fb", linewidths=1.5)
        ax.text(c + 0.5, r + 0.5, "@", ha="center", va="center",
                color="#eef2fb", fontsize=12, fontweight="bold", zorder=8)
    if title:
        ax.set_title(title, color=COL_TXT, fontsize=11, pad=6)
    plt.tight_layout(pad=0.4)
    return fig


def plot_learning_curve(reward_history):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    for sp in ax.spines.values():
        sp.set_color(COL_GRID)
    ax.tick_params(colors=COL_TXT)
    ep = np.arange(1, len(reward_history) + 1)
    rh = np.array(reward_history, dtype=float)
    ax.plot(ep, rh, color="#cbd5e1", lw=1.0, alpha=0.7, label="Reward / episode")
    # 移動平均でトレンドを見やすく
    win = max(1, len(rh) // 20)
    if win > 1:
        kernel = np.ones(win) / win
        ma = np.convolve(rh, kernel, mode="valid")
        ax.plot(np.arange(win, len(rh) + 1), ma, color="#059669", lw=2.6,
                label=f"Moving avg ({win})")
    ax.axhline(0, color="#dc2626", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Episode", color=COL_TXT, fontsize=9)
    ax.set_ylabel("Total reward", color=COL_TXT, fontsize=9)
    ax.set_title("Learning Curve (reward per episode)", color="#b45309",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, color=COL_GRID)
    ax.legend(labelcolor=COL_TXT, facecolor=COL_BG, edgecolor=COL_GRID, fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig


def plot_policy(maze, Q):
    """maxQ のヒートマップ＋各マスのベスト行動の矢印。"""
    nrows, ncols = len(maze), len(maze[0])
    value = np.full((nrows, ncols), np.nan)
    for r in range(nrows):
        for c in range(ncols):
            ch = maze[r][c]
            if ch in ("#",):
                continue
            value[r, c] = float(np.max(Q[r * ncols + c]))
    masked = np.ma.masked_invalid(value)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(COL_WALL)

    fig, ax = _new_maze_fig(nrows, ncols, figsize=(0.85 * ncols + 1.6, 0.72 * nrows + 1.0))
    im = ax.imshow(masked, extent=[0, ncols, nrows, 0], origin="upper",
                   cmap=cmap, alpha=0.85, zorder=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COL_TXT)
    cbar.set_label("Estimated value (max Q)", color=COL_TXT, fontsize=8)
    cbar.outline.set_edgecolor(COL_GRID)

    for r in range(nrows):
        for c in range(ncols):
            ch = maze[r][c]
            # グリッド線
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="none",
                                   edgecolor=COL_GRID, linewidth=1.0, zorder=2))
            if ch in ("S", "G", "T"):
                ax.text(c + 0.5, r + 0.5, ch, ha="center", va="center",
                        color="#eef2fb", fontsize=12, fontweight="bold", zorder=5)
                continue
            if ch == "#":
                continue
            # 方策の矢印（そのマスでのベスト行動）
            a = int(np.argmax(Q[r * ncols + c]))
            dr, dc = ACTIONS[a]
            ax.annotate("", xy=(c + 0.5 + dc * 0.3, r + 0.5 + dr * 0.3),
                        xytext=(c + 0.5 - dc * 0.15, r + 0.5 - dr * 0.15),
                        arrowprops=dict(arrowstyle="-|>", color="#eef2fb",
                                        lw=1.8), zorder=4)
    ax.set_title("Learned Policy & Value Map", color="#b45309",
                 fontsize=11, pad=6)
    plt.tight_layout(pad=0.4)
    return fig


def plot_q_bars(Q, r, c, ncols):
    s = r * ncols + c
    vals = Q[s]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    for sp in ax.spines.values():
        sp.set_color(COL_GRID)
    ax.tick_params(colors=COL_TXT)
    colors = ["#0284c7", "#c026d3", "#7c3aed", "#b45309"]
    best = int(np.argmax(vals))
    bar_colors = [colors[i] if i == best else "#cbd5e1" for i in range(4)]
    ax.bar(ACTION_LABELS_EN, vals, color=bar_colors, edgecolor="#eef2fb")
    ax.axhline(0, color=COL_GRID, lw=0.8)
    ax.set_ylabel("Q value", color=COL_TXT, fontsize=9)
    ax.set_title(f"Q values at cell (row={r}, col={c})", color=COL_TXT, fontsize=10)
    ax.grid(True, axis="y", alpha=0.15, color=COL_GRID)
    plt.tight_layout(pad=0.5)
    return fig, best, vals


# ================================================================
# SECTION 6: CSSロード & ページタイトル
# ================================================================
load_css(CSS_FILE)
render_top_nav("rlgame")

# ページ固有の小さなスコープ付きスタイル（rl- 接頭辞）
st.markdown("""
<style>
.rl-reward-table { width:100%; border-collapse:collapse; font-size:0.9rem; }
.rl-reward-table th, .rl-reward-table td {
    border:1px solid #cbd5e1; padding:7px 10px; text-align:left; color:#000000;
}
.rl-reward-table th { color:#b45309; background:#eef2fb; }
.rl-chip {
    display:inline-flex; align-items:center; gap:0.4em; padding:3px 10px;
    border-radius:12px; font-size:0.78rem; font-weight:bold; margin:2px;
}
.rl-legend { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0; }
@media (max-width: 640px) {
    .rl-reward-table th, .rl-reward-table td { padding:5px 6px; font-size:0.8rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("gamepad", size=32)} 育成ゲーム</h1>
    <p class="sub-title-text">MISSION 07 — 報酬と失敗の繰り返しでAIエージェントを鍛えろ</p>
</div>
''', unsafe_allow_html=True)

# ================================================================
# SECTION 7: セッション状態の初期化
# ================================================================
for k, v in {
    "rl_result": None,
    "rl_human_pos": None,
    "rl_human_path": None,
    "rl_human_steps": 0,
    "rl_human_done": False,
    "rl_human_win": False,
    "rl_human_gaveup": False,
    "rl_train_count": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
# SECTION 8: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#b45309; font-weight:bold; font-size:0.9rem;">
        {svg_icon("gamepad", size=15, color="#b45309")} RL AGENT MODE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"{icon_html('sliders', '迷路の設定', size=15, color='#0284c7')}",
                unsafe_allow_html=True)
    maze_choice_sb = st.selectbox(
        "迷路プリセット", list(PRESET_MAZES.keys()), key="maze_choice_rl",
    )
    trap_density_sb = st.slider(
        "罠の密度", 0.0, 0.30, 0.10, 0.02, key="trap_density_rl",
        help="空きマスのうち何割を罠にするか",
    )
    if "trap_seed_rl" not in st.session_state:
        st.session_state["trap_seed_rl"] = 7
    if st.button("罠を配置しなおす",
                 use_container_width=True, key="reshuffle_rl"):
        st.session_state["trap_seed_rl"] = int(np.random.randint(0, 10_000))

    st.divider()
    st.markdown(f"{icon_html('compass', 'Navigation', size=15, color='#059669')}",
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

# アクティブな迷路を構築（全タブで共有）
active_maze = apply_traps(
    PRESET_MAZES[maze_choice_sb], trap_density_sb, st.session_state["trap_seed_rl"]
)
maze_nrows, maze_ncols = len(active_maze), len(active_maze[0])

# ================================================================
# SECTION 9: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{svg_icon("gamepad", color="#0284c7")} 強化学習（Reinforcement Learning）とは？</h3>
AIエージェントが<b>「報酬（ごほうび）」</b>と<b>「罰（ペナルティ）」</b>だけを手がかりに、
何度も試行錯誤を繰り返しながら<b>自力で賢くなっていく</b>仕組みです。<br><br>
これは<b>犬のしつけ</b>とそっくりです。「おすわり」ができたらオヤツ（＋報酬）、
イタズラをしたら叱る（－罰）。犬は言葉の意味を教わらなくても、
「どう動けばごほうびがもらえるか」を経験から学びます。<br><br>
このミッションでは、迷路に置かれたAIエージェントが最初は<b>完全にランダム</b>にさまよう状態から、
ゴール（＋10）と罠（−10）の信号だけを頼りに、
<b>何百回もの失敗を重ねて最短ルートを自力で発見する</b>様子を、あなた自身の手で観察します。
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 迷路をデザインしよう",
    "② AIエージェントを学習させる",
    "③ Q値と方策を覗いてみよう",
    "④ 人間 vs AI 対戦",
])

# ---------------------------------------------------------------
# TAB 1: 迷路デザイン
# ---------------------------------------------------------------
with tab1:
    st.markdown(heading("迷路と報酬のルールを決めよう", "map", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        c_txt, c_map = st.columns([3, 2])

        with c_txt:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("target", color="#0284c7")} エージェントに見えているもの・見えていないもの</h3>
            AIエージェントは迷路の中にポツンと置かれた1体のコマです。<br>
            エージェントが<b>知っているのは「今どのマスにいるか」だけ</b>。
            そして選べる行動は<b>上・下・左・右の4方向</b>に1歩動くことだけです。<br><br>
            <b style="color:#dc2626;">重要：</b>エージェントには
            「ゴールがどこにあるか」も「どう進めば正解か」も<b>一切教えません</b>。
            動いた<b>後で</b>もらえる報酬／罰の数字だけが唯一のヒントです。<br>
            <ul>
                <li><b style="color:#059669;">ゴール</b>に着いたら <b>＋{R_GOAL:.0f}</b>（大成功・そこで1回終了）</li>
                <li><b style="color:#dc2626;">罠</b>にはまったら <b>{R_TRAP:.0f}</b>（失敗・そこで1回終了）</li>
                <li><b>空きマス</b>を1歩進むごとに <b>{R_STEP}</b>（遠回りするほど損）</li>
                <li><b>壁</b>にぶつかると <b>{R_WALL}</b>（進めずその場に足踏み）</li>
            </ul>
            空きマスにわずかなマイナスを付けるのがコツです。こうすると
            「だらだら遠回りするより<b>最短でゴールしたほうが得</b>」という動機が生まれます。
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <table class="rl-reward-table">
              <tr><th>マスの種類</th><th>報酬</th><th>意味</th></tr>
              <tr><td>ゴール (G)</td><td style="color:#059669;">+{R_GOAL:.0f}</td><td>エピソード成功・終了</td></tr>
              <tr><td>罠 (T)</td><td style="color:#dc2626;">{R_TRAP:.0f}</td><td>エピソード失敗・終了</td></tr>
              <tr><td>空きマス (.)</td><td style="color:#b45309;">{R_STEP}</td><td>1歩ぶんのコスト</td></tr>
              <tr><td>壁 (#)</td><td style="color:#c2410c;">{R_WALL}</td><td>進めず足踏み</td></tr>
              <tr><td>スタート (S)</td><td>0</td><td>毎回ここから開始</td></tr>
            </table>
            """, unsafe_allow_html=True)

        with c_map:
            st.markdown(f"""
            <div class="rl-legend">
              <span class="rl-chip" style="background:#0284c733; color:#0284c7;">■ スタート</span>
              <span class="rl-chip" style="background:#05966933; color:#059669;">■ ゴール</span>
              <span class="rl-chip" style="background:#dc262633; color:#dc2626;">■ 罠</span>
              <span class="rl-chip" style="background:#2a3a5288; color:#64748b;">■ 壁</span>
            </div>
            """, unsafe_allow_html=True)
            fig_prev = draw_maze(active_maze, title=f"Preview: {maze_choice_sb}")
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_prev)
            plt.close(fig_prev)

            n_traps = sum(row.count("T") for row in active_maze)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("マップサイズ", f"{maze_nrows}x{maze_ncols}")
            mc2.metric("罠の数", f"{n_traps} 個")
            mc3.metric("総マス数", f"{maze_nrows * maze_ncols}")

    st.info(
        "左サイドバーの「迷路プリセット」「罠の密度」を変えたり、"
        "「罠を配置しなおす」で配置をランダムに変えられます。設定できたら ② のタブへ！",
    )

# ---------------------------------------------------------------
# TAB 2: 学習
# ---------------------------------------------------------------
with tab2:
    st.markdown(heading("AIエージェントを何百回も走らせて鍛える", "dumbbell", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("refresh-cw", color="#0284c7")} 「試行錯誤（トライ＆エラー）」を高速で回す</h3>
        1回のチャレンジ（スタートから終了まで）を<b>エピソード</b>と呼びます。<br>
        最初のエピソードでは、エージェントは何も知らないので<b>ほぼランダム</b>にさまよい、
        たいてい罠にはまるか時間切れになります。でも大丈夫。<br>
        その1回1回の結果が<b>「この場所でこう動いたら、後でどれくらい報酬がもらえたか」</b>という
        メモ（Q値）として少しずつ蓄積されていきます。<br><br>
        これを何百回も繰り返すと、エージェントは<b>「ここでは右に行くと得をしやすい」</b>という
        地図なきカンを自力で身につけ、やがて最短ルートへ収束します。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">{svg_icon("sliders", color="#b45309")} 4つのツマミの意味</b>
        <ul>
          <li><b>エピソード数</b>：練習の回数。多いほど賢くなるが時間がかかる。</li>
          <li><b>学習率 α</b>：1回の経験をどれだけ強く信じるか。大きいと素早いが不安定。</li>
          <li><b>割引率 γ</b>：「先の報酬」をどれだけ重視するか。1に近いほど遠いゴールを見据える。</li>
          <li><b>探索率 ε</b>：最初にどれだけ冒険（ランダム行動）するか。学習が進むと自動で下がる。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("学習パラメータを設定して実行", "play", level=3),
                unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        episodes_rl = st.slider("エピソード数（練習回数）", 10, 500, 300, 10,
                                key="episodes_rl")
        alpha_rl = st.slider("学習率 α", 0.01, 1.0, 0.20, 0.01,
                             key="alpha_rl")
    with p2:
        gamma_rl = st.slider("割引率 γ", 0.50, 0.99, 0.95, 0.01,
                             key="gamma_rl")
        epsilon_rl = st.slider("探索率 ε（開始時）", 0.05, 1.0, 1.0, 0.05,
                               key="epsilon_rl")

    run_col, info_col = st.columns([1, 2])
    with run_col:
        run_clicked = st.button("学習を実行",
                                use_container_width=True, key="run_train_rl")
    with info_col:
        st.caption(
            "探索率εは、開始値から終了までに約0.02へ自動で減衰します"
            "（最初は大胆に冒険し、後半は学んだ経験を信じる）。"
        )

    if run_clicked:
        # εが最終的に約0.02へ減衰するよう1エピソードあたりの減衰率を逆算
        target = 0.02
        if epsilon_rl > target and episodes_rl > 1:
            decay = (target / epsilon_rl) ** (1.0 / max(1, episodes_rl - 1))
        else:
            decay = 1.0
        with st.spinner("エージェントが迷路を何百回も試行中..."):
            Q, rewards, snaps, succ, eps_hist = q_learning_train(
                active_maze, episodes_rl, alpha_rl, gamma_rl,
                epsilon_rl, decay, seed=0,
            )
            g_path, g_reached = greedy_rollout(active_maze, Q)
        st.session_state["rl_result"] = {
            "Q": Q, "rewards": rewards, "snaps": snaps, "succ": succ,
            "eps_hist": eps_hist, "maze": active_maze,
            "greedy_path": g_path, "greedy_reached": g_reached,
            "greedy_len": len(g_path) - 1,
            "episodes": episodes_rl,
            "maze_name": maze_choice_sb,
        }
        st.session_state["rl_train_count"] += 1

    result = st.session_state.get("rl_result")

    if not result:
        st.info("上の「学習を実行」ボタンを押して、AIの成長を観察しよう！")
    else:
        rewards = result["rewards"]
        succ = np.array(result["succ"])
        eps_used = result["episodes"]
        # 直近20%の成功率
        tail = max(1, eps_used // 5)
        success_rate = float(np.mean(succ[-tail:])) * 100
        first_reward = rewards[0]
        last_reward = rewards[-1]

        st.markdown(heading("学習結果レポート", "bar-chart", level=3),
                    unsafe_allow_html=True)

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("実行エピソード数", f"{eps_used}")
        rm2.metric("最初の報酬", f"{first_reward:.1f}")
        rm3.metric("最後の報酬", f"{last_reward:.1f}", f"{last_reward - first_reward:+.1f}")
        rm4.metric(f"成功率（終盤{tail}回）", f"{success_rate:.0f}%")

        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        fig_lc = plot_learning_curve(rewards)
        st.pyplot(fig_lc)
        plt.close(fig_lc)

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("trending-up", color="#059669")} このカーブが「右肩上がり」なら学習成功</h3>
        グラフの緑の線（移動平均）が<b>右に行くほど上がっていれば</b>、
        エージェントは回を重ねるごとに<b>より多くの報酬を得られるようになった</b>＝賢くなった証拠です。<br>
        序盤は報酬が大きくマイナス（ランダムに罠へ突っ込む）ですが、
        後半は安定してプラス（最短でゴール）へ近づいていきます。
        カーブがなかなか上がらないときは、エピソード数を増やす・学習率αを上げる・
        罠の密度を下げる、などを試してみましょう。
        </div>
        """, unsafe_allow_html=True)

        st.markdown(heading("学習前 vs 学習後：同じ迷路での動きを比べる", "contrast", level=3),
                    unsafe_allow_html=True)

        snaps = result["snaps"]
        ep_keys = sorted(snaps.keys())
        before_key = ep_keys[0]
        after_key = ep_keys[-1]

        bc1, bc2 = st.columns(2)
        with bc1:
            fig_b = draw_maze(result["maze"], path=snaps[before_key],
                              path_color=COL_TRAP,
                              title=f"BEFORE: episode {before_key + 1} (random)")
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_b)
            plt.close(fig_b)
            st.caption(
                f"最初のエピソード。{len(snaps[before_key]) - 1} 歩さまよって"
                "終了。ほぼランダムに動いています。"
            )
        with bc2:
            fig_a = draw_maze(result["maze"], path=snaps[after_key],
                              path_color=COL_PATH,
                              title=f"AFTER: episode {after_key + 1} (trained)")
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_a)
            plt.close(fig_a)
            if result["greedy_reached"]:
                st.caption(
                    f"学習後の方策で解くと最短 {result['greedy_len']} 歩でゴール。"
                    "無駄のないルートに収束しました。"
                )
            else:
                st.caption(
                    "学習後のエピソード。まだ完全収束していないかも。"
                    "エピソード数を増やして再挑戦してみましょう。"
                )

        if success_rate >= 80:
            st.success(
                f"見事！ 終盤の成功率 {success_rate:.0f}% — エージェントは"
                "自力で迷路の解き方を身につけました。",
            )
        elif success_rate >= 40:
            st.warning(
                f"成功率 {success_rate:.0f}% — あと一歩。エピソード数を増やすか、"
                "罠の密度を下げて再挑戦してみましょう。",
            )
        else:
            st.error(
                f"成功率 {success_rate:.0f}% — まだ迷子です。エピソード数を増やす／"
                "割引率γを上げる／罠を減らす、を試してみて。",
            )

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("lightbulb", color="#b45309")} ここが「教師あり学習」との決定的な違い</h3>
        注目すべきは、エージェントに<b>「この場面ではこう動くのが正解」という手本を一切与えていない</b>点です。
        与えたのは<b>行動した後の報酬／罰という数字だけ</b>。それでも試行錯誤の繰り返しから、
        良い戦略（方策）を自分で見つけ出しました。<br><br>
        これは <b>M03「AI育成」</b> で体験した教師あり学習
        （＝1問ごとに「正解ラベル」を与えて誤差を減らす方式）とは根本的に異なります。
        強化学習では正解ラベルが存在せず、<b>「どう動けば将来トクするか」を経験から自分で組み立てる</b>のです。
        囲碁AIのAlphaGoやロボットの歩行制御は、この仕組みで人間を超えました。
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: Q値と方策
# ---------------------------------------------------------------
with tab3:
    st.markdown(heading("エージェントの「頭の中」を覗く", "brain", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("help-circle", color="#0284c7")} 「Q値」ってなに？</h3>
        <b>Q値</b>とは、ひとことで言うと——<br>
        <b style="color:#b45309;">「その場所で・その方向に動いたら、この先もらえる報酬の合計がどれくらいになりそうか」
        というAIの見積もり</b>です。<br><br>
        各マスには「上・下・左・右」それぞれのQ値があり、エージェントは
        <b>いちばんQ値が高い方向</b>を選びます。これが<b>方策（ポリシー）</b>です。<br>
        下の図では、各マスの<b>背景色</b>がそのマスの価値（一番高いQ値）、
        <b>矢印</b>がそのマスでのベスト行動を表します。
        緑っぽいマスほど「ゴールに近くて価値が高い」、赤っぽいマスほど「危険・遠い」という意味です。
        </div>
        """, unsafe_allow_html=True)

    result = st.session_state.get("rl_result")
    if not result:
        st.info("まず ② のタブでエージェントを学習させてください。"
                "学習後、ここに方策マップが表示されます。")
    else:
        maze = result["maze"]
        Q = result["Q"]
        ncols = len(maze[0])

        pc1, pc2 = st.columns([3, 2])
        with pc1:
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            fig_pol = plot_policy(maze, Q)
            st.pyplot(fig_pol)
            plt.close(fig_pol)
            st.caption(
                "矢印＝そのマスでのベスト行動、背景の色＝そのマスの価値(max Q)。"
                "矢印をつないでいくとゴールへの最短ルートが浮かび上がります。"
            )

        with pc2:
            st.markdown(heading("特定のマスを調べる", "search", level=3),
                        unsafe_allow_html=True)
            valid_cells = [
                (r, c)
                for r in range(len(maze))
                for c in range(ncols)
                if maze[r][c] not in ("#", "G", "T")
            ]
            labels = [f"row={r}, col={c}" + (" (S)" if maze[r][c] == "S" else "")
                      for (r, c) in valid_cells]
            sel = st.selectbox("マスを選ぶ", range(len(valid_cells)),
                               format_func=lambda i: labels[i], key="cell_sel_rl")
            sel_r, sel_c = valid_cells[sel]

            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            fig_q, best_a, qvals = plot_q_bars(Q, sel_r, sel_c, ncols)
            st.pyplot(fig_q)
            plt.close(fig_q)

            best_label = ACTION_LABELS_JP[best_a]
            # そのマスから best 方向に進んだ先が何か
            dr, dc = ACTIONS[best_a]
            nr, nc = sel_r + dr, sel_c + dc
            if 0 <= nr < len(maze) and 0 <= nc < ncols:
                nxt = maze[nr][nc]
            else:
                nxt = "#"
            reason = {
                "G": "その方向にゴールがあるから、迷わずそこへ向かいます。",
                "T": "本来は避けたい方向ですが、ほかがもっと悪いと学んだのかもしれません。",
                "#": "壁側ですが、遠回りしてでもそちらから回る学習結果のようです。",
                ".": "その先のマスのほうがゴールに近く、価値が高いと見積もっているからです。",
                "S": "スタート方向。まだ十分に学習が進んでいない可能性があります。",
            }.get(nxt, "その先のほうが将来の報酬が大きいと見積もっています。")

            st.markdown(f"""
            <div class="guide-box">
              <h4>{svg_icon("target", color="#059669")} このマスの学習結果</h4>
              <p>マス (row={sel_r}, col={sel_c}) で、エージェントが選ぶベスト行動は
              <b style="color:#b45309;">「{best_label}」</b>（Q値 {qvals[best_a]:.2f}）。<br>
              {reason}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("map", color="#b45309")} 方策マップの読み方</h3>
        矢印をスタートからたどっていくと、<b>エージェントが考える最短ルート</b>が見えてきます。
        罠(T)のマスの周りでは、矢印が<b>罠を避けるように</b>迂回しているはずです。
        これは「罠に入ると−{-R_TRAP:.0f}という大きな罰を受ける」という経験が、
        近くのマスのQ値を下げ、<b>危険地帯として学習された</b >結果です。<br>
        ゴールに近いマスほど背景が緑（価値が高い）で、遠いマスや行き止まりほど赤に近づきます。
        エージェントは地図を渡されていないのに、報酬の伝播だけでこの「価値の地形」を描き上げたのです。
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 人間 vs AI
# ---------------------------------------------------------------
with tab4:
    st.markdown(heading("同じ迷路であなたも挑戦！ 人間 vs AI", "trophy", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("gamepad", color="#0284c7")} ルール</h3>
        AIが学習したのと<b>まったく同じ迷路</b>を、あなたが手動で解きます。
        上・下・左・右のボタンで1歩ずつ進み、<b>ゴール(G)を目指しましょう</b>。
        罠(T)を踏むと即アウトです。<br>
        ゴールしたら（または「あきらめる」を押したら）、あなたの歩数と、
        <b>AIが何百回もの失敗の末に見つけた最短ルートの歩数</b>を比べます。<br><br>
        <b style="color:#b45309;">人間はふつう、迷路を上から見渡して直感で解けます。
        でもAIは迷路の全体像を一度も見ていません。</b>
        その違いを味わってください。
        </div>
        """, unsafe_allow_html=True)

    # 人間プレイの対象は「現在サイドバーで選ばれている迷路」
    human_maze = active_maze
    start_pos = find_cell(human_maze, "S")

    # 迷路が切り替わったら人間の進行状況をリセット
    if st.session_state["rl_human_pos"] is None or \
       st.session_state.get("rl_human_maze_sig") != (maze_choice_sb, tuple(human_maze)):
        st.session_state["rl_human_pos"] = start_pos
        st.session_state["rl_human_path"] = [start_pos]
        st.session_state["rl_human_steps"] = 0
        st.session_state["rl_human_done"] = False
        st.session_state["rl_human_win"] = False
        st.session_state["rl_human_gaveup"] = False
        st.session_state["rl_human_maze_sig"] = (maze_choice_sb, tuple(human_maze))

    def _human_move(action):
        if st.session_state["rl_human_done"]:
            return
        r, c = st.session_state["rl_human_pos"]
        nr, nc, _, done, goal = step_env(human_maze, r, c, action)
        if (nr, nc) == (r, c):
            return  # 壁：動かない・歩数も増やさない
        st.session_state["rl_human_pos"] = (nr, nc)
        st.session_state["rl_human_path"].append((nr, nc))
        st.session_state["rl_human_steps"] += 1
        if done:
            st.session_state["rl_human_done"] = True
            st.session_state["rl_human_win"] = goal

    def _human_reset():
        st.session_state["rl_human_pos"] = start_pos
        st.session_state["rl_human_path"] = [start_pos]
        st.session_state["rl_human_steps"] = 0
        st.session_state["rl_human_done"] = False
        st.session_state["rl_human_win"] = False
        st.session_state["rl_human_gaveup"] = False

    ctrl_col, view_col = st.columns([1, 1])

    with ctrl_col:
        st.markdown(heading("操作パネル", "sliders", level=3), unsafe_allow_html=True)
        disabled = st.session_state["rl_human_done"]

        # 上
        _, up_c, _ = st.columns([1, 1, 1])
        up_c.button("上", key="btn_up_rl",
                    on_click=_human_move, args=(0,), disabled=disabled,
                    use_container_width=True)
        # 左・下・右
        l_c, d_c, r_c = st.columns([1, 1, 1])
        l_c.button("左", key="btn_left_rl",
                   on_click=_human_move, args=(2,), disabled=disabled,
                   use_container_width=True)
        d_c.button("下", key="btn_down_rl",
                   on_click=_human_move, args=(1,), disabled=disabled,
                   use_container_width=True)
        r_c.button("右", key="btn_right_rl",
                   on_click=_human_move, args=(3,), disabled=disabled,
                   use_container_width=True)

        st.divider()
        act1, act2 = st.columns(2)
        act1.button("リセット", key="btn_reset_rl",
                    on_click=_human_reset, use_container_width=True)
        if act2.button("あきらめる", key="btn_giveup_rl",
                       use_container_width=True, disabled=disabled):
            st.session_state["rl_human_done"] = True
            st.session_state["rl_human_gaveup"] = True

        st.metric("あなたの歩数", f"{st.session_state['rl_human_steps']} 歩")

    with view_col:
        st.markdown(heading("あなたの現在位置", "walk", level=3), unsafe_allow_html=True)
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        fig_h = draw_maze(
            human_maze,
            path=st.session_state["rl_human_path"],
            agent_pos=st.session_state["rl_human_pos"],
            path_color=COL_HUMAN,
            title="Your run",
        )
        st.pyplot(fig_h)
        plt.close(fig_h)

    # 結果の比較
    if st.session_state["rl_human_done"]:
        human_steps = st.session_state["rl_human_steps"]
        result = st.session_state.get("rl_result")
        ai_trained_same = (
            result is not None
            and result.get("maze") == human_maze
            and result.get("greedy_reached")
        )

        if st.session_state["rl_human_gaveup"]:
            st.warning("あきらめました。リセットしてもう一度挑戦できます。")
        elif st.session_state["rl_human_win"]:
            st.success(f"ゴール到達！ あなたは {human_steps} 歩でクリアしました！")
        else:
            st.error("罠にはまってしまいました。リセットして再挑戦しよう。")

        st.markdown(heading("人間 vs AI 対戦結果", "scale", level=3),
                    unsafe_allow_html=True)

        if not ai_trained_same:
            st.info(
                "この迷路でのAIの最短ルートがまだありません。"
                "② のタブで（今と同じ迷路設定のまま）学習を実行すると、"
                "ここでAIの歩数と比較できます。",
            )
        else:
            ai_steps = result["greedy_len"]
            cmp1, cmp2 = st.columns(2)
            cmp1.metric("あなたの歩数", f"{human_steps} 歩")
            cmp2.metric("AIの最短歩数", f"{ai_steps} 歩",
                        f"{human_steps - ai_steps:+d}")

            if st.session_state["rl_human_win"] and human_steps <= ai_steps:
                st.success(
                    "人間の直感は強力です！ 迷路を一目で見渡してひらめく力は、"
                    "AIにはない武器。ただしAIは同じ迷路を何度でも疲れずに、"
                    "文句も言わず解き続けられる——そこが人間との違いです。",
                )
            elif st.session_state["rl_human_win"]:
                st.info(
                    "AIのほうが少ない歩数でした。でも忘れないで——AIは"
                    f"何百回も罠にはまり失敗を繰り返して、ようやくこの最短 {ai_steps} 歩を"
                    "見つけたのです。人間が初見でこれだけ解ければ十分すごい！",
                )
            else:
                st.info(
                    f"今回はゴールできませんでしたが、AIも最初の数百回は"
                    "あなたと同じように失敗続きでした。何度も試すのが強化学習の本質。"
                    "リセットしてリベンジしよう！",
                )

        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("users", color="#059669")} 人間とAI、学び方はこんなに違う</h3>
        <ul>
          <li><b>人間</b>：迷路を<b>俯瞰</b>して、経験や常識から一瞬でルートをひらめく。
          少ない試行で解けるが、疲れるし飽きる。</li>
          <li><b>AI（強化学習）</b>：全体像は見えず、報酬という数字だけが頼り。
          <b>膨大な失敗</b>が必要だが、一度学べば<b>疲れず・ブレず・高速で</b>解き続ける。</li>
        </ul>
        どちらが上・下ではなく、<b>得意分野が違う</b>のです。
        現実のAIは、この「大量の試行錯誤に耐える力」を活かして、
        人間には根気が続かない領域（創薬の候補探索、データセンターの冷却最適化など）で活躍しています。
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# SECTION 10: フッター
# ================================================================
st.markdown('''
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 07: 育成ゲーム — 強化学習は試行錯誤から生まれる</p>
</div>
''', unsafe_allow_html=True)
