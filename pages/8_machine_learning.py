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
    page_title="ミッション08: 機械学習の仕組み | AI Inquiry Lab",
    page_icon=":material/model_training:",
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

# --- 白テーマの配色（グラフ用） -------------------------------------------
COL_BG    = "#ffffff"
COL_GRID  = "#cbd5e1"
COL_TXT   = "#000000"
COL_DIM   = "#94a3b8"

# アクセントパレット（クラス/クラスタの色分けに使用）
PALETTE = ["#0284c7", "#c026d3", "#059669", "#b45309", "#7c3aed", "#c2410c"]

# 教師あり学習デモ（フルーツ）のクラス定義
FRUIT_NAMES    = ["リンゴ", "ブルーベリー", "レモン"]
FRUIT_NAMES_EN = ["Apple", "Blueberry", "Lemon"]
FRUIT_COLORS   = ["#dc2626", "#7c3aed", "#b45309"]
FRUIT_CENTERS  = [(7.0, 6.6), (6.0, 2.2), (2.3, 5.6)]
FRUIT_SPREAD   = [0.85, 0.80, 0.85]


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


# ---------------------------------------------------------------
# 教師あり学習：ラベル付きデータ生成 & KNN（手書き・sklearn不使用）
# ---------------------------------------------------------------
@st.cache_data
def make_fruit_data(seed: int):
    """3クラスのラベル付き2Dデータ（x=甘さ, y=大きさ）を作る。"""
    rng = np.random.RandomState(seed)
    n_per = 16
    X, y = [], []
    for j, (cx, cy) in enumerate(FRUIT_CENTERS):
        pts = rng.randn(n_per, 2) * FRUIT_SPREAD[j] + np.array([cx, cy])
        X.append(pts)
        y += [j] * n_per
    X = np.clip(np.vstack(X), 0.3, 9.7)
    return X, np.array(y)


@st.cache_data
def knn_predict(X, y, query, k):
    """K近傍法を素のnumpyで実装。

    戻り値: (予測ラベル, 近傍のインデックス配列, 全点への距離配列)
    多数決。同数タイのときは「一番近い点のクラス」で決着させる。
    """
    dists = np.linalg.norm(X - np.asarray(query), axis=1)
    order = np.argsort(dists)          # 距離が近い順に並べたインデックス
    knn_idx = order[:k]                # 近い方から k 個
    knn_labels = y[knn_idx]
    classes, counts = np.unique(knn_labels, return_counts=True)
    top = counts.max()
    tied = classes[counts == top]
    if len(tied) == 1:
        pred = int(tied[0])
    else:
        # タイ：k近傍の中で最も近い点（=先頭）の属するタイクラスを採用
        pred = int(knn_labels[0])
        for lbl in knn_labels:
            if lbl in tied:
                pred = int(lbl)
                break
    return pred, knn_idx, dists


def plot_knn(X, y, query=None, knn_idx=None, pred=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    for sp in ax.spines.values():
        sp.set_color(COL_GRID)
    ax.tick_params(colors=COL_TXT)

    # 近傍への線（背面）
    if query is not None and knn_idx is not None:
        for i in knn_idx:
            ax.plot([query[0], X[i, 0]], [query[1], X[i, 1]],
                    color=COL_DIM, lw=1.0, ls="--", zorder=1, alpha=0.9)

    # ラベル付き学習データ（クラスごとに色分け）
    for j in range(len(FRUIT_NAMES_EN)):
        m = y == j
        ax.scatter(X[m, 0], X[m, 1], color=FRUIT_COLORS[j], s=70,
                   edgecolors="#ffffff", linewidths=1.0, zorder=3,
                   label=FRUIT_NAMES_EN[j])

    # 分類したい新しい点（星）
    if query is not None:
        pcol = FRUIT_COLORS[pred] if pred is not None else COL_TXT
        ax.scatter([query[0]], [query[1]], color=pcol, s=460, marker="*",
                   edgecolors="#000000", linewidths=1.4, zorder=5,
                   label="New point")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Feature 1: Sweetness", color=COL_TXT, fontsize=9)
    ax.set_ylabel("Feature 2: Size", color=COL_TXT, fontsize=9)
    ax.set_title("Supervised: K-Nearest Neighbors", color="#0284c7",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, color=COL_GRID)
    ax.legend(labelcolor=COL_TXT, facecolor=COL_BG, edgecolor=COL_GRID,
              fontsize=8, loc="upper left")
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------
# 教師なし学習：ラベルなし点群 & k-means（手書き・sklearn不使用）
# ---------------------------------------------------------------
@st.cache_data
def make_cloud_data(seed: int):
    """ラベルを一切持たない2D点群（内部的には4つの塊から生成）。"""
    rng = np.random.RandomState(seed)
    hidden_centers = [(2.5, 2.8), (7.4, 2.4), (2.2, 7.3), (7.6, 7.4)]
    n_per = 16
    pts = []
    for cx, cy in hidden_centers:
        pts.append(rng.randn(n_per, 2) * 0.85 + np.array([cx, cy]))
    return np.clip(np.vstack(pts), 0.3, 9.7)


def km_init_centroids(points, k, seed):
    """k個の初期セントロイドをデータ点からランダムに選ぶ（k-means++風の素朴版）。"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(points), size=k, replace=False)
    return points[idx].copy()


def km_assign(points, centroids):
    """各点を最も近いセントロイドに割り当てる（割り当てラベル配列を返す）。"""
    # 距離行列: shape (N, k)
    d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def km_update(points, assignments, centroids):
    """各セントロイドを、割り当てられた点の平均へ移動する。空クラスタは据え置き。"""
    new_c = centroids.copy()
    for j in range(len(centroids)):
        m = assignments == j
        if np.any(m):
            new_c[j] = points[m].mean(axis=0)
    return new_c


def plot_kmeans(points, centroids, assignments, k):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    for sp in ax.spines.values():
        sp.set_color(COL_GRID)
    ax.tick_params(colors=COL_TXT)

    if assignments is None:
        # まだグループ分けされていない＝全部グレー（「ラベルが無い」ことを強調）
        ax.scatter(points[:, 0], points[:, 1], color=COL_DIM, s=70,
                   edgecolors="#ffffff", linewidths=1.0, zorder=3,
                   label="Unlabeled")
    else:
        for j in range(k):
            m = assignments == j
            ax.scatter(points[m, 0], points[m, 1], color=PALETTE[j % len(PALETTE)],
                       s=70, edgecolors="#ffffff", linewidths=1.0, zorder=3,
                       label=f"Group {j + 1}")

    if centroids is not None:
        for j in range(len(centroids)):
            ax.scatter([centroids[j, 0]], [centroids[j, 1]],
                       color=PALETTE[j % len(PALETTE)], s=340, marker="X",
                       edgecolors="#000000", linewidths=1.8, zorder=6)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Feature 1", color=COL_TXT, fontsize=9)
    ax.set_ylabel("Feature 2", color=COL_TXT, fontsize=9)
    ax.set_title("Unsupervised: K-means Clustering", color="#c026d3",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, color=COL_GRID)
    ax.legend(labelcolor=COL_TXT, facecolor=COL_BG, edgecolor=COL_GRID,
              fontsize=8, loc="upper left")
    plt.tight_layout(pad=0.5)
    return fig


# ================================================================
# SECTION 4: CSSロード & ページ固有スタイル
# ================================================================
load_css(CSS_FILE)
render_top_nav("ml")

st.markdown("""
<style>
.ml8-flow {
    display:flex; flex-wrap:wrap; align-items:stretch;
    gap:6px; justify-content:center; margin:12px 0;
}
.ml8-flow-box {
    flex:1 1 130px; background:#ffffff; border:2px solid #cbd5e1;
    border-radius:10px; padding:12px 10px; text-align:center;
    box-shadow:3px 3px 0 rgba(15,23,42,0.12);
}
.ml8-flow-box .ml8-fl-t { color:#0284c7; font-weight:900; font-size:0.92rem; }
.ml8-flow-box .ml8-fl-s { color:#64748b; font-size:0.76rem; }
.ml8-flow-arrow {
    display:flex; align-items:center; justify-content:center; color:#c026d3;
}
.ml8-cmp-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.ml8-cmp-table th, .ml8-cmp-table td {
    border:1px solid #cbd5e1; padding:8px 10px; text-align:left; color:#000000;
    vertical-align:top;
}
.ml8-cmp-table th { color:#c026d3; background:#eef2fb; }
.ml8-cmp-table td.ml8-yes { color:#059669; font-weight:900; }
.ml8-cmp-table td.ml8-no  { color:#dc2626; font-weight:900; }
@media (max-width: 640px) {
    .ml8-cmp-table th, .ml8-cmp-table td { padding:5px 6px; font-size:0.78rem; }
    .ml8-flow-arrow { transform:rotate(90deg); }
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'''
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("cluster", size=32)} 機械学習の仕組み</h1>
    <p class="sub-title-text">MISSION 08 — 「教えて学ばせる」か「自分で気づかせる」か</p>
</div>
''', unsafe_allow_html=True)

# ================================================================
# SECTION 5: セッション状態の初期化
# ================================================================
for _k, _v in {
    # 教師あり（KNN）
    "knn_seed_ml8": 3,
    # 教師なし（k-means）
    "km_seed_ml8": 5,
    "km_k_ml8": 4,
    "km_centroids_ml8": None,
    "km_assignments_ml8": None,
    "km_iter_ml8": 0,
    "km_changed_ml8": None,
    "km_converged_ml8": False,
    "km_celebrated_ml8": False,
    "km_sig_ml8": None,
    # クイズ
    "quiz_idx_ml8": 0,
    "quiz_records_ml8": {},
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ================================================================
# SECTION 6: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#c026d3; font-weight:bold; font-size:0.9rem;">
        {svg_icon("cluster", size=15, color="#c026d3")} ML LEARNING MODE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"{icon_html('book-open', 'このミッションの狙い', size=15, color='#0284c7')}",
                unsafe_allow_html=True)
    st.caption(
        "機械学習の3つの学び方——教師あり・教師なし・強化学習——を、"
        "自分の手で動かして体感します。各タブのシミュレーションは、"
        "難しい数式ライブラリを使わず素朴なアルゴリズムを1から動かしています。"
    )

    st.divider()
    st.markdown(f"{icon_html('compass', 'Navigation', size=15, color='#059669')}",
                unsafe_allow_html=True)
    st.page_link("main_app.py",                      label="司令室 (Home)")
    st.page_link("pages/1_vision.py",                label="ミッション01: AIの目")
    st.page_link("pages/2_adversarial.py",           label="ミッション02: AI騙し")
    st.page_link("pages/3_training.py",              label="ミッション03: AI育成")
    st.page_link("pages/4_llm_mechanism.py",         label="ミッション04: LLMの脳内")
    st.page_link("pages/5_cpu_gpu.py",               label="ミッション05: CPU対GPU")
    st.page_link("pages/6_physical_digital_ai.py",   label="ミッション06: AIの二形態")
    st.page_link("pages/7_rl_game.py",               label="ミッション07: 育成ゲーム")
    st.page_link("pages/8_machine_learning.py",      label="ミッション08: 機械学習の仕組み")

    st.divider()
    st.success("SHIELD: ONLINE")

# ================================================================
# SECTION 7: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{svg_icon("cluster", color="#0284c7")} 「機械学習」には、大きく分けて3つの学び方がある</h3>
AI（機械学習）は、ひとつのやり方で学んでいるわけではありません。目的に応じて、
大きく<b>3つの学び方</b>を使い分けています。<br>
<ul>
  <li><b style="color:#0284c7;">教師あり学習</b>：正解ラベル付きの「お手本」から学ぶ。</li>
  <li><b style="color:#c026d3;">教師なし学習</b>：正解ラベルが一切ないデータから、隠れた構造を<b>自分で</b>見つける。</li>
  <li><b style="color:#059669;">強化学習</b>：報酬と罰による試行錯誤で、良い行動を学ぶ（ミッション07で体験済み）。</li>
</ul>
このページでは、まだどこでも扱っていない<b>教師あり</b>と<b>教師なし</b>の2つを、
シミュレーションで<b>並べて</b>体感します。動かせば「なるほど、こう違うのか」が一目でわかります。
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① 機械学習ってそもそも何？",
    "② 教師あり学習：お手本から学ぶ",
    "③ 教師なし学習：自分でグループを見つける",
    "④ 3つの学び方をくらべる",
])

# ---------------------------------------------------------------
# TAB 1: 機械学習ってそもそも何？
# ---------------------------------------------------------------
with tab1:
    st.markdown(heading("ルールを「書く」時代から、ルールを「学ばせる」時代へ", "lightbulb", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("contrast", color="#0284c7")} 従来のプログラミング vs 機械学習</h3>
        いちばん大事な違いを、<b>スパムメール（迷惑メール）の判定</b>を例に見てみましょう。<br><br>
        <b style="color:#c2410c;">■ 従来のプログラミング（人がルールを書く）</b><br>
        プログラマーが手作業で「もし『無料』という単語が入っていたらスパム」「もし『当選』が入っていたらスパム」…と、
        禁止ワードのリストを<b>ひとつずつ書き並べます</b>。
        でもこれは<b>もろい</b>やり方です。悪い人が「無料」を「無 料」とわざと崩して書けば、すぐ通り抜けてしまう。
        新しい手口が出るたびに、人間がルールを書き足し続けなければなりません。<br><br>
        <b style="color:#0284c7;">■ 機械学習（データからルールを学ばせる）</b><br>
        こちらは発想が逆です。人間はルールを書きません。かわりに
        <b>「これはスパム」「これは普通のメール」と正解を付けた大量のメール</b>をアルゴリズムに見せます。
        すると機械が<b>「スパムに共通するパターン」を自分で発見</b>し、
        ルールそのものをデータから作り出します。新しい手口も、例を追加して学び直せば自動で対応できます。<br><br>
        <b>まとめ：</b>従来のプログラミングは<b>人がルールを与える</b>。
        機械学習は<b>人がデータを与え、ルールは機械が見つける</b>。この一点が決定的な違いです。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("機械学習の基本の流れ", "arrow-right", level=3), unsafe_allow_html=True)

    arrow = f'<div class="ml8-flow-arrow">{svg_icon("arrow-right", size=22, color="#c026d3")}</div>'
    st.markdown(f"""
    <div class="ml8-flow">
      <div class="ml8-flow-box">
        <div class="ml8-fl-t">データ</div>
        <div class="ml8-fl-s">たくさんの例<br>（正解付きの場合も）</div>
      </div>
      {arrow}
      <div class="ml8-flow-box">
        <div class="ml8-fl-t">学習アルゴリズム</div>
        <div class="ml8-fl-s">パターンを<br>探し出す仕組み</div>
      </div>
      {arrow}
      <div class="ml8-flow-box">
        <div class="ml8-fl-t">モデル</div>
        <div class="ml8-fl-s">学びの結晶<br>＝発見したルール</div>
      </div>
      {arrow}
      <div class="ml8-flow-box">
        <div class="ml8-fl-t">新しい入力への予測</div>
        <div class="ml8-fl-s">はじめて見る<br>データに答える</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("3つの学び方をざっくり掴む", "layers", level=3), unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""
        <div class="guide-box">
          <h4>{svg_icon("tag", color="#0284c7")} 教師あり学習</h4>
          <p><b>正解ラベル付き</b>のお手本から学ぶ。<br>
          目的は<b>予測</b>。<br><br>
          例：正解の付いた写真集で「これは猫、これは犬」と学び、
          新しい写真の動物を当てる。</p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="guide-box">
          <h4>{svg_icon("cluster", color="#c026d3")} 教師なし学習</h4>
          <p><b>正解ラベルなし</b>。データの中の<br>隠れた構造を自分で見つける。<br>
          目的は<b>構造の発見</b>。<br><br>
          例：買い物履歴だけを見て、似た客同士を勝手にグループ分けする。</p>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="guide-box">
          <h4>{svg_icon("gamepad", color="#059669")} 強化学習</h4>
          <p><b>報酬と罰</b>による試行錯誤で、<br>良い行動を学ぶ。<br>
          目的は<b>行動の最適化</b>。<br><br>
          例：迷路やゲームを何度も遊び、勝ち方を自分で発見する。</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "強化学習は、ミッション07「育成ゲーム」で実際に体験ずみです。"
        "このページでは残りの2つ（教師あり・教師なし）を深掘りします。"
    )
    st.page_link("pages/7_rl_game.py", label="→ ミッション07『育成ゲーム』で強化学習を体験する")

# ---------------------------------------------------------------
# TAB 2: 教師あり学習（KNN）
# ---------------------------------------------------------------
with tab2:
    st.markdown(heading("教師あり学習：答えの書かれたお手本から学ぶ", "tag", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("book", color="#0284c7")} 「正解ラベル」がついているのがポイント</h3>
        教師あり学習では、練習に使うすべての例に<b>あらかじめ正解（ラベル）が付いています</b>。
        アルゴリズムの仕事は、<b>入力 → 正解</b>を結ぶルールを学び、
        <b>まだ見たことのない新しい入力</b>に対して正解を言い当てることです。<br><br>
        イメージは<b>単語カード（フラッシュカード）</b>。表に問題、裏に答えが書いてあります。
        何枚もめくって「表がこうなら裏はこう」を覚えれば、
        初めて見るカードでも裏（＝答え）を予想できるようになります。<br><br>
        下のデモでは、<b>甘さ</b>と<b>大きさ</b>という2つの特徴で3種類のフルーツ
        （リンゴ・ブルーベリー・レモン）を見分けます。色分けされた点が「正解ラベル付きのお手本」です。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("K近傍法（KNN）で新しいフルーツを分類する", "target", level=3),
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="point-box">
    <b>{svg_icon("search", color="#0284c7")} K近傍法（K-Nearest Neighbors）の考え方</b><br>
    とてもシンプルです。<b>「新しい点の近くにある“お手本”の多数決で決める」</b>だけ。<br>
    新しいフルーツを、<b>いちばん近い K 個のお手本</b>と見比べて、
    その中でいちばん多かった種類を「これだ！」と予測します。近所付き合いで正体を推理する感覚です。
    </div>
    """, unsafe_allow_html=True)

    # データ生成（再ロール可能）
    Xf, yf = make_fruit_data(st.session_state["knn_seed_ml8"])

    ctrl, view = st.columns([2, 3])
    with ctrl:
        st.markdown(heading("新しい点を置いて分類", "sliders", level=3), unsafe_allow_html=True)
        if st.button("データを作り直す", use_container_width=True, key="knn_reroll_ml8"):
            st.session_state["knn_seed_ml8"] = int(np.random.randint(0, 10_000))
            st.rerun()

        qx = st.slider("新しい点の 甘さ (x)", 0.0, 10.0, 5.0, 0.1, key="knn_qx_ml8")
        qy = st.slider("新しい点の 大きさ (y)", 0.0, 10.0, 5.0, 0.1, key="knn_qy_ml8")
        k_knn = st.select_slider("K（見比べる近所の数）", options=[1, 3, 5, 7],
                                 value=3, key="knn_k_ml8")

        pred, knn_idx, dists = knn_predict(Xf, yf, (qx, qy), k_knn)
        pred_name = FRUIT_NAMES[pred]
        pred_color = FRUIT_COLORS[pred]

        # 近傍の内訳
        neigh_labels = yf[knn_idx]
        counts = [int(np.sum(neigh_labels == j)) for j in range(len(FRUIT_NAMES))]

        st.markdown(f"""
        <div class="guide-box">
          <h4>{svg_icon("check-circle", color="#059669")} 予測結果</h4>
          <p>近い順に選んだ <b>{k_knn}個</b> の内訳：<br>
          <b style="color:#dc2626;">リンゴ {counts[0]}</b> ／
          <b style="color:#7c3aed;">ブルーベリー {counts[1]}</b> ／
          <b style="color:#b45309;">レモン {counts[2]}</b><br>
          多数決の結果 →
          <span style="color:{pred_color}; font-weight:900; font-size:1.1rem;">{pred_name}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.metric("KNNの予測", pred_name, f"K = {k_knn}")

    with view:
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        fig_knn = plot_knn(Xf, yf, query=(qx, qy), knn_idx=knn_idx, pred=pred)
        st.pyplot(fig_knn)
        plt.close(fig_knn)
        st.caption(
            "色つきの点＝正解ラベル付きのお手本。星＝あなたが置いた新しい点。"
            "点線でつながっているのが「近所の K 個」。この多数決で色（種類）が決まります。"
        )

    st.markdown(f"""
    <div class="explanation-box">
    <h3>{svg_icon("info", color="#c026d3")} なぜこれが「教師あり学習」なのか</h3>
    ポイントは、お手本の点が<b>最初から色（＝正解ラベル）付き</b>だったことです。
    「この甘さ・大きさならリンゴ」という<b>正解を人間が教えてある</b>——
    だからこそ、新しい点の種類を<b>言い当てる</b>ことができました。これが教師あり学習の本質です。<br><br>
    <b style="color:#b45309;">実は、ミッション03「AI育成」で動かしたニューロン（パーセプトロン）も教師あり学習です。</b>
    あちらは「重み」と「勾配降下法」という別の数学を使いますが、
    やっていることは同じ——<b>正解ラベル付きのデータから、入力→正解のルールを学ぶ</b>ことです。
    KNNは「近所の多数決」、ニューロンは「重みの微調整」。<b>道具は違っても、目的は同じ</b>なのです。<br><br>
    KはK近傍法の“性格”を決めます。<b>K=1</b>だと一番近い1個だけを信じるので敏感（外れ値に弱い）、
    <b>Kを大きく</b>すると多数決が安定する反面、境界がぼんやりします。スライダーで試してみましょう。
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 3: 教師なし学習（k-means）
# ---------------------------------------------------------------
with tab3:
    st.markdown(heading("教師なし学習：だれも正解を教えない世界", "cluster", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("help-circle", color="#c026d3")} ラベルが「まったく無い」データを相手にする</h3>
        教師なし学習では、データに<b>正解ラベルが一切ありません</b>。
        「これが正解」と教えてくれる先生がいないのです。
        そこでアルゴリズムは、<b>データの中の自然なまとまり（構造）を自分で探し出します</b>。<br><br>
        イメージは<b>ごちゃ混ぜのボタン箱を仕分けする</b>こと。
        だれも「これは“Aグループ”」なんて名前を教えてくれません。
        それでもあなたは「この辺は色も大きさも似てるから同じ仲間だな」と、
        <b>似ているもの同士を勝手にまとめられます</b>。教師なし学習はこれを機械にやらせます。<br><br>
        下の点群には<b>色がついていません</b>（＝ラベルなし）。
        この状態から、<b>k-means法</b>がグループを見つけていく様子を、
        <b>1ステップずつ</b>自分の目で追ってみましょう。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("k-means法：グループ分けが育っていく様子を観察", "sparkles", level=3),
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="point-box">
    <b>{svg_icon("refresh-cw", color="#c026d3")} k-means法は「割り当て」と「移動」の繰り返し</b><br>
    ① まず、グループの中心（セントロイド）を<b>K個</b>てきとうに置く。<br>
    ② 各点を、<b>いちばん近い中心</b>のグループに割り当てる（＝色分け）。<br>
    ③ 各中心を、<b>担当する点たちの真ん中（平均）</b>へ動かす。<br>
    ②→③をくり返すと、割り当てが変わらなくなる＝<b>収束</b>します。それがグループ分けの完成です。
    </div>
    """, unsafe_allow_html=True)

    # データ生成
    cloud = make_cloud_data(st.session_state["km_seed_ml8"])

    kc, kv = st.columns([2, 3])
    with kc:
        st.markdown(heading("コントロール", "sliders", level=3), unsafe_allow_html=True)

        k_km = st.slider("K（見つけたいグループの数）", 2, 6,
                         st.session_state["km_k_ml8"], 1, key="km_k_slider_ml8")

        # 状態のシグネチャ（データseed + K）が変わったら初期化し直す
        sig = (st.session_state["km_seed_ml8"], k_km)
        if st.session_state["km_sig_ml8"] != sig:
            st.session_state["km_k_ml8"] = k_km
            st.session_state["km_centroids_ml8"] = km_init_centroids(
                cloud, k_km, st.session_state["km_seed_ml8"] + 100)
            st.session_state["km_assignments_ml8"] = None
            st.session_state["km_iter_ml8"] = 0
            st.session_state["km_changed_ml8"] = None
            st.session_state["km_converged_ml8"] = False
            st.session_state["km_celebrated_ml8"] = False
            st.session_state["km_sig_ml8"] = sig

        b1, b2 = st.columns(2)
        with b1:
            if st.button("データを作り直す", use_container_width=True, key="km_reroll_ml8"):
                st.session_state["km_seed_ml8"] = int(np.random.randint(0, 10_000))
                st.session_state["km_sig_ml8"] = None  # 次の実行で再初期化
                st.rerun()
        with b2:
            if st.button("最初からやり直す", use_container_width=True, key="km_reset_ml8"):
                st.session_state["km_centroids_ml8"] = km_init_centroids(
                    cloud, k_km, int(np.random.randint(0, 10_000)))
                st.session_state["km_assignments_ml8"] = None
                st.session_state["km_iter_ml8"] = 0
                st.session_state["km_changed_ml8"] = None
                st.session_state["km_converged_ml8"] = False
                st.session_state["km_celebrated_ml8"] = False
                st.rerun()

        step_disabled = st.session_state["km_converged_ml8"]
        if st.button("次のステップへ ▶", use_container_width=True,
                     key="km_step_ml8", disabled=step_disabled):
            pts = cloud
            centroids = st.session_state["km_centroids_ml8"]
            old_assign = st.session_state["km_assignments_ml8"]
            # ② 割り当て
            new_assign = km_assign(pts, centroids)
            if old_assign is None:
                changed = len(pts)
            else:
                changed = int(np.sum(new_assign != old_assign))
            # ③ 中心を移動
            new_centroids = km_update(pts, new_assign, centroids)

            st.session_state["km_assignments_ml8"] = new_assign
            st.session_state["km_centroids_ml8"] = new_centroids
            st.session_state["km_iter_ml8"] += 1
            st.session_state["km_changed_ml8"] = changed
            if old_assign is not None and changed == 0:
                st.session_state["km_converged_ml8"] = True
            st.rerun()

        st.metric("イテレーション（くり返し回数）",
                  f"{st.session_state['km_iter_ml8']} 回")

        changed = st.session_state["km_changed_ml8"]
        if changed is None:
            st.info("「次のステップへ」を押して、グループ分けが始まる様子を見よう。")
        elif st.session_state["km_converged_ml8"]:
            st.success(
                f"収束しました！ 割り当てがもう変化しません"
                f"（前ステップからの変化：{changed} 点）。グループ分け完成です。"
            )
        else:
            st.warning(f"このステップで {changed} 個の点がグループを変えました。"
                       "まだ動いている＝学習の途中です。")

    with kv:
        st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
        fig_km = plot_kmeans(cloud, st.session_state["km_centroids_ml8"],
                             st.session_state["km_assignments_ml8"],
                             st.session_state["km_k_ml8"])
        st.pyplot(fig_km)
        plt.close(fig_km)
        st.caption(
            "大きな × ＝グループの中心（セントロイド）。点の色＝今どのグループに属しているか。"
            "ステップを進めるほど、中心が点の“真ん中”へ動き、色分けが安定していきます。"
        )

    # 収束時のお祝い（バルーンは一度だけ）
    if st.session_state["km_converged_ml8"] and not st.session_state["km_celebrated_ml8"]:
        st.balloons()
        st.session_state["km_celebrated_ml8"] = True

    st.markdown(f"""
    <div class="explanation-box">
    <h3>{svg_icon("contrast", color="#c026d3")} タブ②との決定的な違い</h3>
    さっきのKNN（教師あり）では、点に<b>最初から色（正解ラベル）が付いていました</b>。
    でも今回の点群には、<b>正解のグループなんてどこにも存在しません</b>。
    「このまとまりが正しい」と教えてくれる人はいない——
    それでもアルゴリズムは、<b>点の位置だけを手がかりに、自分でグループを発明</b>しました。
    これが教師なし学習です。<br><br>
    ちなみに<b>Kの数は人間が決めます</b>。同じ点群でも「2つに分けて」と言えば2グループ、
    「4つに分けて」と言えば4グループになります。<b>“正解”が無い</b>からこそ、
    どう分けるかは目的しだい。スライダーでKを変えて確かめてみましょう。<br><br>
    <b style="color:#b45309;">現実世界での使い道：</b>
    「似た買い物をする顧客をまとめる（顧客セグメンテーション）」「似た話題のニュース記事をまとめる」
    「いつもと違う変な動きを見つける（異常検知）」など。
    <b>正解ラベルを人間が用意しなくていい</b>のが、教師なし学習の大きな強みです。
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TAB 4: 3つの学び方をくらべる
# ---------------------------------------------------------------
with tab4:
    st.markdown(heading("教師あり・教師なし・強化学習をくらべる", "scale", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("layers", color="#0284c7")} 3つの学び方、ひと目で比較</h3>
        ここまで動かしてきた3つの学び方を、いちど表で整理しましょう。
        <b>「正解ラベルはあるか」</b>と<b>「そもそも何を目的にしているか」</b>で見分けるのがコツです。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <table class="ml8-cmp-table">
          <tr>
            <th>学び方</th><th>正解ラベル</th><th>目的</th><th>身近な例</th>
          </tr>
          <tr>
            <td><b style="color:#0284c7;">教師あり学習</b></td>
            <td class="ml8-yes">あり</td>
            <td>予測（入力→正解を当てる）</td>
            <td>写真の動物あて、家の価格予測、スパム判定</td>
          </tr>
          <tr>
            <td><b style="color:#c026d3;">教師なし学習</b></td>
            <td class="ml8-no">なし</td>
            <td>構造の発見（似たもの同士をまとめる）</td>
            <td>顧客のグループ分け、記事の話題まとめ、異常検知</td>
          </tr>
          <tr>
            <td><b style="color:#059669;">強化学習</b></td>
            <td class="ml8-no">なし（かわりに報酬信号）</td>
            <td>行動の最適化（どう動けば得か）</td>
            <td>ゲーム攻略、ロボット制御、囲碁AI（ミッション07）</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    st.page_link("pages/7_rl_game.py", label="→ 強化学習をもう一度体験する（ミッション07）")

    st.markdown(heading("ミニクイズ：これはどの学び方？", "help-circle", level=3),
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="point-box">
    <b>{svg_icon("target", color="#059669")} 遊び方</b><br>
    現実にありそうな場面を読んで、<b>教師あり／教師なし／強化学習</b>のどれに当てはまるか選びましょう。
    答え合わせをすると、なぜそうなるかの解説が出ます。全部で6問です。
    </div>
    """, unsafe_allow_html=True)

    QUIZ = [
        {
            "q": "過去の「家の広さ」と「価格」のデータ（価格つき）から、新しい家の価格を予測する。",
            "ans": "教師あり学習",
            "why": "各データに「価格」という正解が付いていて、それを当てる（予測する）のが目的。"
                   "典型的な教師あり学習（回帰）です。",
        },
        {
            "q": "購買履歴だけを見て、似た買い物をする顧客同士をグループ分けする（グループ名は誰も決めていない）。",
            "ans": "教師なし学習",
            "why": "正解のグループ名が最初から無い状態で、似たもの同士のまとまりを自分で見つけています。"
                   "まさに教師なし学習（クラスタリング）。",
        },
        {
            "q": "囲碁で勝つ手を、何百万局も自己対戦しながら学ぶ。",
            "ans": "強化学習",
            "why": "「勝ち＝報酬」を手がかりに、試行錯誤で良い手を学びます。正解の手は教えられていません。"
                   "これが強化学習です（ミッション07で体験した仕組み）。",
        },
        {
            "q": "写真に写っている動物の種類を、正解付きの写真集を使って学習する。",
            "ans": "教師あり学習",
            "why": "「これは猫」「これは犬」という正解ラベル付きのお手本から学ぶので教師あり学習（分類）。"
                   "タブ②のKNNと同じ枠組みです。",
        },
        {
            "q": "大量のニュース記事を、話題ごとに自動でまとめる（トピック名は後から人間が付ける）。",
            "ans": "教師なし学習",
            "why": "最初はラベルが無く、記事の中身の似かたでまとめています。名前は後付け。"
                   "タブ③のk-meansと同じ、教師なし学習です。",
        },
        {
            "q": "ロボットアームが、倒さずにコップを運べたら報酬、こぼしたら罰、をくり返して動き方を覚える。",
            "ans": "強化学習",
            "why": "正解の動き方は教えず、報酬と罰だけを頼りに試行錯誤で最適な動作を獲得します。"
                   "強化学習の典型例です。",
        },
    ]
    OPTIONS = ["教師あり学習", "教師なし学習", "強化学習"]
    N = len(QUIZ)
    idx = st.session_state["quiz_idx_ml8"]
    records = st.session_state["quiz_records_ml8"]

    st.progress(min(idx, N) / N, text=f"問 {min(idx + 1, N)} / {N}")

    if idx < N:
        cur = QUIZ[idx]
        st.markdown(f"""
        <div class="guide-box">
          <h4>{svg_icon("message-circle", color="#0284c7")} 第{idx + 1}問</h4>
          <p style="font-size:0.95rem; color:#000000;">{cur['q']}</p>
        </div>
        """, unsafe_allow_html=True)

        answered = idx in records
        choice = st.radio("どの学び方？", OPTIONS, key=f"quiz_radio_ml8_{idx}",
                          disabled=answered)

        if not answered:
            if st.button("答え合わせ", use_container_width=True, key=f"quiz_check_ml8_{idx}"):
                records[idx] = (choice == cur["ans"])
                st.session_state["quiz_records_ml8"] = records
                st.rerun()
        else:
            if records[idx]:
                st.success("正解！")
            else:
                st.error(f"おしい！ 正解は「{cur['ans']}」でした。")
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("lightbulb", color="#b45309")} 解説：答えは「{cur['ans']}」</h3>
            {cur['why']}
            </div>
            """, unsafe_allow_html=True)

            if idx < N - 1:
                if st.button("次の問題へ ▶", use_container_width=True,
                             key=f"quiz_next_ml8_{idx}"):
                    st.session_state["quiz_idx_ml8"] += 1
                    st.rerun()
            else:
                if st.button("結果を見る", use_container_width=True, key="quiz_finish_ml8"):
                    st.session_state["quiz_idx_ml8"] += 1
                    st.rerun()
    else:
        score = sum(1 for v in records.values() if v)
        st.markdown(heading("クイズ結果", "trophy", level=3), unsafe_allow_html=True)
        st.metric("スコア", f"{score} / {N} 問正解")
        if score == N:
            st.success("全問正解！ 3つの学び方をしっかり見分けられています。")
            st.balloons()
        elif score >= N - 2:
            st.info("いい線です！ 迷ったら「正解ラベルはあるか？」を思い出してみましょう。")
        else:
            st.warning("もう一度、各タブのシミュレーションを触ってから挑戦するとバッチリです。")
        if st.button("もう一度挑戦する", use_container_width=True, key="quiz_retry_ml8"):
            st.session_state["quiz_idx_ml8"] = 0
            st.session_state["quiz_records_ml8"] = {}
            st.rerun()

    st.markdown(f"""
    <div class="explanation-box">
    <h3>{svg_icon("glasses", color="#0284c7")} 最後に——ニュースの「AIが〇〇した」を読み解くために</h3>
    正直に言うと、現実のAIは<b>これらを組み合わせて</b>使うことがほとんどです。
    たとえば対話AI（チャットボット）は、まず大量の文章から自己教師あり的に下地を学び、
    そのあと人間のフィードバック（強化学習の一種）で仕上げる、といった具合に何段階も重ねます。<br><br>
    それでも、<b>「教師あり／教師なし／強化学習」という3つの土台</b>を知っているだけで、
    世の中の「AIが〇〇した」というニュースの<b>ほとんどは正しく評価できます</b>。
    「それは正解ラベル付きで学んだの？」「グループを勝手に見つけたの？」「試行錯誤で覚えたの？」——
    そう問いかけられるようになれば、あなたはもうAIを<b>むやみに恐れる側ではなく、理解する側</b>です。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# SECTION 8: フッター
# ================================================================
st.markdown('''
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 08: 機械学習の仕組み — 教師あり・教師なし、2つの学び方</p>
</div>
''', unsafe_allow_html=True)
