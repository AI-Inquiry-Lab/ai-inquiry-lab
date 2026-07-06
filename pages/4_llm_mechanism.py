import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import os
from collections import Counter, defaultdict
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
    page_title="ミッション04: LLMの脳内 | AI Inquiry Lab",
    page_icon=":material/chat:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# SECTION 1: セキュリティガード（他ページと同一の定型文）
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

# トークンを彩色するためのアクセントカラー（アイコンテーマと統一）
TOKEN_COLORS = ["#0284c7", "#c026d3", "#059669", "#b45309", "#7c3aed", "#c2410c", "#dc2626"]

# --- Tab3/Tab4 で使う小さな内蔵コーパス（日常テーマの短文・約20文） ---
# 実在のLLMは数兆語で学習するが、ここでは仕組みを体感するための極小データセット。
CORPUS = [
    "今日はいい天気ですね。",
    "今日は雨が降っています。",
    "今日は学校に行きます。",
    "猫が窓の外を見ています。",
    "猫はとても可愛いです。",
    "犬が公園を走っています。",
    "犬と猫が仲良く遊んでいます。",
    "私はりんごが好きです。",
    "私はパンを食べます。",
    "朝ごはんにパンを食べました。",
    "彼女は本を読んでいます。",
    "彼は音楽を聴いています。",
    "空がとても青いです。",
    "海で魚が泳いでいます。",
    "山に登るのは楽しいです。",
    "電車で学校に通っています。",
    "友達と公園で遊びました。",
    "夜になると星が見えます。",
    "花がきれいに咲いています。",
    "水を飲むと元気が出ます。",
]

# Tab4 のハルシネーション用に「コーパスに答えが存在しない」質問プリセット
HALLU_PROMPTS = {
    "田中太郎さんの誕生日は？": "田中太郎の誕生日",
    "月にある海の名前は？":     "月の海の名前",
    "架空の国ザラン王国の首都は？": "ザラン王国の首都",
}

# ================================================================
# SECTION 3: ユーティリティ（CSS）
# ================================================================
@st.cache_data
def _read_css_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def load_css(path: str) -> None:
    if os.path.exists(path):
        st.markdown(f"<style>{_read_css_text(path)}</style>", unsafe_allow_html=True)

# ================================================================
# SECTION 4: トイ・トークナイザ（本物のBPEを模した簡易版）
# ================================================================
def toy_tokenize(text: str):
    """空白・記号で区切り、長い語をさらに小片へ分割する簡易トークナイザ。

    - 半角の英数語が5文字以上 → 3文字ごとの小片（サブワード）へ
    - 日本語などの連続文字が3文字以上 → 2文字ごとの小片へ
    本物のBPE（Byte Pair Encoding）を大幅に単純化した教育用モデル。
    """
    if not text:
        return []
    # まず空白・句読点・記号で大まかに分割（区切り文字自体もトークンとして残す）
    rough = re.findall(r"[A-Za-z0-9]+|[ぁ-んァ-ヶ一-龠々ー]+|[^\sA-Za-z0-9ぁ-んァ-ヶ一-龠々ー]", text)
    tokens = []
    for chunk in rough:
        if re.fullmatch(r"[A-Za-z0-9]+", chunk):
            if len(chunk) > 4:
                for i in range(0, len(chunk), 3):
                    tokens.append(chunk[i:i + 3])
            else:
                tokens.append(chunk)
        elif re.fullmatch(r"[ぁ-んァ-ヶ一-龠々ー]+", chunk):
            if len(chunk) > 2:
                for i in range(0, len(chunk), 2):
                    tokens.append(chunk[i:i + 2])
            else:
                tokens.append(chunk)
        else:
            # 記号・句読点はそのまま1トークン（空白は無視）
            if chunk.strip():
                tokens.append(chunk)
    return tokens

# ================================================================
# SECTION 5: トイ注意機構（Self-Attention の縮小版）
# ================================================================
def token_embedding(token: str, dim: int = 8) -> np.ndarray:
    """トークン文字列を決定論的に dim 次元ベクトルへ写像（毎回同じ値）。"""
    seed = abs(hash(token)) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randn(dim)

@st.cache_data
def compute_attention(tokens, dim: int = 8):
    """softmax(Q K^T / sqrt(d)) を実際に計算して注意重み行列を返す。"""
    if len(tokens) == 0:
        return np.zeros((0, 0))
    E = np.stack([token_embedding(t, dim) for t in tokens])  # (n, dim)
    rng = np.random.RandomState(12345)          # Q/K 射影は固定シードで安定
    Wq = rng.randn(dim, dim) * 0.5
    Wk = rng.randn(dim, dim) * 0.5
    Q = E @ Wq
    K = E @ Wk
    scores = Q @ K.T / np.sqrt(dim)
    scores = scores - scores.max(axis=1, keepdims=True)      # 数値安定化
    expv = np.exp(scores)
    weights = expv / expv.sum(axis=1, keepdims=True)
    return weights

def plot_attention_heatmap(tokens, weights):
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(min(1.0 + n * 0.55, 8), min(1.0 + n * 0.55, 8)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    im = ax.imshow(weights, cmap="cool", vmin=0, vmax=weights.max() if weights.size else 1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [f"T{i+1}" for i in range(n)]
    ax.set_xticklabels(labels, color="#000000", fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, color="#000000", fontsize=8)
    ax.set_xlabel("Key (attended to)", color="#000000", fontsize=9)
    ax.set_ylabel("Query (looking from)", color="#000000", fontsize=9)
    ax.set_title("Attention Weights", color="#b45309", fontsize=11)
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#000000", labelsize=7)
    cbar.outline.set_edgecolor("#cbd5e1")
    plt.tight_layout(pad=0.4)
    return fig

def plot_attention_bars(tokens, weights, focus_idx):
    row = weights[focus_idx]
    order = np.argsort(row)[::-1]
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.45 * n)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ypos = np.arange(n)
    vals = row[order]
    labels = [f"T{i+1}" for i in order]
    colors = ["#b45309" if i == focus_idx else "#0284c7" for i in order]
    ax.barh(ypos, vals, color=colors, edgecolor="#eef2fb")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color="#000000", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attention weight", color="#000000", fontsize=9)
    ax.set_title(f"What T{focus_idx+1} attends to", color="#000000", fontsize=10)
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.tick_params(colors="#000000")
    ax.grid(True, axis="x", alpha=0.15, color="#cbd5e1")
    for y, v in zip(ypos, vals):
        ax.text(v + 0.01, y, f"{v:.2f}", color="#000000", va="center", fontsize=8)
    ax.set_xlim(0, min(1.0, vals.max() + 0.15))
    plt.tight_layout(pad=0.4)
    return fig

# ================================================================
# SECTION 6: トイ言語モデル（n-gram / マルコフ連鎖）
# ================================================================
def corpus_char_tokens(sentence: str):
    """文を文字単位トークン列へ（句点「。」も1トークンとして残す）。"""
    return list(sentence)

@st.cache_data(show_spinner=False)
def build_ngram_model():
    """内蔵コーパスから trigram / bigram / unigram の頻度表を構築する。"""
    unigram = Counter()
    bigram = defaultdict(Counter)        # 直前1文字 -> 次の文字
    trigram = defaultdict(Counter)       # 直前2文字 -> 次の文字
    for s in CORPUS:
        toks = corpus_char_tokens(s)
        for i, t in enumerate(toks):
            unigram[t] += 1
            if i >= 1:
                bigram[toks[i - 1]][t] += 1
            if i >= 2:
                trigram[(toks[i - 2], toks[i - 1])][t] += 1
    return unigram, dict(bigram), dict(trigram)

UNIGRAM, BIGRAM, TRIGRAM = build_ngram_model()

@st.cache_data
def predict_next(context_tokens):
    """直近1〜2文字の文脈から、次トークンの確率分布と使用した情報源を返す。

    戻り値: (dist, source)
      dist   : [(token, prob), ...] 確率降順で正規化済み
      source : "trigram" / "bigram" / "unigram(fallback)"
    """
    counter = None
    source = "unigram(fallback)"
    if len(context_tokens) >= 2:
        key = (context_tokens[-2], context_tokens[-1])
        if key in TRIGRAM and sum(TRIGRAM[key].values()) > 0:
            counter = TRIGRAM[key]
            source = "trigram"
    if counter is None and len(context_tokens) >= 1:
        key1 = context_tokens[-1]
        if key1 in BIGRAM and sum(BIGRAM[key1].values()) > 0:
            counter = BIGRAM[key1]
            source = "bigram"
    if counter is None:
        counter = UNIGRAM
        source = "unigram(fallback)"
    total = sum(counter.values())
    dist = [(tok, cnt / total) for tok, cnt in counter.items()]
    dist.sort(key=lambda x: x[1], reverse=True)
    return dist, source

def apply_temperature(dist, temperature):
    """p_i^(1/T) で分布を作り直して再正規化する。"""
    T = max(0.05, float(temperature))
    toks = [t for t, _ in dist]
    probs = np.array([p for _, p in dist], dtype=float)
    probs = np.clip(probs, 1e-12, None)
    reshaped = probs ** (1.0 / T)
    reshaped = reshaped / reshaped.sum()
    out = list(zip(toks, reshaped.tolist()))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def plot_topk_bars(dist, k=5, title="Next-token candidates"):
    top = dist[:k]
    labels = [f"C{i+1}" for i in range(len(top))]
    vals = [p for _, p in top]
    fig, ax = plt.subplots(figsize=(6, max(2.0, 0.55 * len(top))))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ypos = np.arange(len(top))
    palette = ["#b45309", "#0284c7", "#059669", "#c026d3", "#7c3aed"]
    ax.barh(ypos, vals, color=[palette[i % len(palette)] for i in range(len(top))],
            edgecolor="#eef2fb")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color="#000000", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Probability", color="#000000", fontsize=9)
    ax.set_title(title, color="#000000", fontsize=10)
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.tick_params(colors="#000000")
    ax.grid(True, axis="x", alpha=0.15, color="#cbd5e1")
    for y, v in zip(ypos, vals):
        ax.text(v + 0.005, y, f"{v:.1%}", color="#000000", va="center", fontsize=8)
    ax.set_xlim(0, min(1.0, (max(vals) if vals else 1) + 0.12))
    plt.tight_layout(pad=0.4)
    return fig

# ================================================================
# SECTION 7: CSSロード & ナビ & タイトル
# ================================================================
load_css(CSS_FILE)
render_top_nav("llm")

st.markdown(f'''
<div class="main-title-container">
    <h1 class="main-title-text">{svg_icon("message-circle", size=32)} LLMの脳内</h1>
    <p class="sub-title-text">MISSION 04 — ChatGPTは「考えて」いない。次の一語を当てているだけ</p>
</div>
''', unsafe_allow_html=True)

# ページ固有のスコープ付きCSS（style.cssは他ワークストリームが編集中のため触らない）
st.markdown("""
<style>
.llm-token-wrap { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.llm-token {
    display:inline-flex; align-items:center; padding:6px 12px; border-radius:8px;
    font-family:monospace; font-weight:700; font-size:1.0rem;
    background:#eef2fb; border:2px solid #0284c7; color:#000000;
}
.llm-token .llm-token-idx { font-size:0.65rem; color:#64748b; margin-right:6px; }
.llm-legend {
    display:flex; flex-wrap:wrap; gap:6px 16px; margin:8px 0;
    font-family:monospace; font-size:0.85rem; color:#000000;
}
.llm-legend span b { color:#b45309; }
.llm-gen-box {
    background:#eef2fb; border:2px solid #b45309; border-left:6px solid #0284c7;
    padding:16px; font-family:monospace; font-size:1.25rem; color:#059669;
    min-height:52px; letter-spacing:1px; word-break:break-all;
}
.llm-cand-chip {
    display:inline-flex; align-items:center; gap:8px; padding:6px 12px; margin:4px 0;
    border-radius:8px; background:#ffffff; border:1px solid #cbd5e1;
    font-family:monospace; font-size:0.95rem; color:#000000; width:100%;
}
.llm-cand-chip b { color:#b45309; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 8: セッション状態の初期化（すべて _llm サフィックス）
# ================================================================
if "llm_gen_tokens" not in st.session_state:
    st.session_state["llm_gen_tokens"] = []
if "llm_gen_count" not in st.session_state:
    st.session_state["llm_gen_count"] = 0
if "llm_sentence_done" not in st.session_state:
    st.session_state["llm_sentence_done"] = False

# ================================================================
# SECTION 9: サイドバー
# ================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="access-key-box">
        <span style="font-size:0.65rem; color:#64748b;">MISSION STATUS</span><br>
        <span style="color:#b45309; font-weight:bold; font-size:0.9rem;">{svg_icon("message-circle", size=15, color="#b45309")} LLM DECODE MODE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(heading("生成パラメータ", "sliders", level=3), unsafe_allow_html=True)
    temperature = st.slider(
        "温度 (Temperature)", 0.1, 2.0, 0.8, 0.1, key="llm_temp_sidebar",
        help="低い＝堅実で反復的 / 高い＝多様で予測不能"
    )
    gen_mode = st.radio(
        "生成モード",
        ["自動（確率に従ってAIが自動選択）", "手動選択"],
        key="llm_mode_sidebar",
    )

    st.markdown(f"""
    <div class="access-key-box" style="margin-top:10px;">
        <span style="font-size:0.65rem; color:#64748b;">GENERATED TOKENS</span><br>
        <span style="color:#059669; font-weight:bold; font-size:1.1rem;">{st.session_state['llm_gen_count']}</span>
        <span style="color:#64748b; font-size:0.7rem;"> 語 生成</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(heading("Navigation", "compass", level=3), unsafe_allow_html=True)
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
# SECTION 10: ヒーロー紹介
# ================================================================
st.markdown(f"""
<div class="explanation-box">
<h3>{svg_icon("message-circle", size=20)} ChatGPTは「魔法」でも「意識」でもない</h3>
大規模言語モデル（LLM＝Large Language Model）は、私たちの文章を<b>細切れの「トークン」</b>に分解し、
直前の文脈から<b>「次に来る可能性が一番高い一語」</b>を確率で選び続けているだけの仕組みです。<br><br>
賢そうに見える返答も、その正体は<b>膨大な確率計算のくり返し</b>。
このミッションでは、その仕組みを4つのタブで<b>ゼロから手を動かして</b>体感します。<br>
<hr style="border-color:#cbd5e1; margin:10px 0;">
<b style="color:#b45309;">{svg_icon("alert-triangle", size=16, color="#b45309")} 正直な注記：</b>
このページの「AI」は、本物のGPTではなく、numpyで手作りした<b>極小の教育用モデル</b>です。
本物のGPTは数千億〜兆のパラメータを持つ、はるかに巨大で複雑な同じ発想の拡張版。
ここでは<b>「考え方の形」だけ</b>を、うそなく体験できるように再現しています。
</div>
""", unsafe_allow_html=True)

st.markdown(heading("4つのタブでLLMの脳内をのぞこう", "search", level=2), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "① トークン化ラボ",
    "② 注意（Attention）の仕組み",
    "③ 次の単語を予言せよ！",
    "④ ハルシネーション体験",
])

# ================================================================
# TAB 1: トークン化ラボ
# ================================================================
with tab1:
    st.markdown(heading("文章を「トークン」に切り分ける", "puzzle", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("puzzle", size=20)} トークンとは何か？</h3>
        LLMは文章を「単語まるごと」ではなく、<b>「トークン」と呼ばれる細切れの部品</b>に分解してから読み込みます。
        トークンは必ずしも1単語とは限らず、単語の一部（サブワード）のこともあります。<br><br>
        <b style="color:#b45309;">具体例（本物のLLMの分け方のイメージ）：</b>
        <ul>
            <li>「がんばれ」 → <b>「がん」＋「ばれ」</b>（意味とは無関係に、よく出る文字のかたまりで区切る）</li>
            <li>「unbelievable」 → <b>「un」＋「believ」＋「able」</b></li>
            <li>「東京都」 → <b>「東京」＋「都」</b></li>
        </ul>
        なぜこんな面倒なことを？ 世界中のあらゆる単語を丸ごと覚えるのは無理でも、
        <b>短い部品の組み合わせ</b>なら、初めて見る言葉でも表現できるからです。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#059669;">{svg_icon("dollar-sign", size=16, color="#059669")} なぜ大事？</b>
        ChatGPTなどの<b>料金や入力上限は「文字数」ではなく「トークン数」で決まります</b>。
        同じ長さの文章でも、言語や単語によってトークン数は変わります。下で実際に数えてみましょう。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("トイ・トークナイザを試す", "flask", level=3), unsafe_allow_html=True)
    st.caption("本物のBPE（Byte Pair Encoding）というアルゴリズムを大幅に単純化した簡易版で分割します。")

    default_text = "今日はいい天気ですね。猫が公園を走っています。"
    user_text = st.text_area(
        "文章を入力（日本語でも英語でもOK）",
        value=default_text,
        key="llm_tok_input",
        height=90,
    )

    tokens = toy_tokenize(user_text)

    # トークンを彩色ボックスで表示
    if tokens:
        chips = ""
        for i, tok in enumerate(tokens):
            color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
            safe = (tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            chips += (
                f'<span class="llm-token" style="border-color:{color}; color:{color};">'
                f'<span class="llm-token-idx">#{i+1}</span>{safe}</span>'
            )
        st.markdown(f'<div class="llm-token-wrap">{chips}</div>', unsafe_allow_html=True)
    else:
        st.info("文章を入力するとトークンに分割されます。")

    char_count = len(user_text.replace("\n", "").replace(" ", ""))
    m1, m2, m3 = st.columns(3)
    m1.metric("トークン数", f"{len(tokens)}")
    m2.metric("文字数（空白除く）", f"{char_count}")
    ratio = (char_count / len(tokens)) if tokens else 0.0
    m3.metric("1トークンあたり文字数", f"{ratio:.2f}")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:14px;">
    <h3>{svg_icon("lightbulb", size=20)} ここから分かること</h3>
    <ul>
    <li>文字数とトークン数は<b>一致しません</b>。長い単語ほど複数トークンに割れます。</li>
    <li>LLMにとって文章は「意味のかたまり」ではなく、まず<b>この番号付きの部品列</b>にすぎません。</li>
    <li>この後のタブでは、この部品列に対してAIが<b>「次の部品」を確率で当てていく</b>様子を見ます。</li>
    </ul>
    <b style="color:#c2410c;">注意：</b> ここは教育用の簡易版なので、本物のGPTの分け方とは異なります。
    考え方（サブワードに割る）だけが本物と共通です。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 2: 注意（Attention）の仕組み
# ================================================================
with tab2:
    st.markdown(heading("各トークンが互いを「見る」仕組み", "eye", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("eye", size=20)} 自己注意（Self-Attention）とは？</h3>
        文章の意味は、単語を1つずつ見るだけでは決まりません。
        たとえば「猫がボールを追いかけた。<b>それ</b>はすばしっこい。」の「それ」は、
        直前の<b>「猫」</b>を指しています。<br><br>
        Transformer（GPTの心臓部）は、<b>それぞれのトークンが文中の他の全トークンを見渡し、
        「どれにどれだけ注目するか」</b>を計算して文脈を組み立てます。
        「それ」は「猫」に強く注目し、「ボール」には弱く注目する——これが注意（Attention）です。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">{svg_icon("wrench", size=16, color="#b45309")} 下でやっていること：</b>
        入力文をトークン化し、各トークンに固定の擬似ベクトル（8次元）を割り当て、
        <span style="font-family:monospace; color:#0284c7;">softmax(Q·Kᵀ / √d)</span>
        で注意重みを<b>実際に計算</b>して表示します。
        </div>
        """, unsafe_allow_html=True)

    att_text = st.text_input(
        "文章を入力（短めがおすすめ）",
        value="猫 が ボール を 追いかけた それ",
        key="llm_att_input",
    )
    att_tokens = toy_tokenize(att_text)
    # 表示・計算が重くなりすぎないよう上限
    att_tokens = att_tokens[:12]

    if len(att_tokens) < 2:
        st.info("2トークン以上になるように文章を入力してください。")
    else:
        weights = compute_attention(att_tokens)

        # トークン番号 → 実トークンの対応表（matplotlibは日本語を描けないためHTMLで凡例表示）
        legend = ""
        for i, tok in enumerate(att_tokens):
            color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
            safe = tok.replace("<", "&lt;").replace(">", "&gt;")
            legend += f'<span><b style="color:{color};">T{i+1}</b> = {safe}</span>'
        st.markdown(f'<div class="llm-legend">{legend}</div>', unsafe_allow_html=True)

        col_hm, col_bar = st.columns([1, 1])
        with col_hm:
            fig_hm = plot_attention_heatmap(att_tokens, weights)
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_hm)
            plt.close(fig_hm)
            st.caption("明るいマスほど「その行のトークンが、その列のトークンに強く注目」を意味します。")

        with col_bar:
            focus_label = st.selectbox(
                "注目元のトークンを選ぶ",
                options=list(range(len(att_tokens))),
                format_func=lambda i: f"T{i+1}（{att_tokens[i]}）",
                key="llm_att_focus",
            )
            fig_bar = plot_attention_bars(att_tokens, weights, focus_label)
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            row = weights[focus_label]
            best = int(np.argmax(row))
            st.markdown(f"""
            <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px; font-size:0.95rem;">
            「<b style="color:#b45309;">{att_tokens[focus_label]}</b>」は
            「<b style="color:#0284c7;">{att_tokens[best]}</b>」に一番注目しています
            （重み <b style="color:#059669;">{row[best]:.2f}</b>）。
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:14px;">
    <h3>{svg_icon("layers", size=20)} これは「本物の縮小版」です</h3>
    本物のTransformerは、この注意計算を<b>並列に何十本も（マルチヘッド）</b>、
    さらに<b>何十層も</b>重ねて実行します。
    ここで見せているのは<b>たった1本・1層</b>の単純化した注意です。<br><br>
    <b style="color:#059669;">正直な注記：</b>
    <b>計算の「形」（似ているものに重みをかけて足し合わせる）は本物と同じ</b>ですが、
    ここでの各トークンのベクトルは<b>ハッシュから作った仮の値</b>。
    本物では、これが<b>数兆語のデータから学習</b>され、意味を反映した値になります。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TAB 3: 次の単語を予言せよ！
# ================================================================
with tab3:
    st.markdown(heading("次の一語を確率で予測してみよう", "target", level=2), unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("target", size=20)} LLMの本体は「次の一語当てゲーム」</h3>
        ここでは、内蔵した<b>約20文の小さな日本語コーパス</b>からその場で
        <b>n-gram（マルコフ連鎖）モデル</b>を作ります。
        「直前の1〜2文字」から「次に来る文字」の確率分布を数え上げただけの、素朴なAIです。<br><br>
        あなたが「次の一語を予測」を押すたびに、モデルは<b>候補トップ5とその確率</b>を提示します。
        これを1文字ずつ繰り返して文章を伸ばす——
        これが本物のLLMの<b>自己回帰（autoregressive）生成</b>とまったく同じ流れです。
        <hr style="border-color:#cbd5e1; margin:10px 0;">
        <b style="color:#b45309;">{svg_icon("help-circle", size=16, color="#b45309")} 「見たことがない文脈」だと？</b>
        コーパスに存在しない並びが来ると、AIは細かい文脈をあきらめて
        <b>単純に「よく出てくる言葉」（全体頻度）に頼ります</b>。これを「フォールバック」と呼びます。
        </div>
        """, unsafe_allow_html=True)

    # --- 開始フレーズの選択 ---
    seed_col1, seed_col2 = st.columns([2, 1])
    with seed_col1:
        preset = st.selectbox(
            "開始フレーズ（プリセット）",
            ["今日は", "猫が", "私は", "（自分で入力する）"],
            key="llm_seed_preset",
        )
    with seed_col2:
        custom_seed = st.text_input("自由入力の開始フレーズ", value="", key="llm_seed_custom")

    start_phrase = custom_seed if preset == "（自分で入力する）" and custom_seed else (
        "" if preset == "（自分で入力する）" else preset
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        if st.button("この開始フレーズで初期化",
                     use_container_width=True, key="llm_init_btn"):
            st.session_state["llm_gen_tokens"] = list(start_phrase)
            st.session_state["llm_sentence_done"] = False
    with ctrl2:
        do_predict = st.button("次の一語を予測",
                               use_container_width=True, key="llm_predict_btn")
    with ctrl3:
        if st.button("リセット",
                     use_container_width=True, key="llm_reset_btn"):
            st.session_state["llm_gen_tokens"] = []
            st.session_state["llm_sentence_done"] = False

    # 現在の文脈から分布を計算
    current = st.session_state["llm_gen_tokens"]
    dist_raw, source = predict_next(current)
    dist = apply_temperature(dist_raw, temperature)
    top5 = dist[:5]

    # 予測ボタン処理（自動モードのときは温度に従ってサンプリング）
    if do_predict and not st.session_state["llm_sentence_done"]:
        if gen_mode.startswith("自動"):
            toks = [t for t, _ in dist]
            probs = np.array([p for _, p in dist], dtype=float)
            probs = probs / probs.sum()
            rng = np.random.RandomState()   # 自動モードは毎回変化させる
            choice = rng.choice(len(toks), p=probs)
            chosen = toks[choice]
            st.session_state["llm_gen_tokens"].append(chosen)
            st.session_state["llm_gen_count"] += 1
            if chosen == "。" or len(st.session_state["llm_gen_tokens"]) >= 25:
                st.session_state["llm_sentence_done"] = True
            st.rerun()

    # --- 生成中の文を表示 ---
    st.markdown(heading("生成中の文", "message-circle", level=3), unsafe_allow_html=True)
    gen_text = "".join(st.session_state["llm_gen_tokens"]) or "（まだ何も生成されていません）"
    safe_gen = gen_text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="llm-gen-box">{safe_gen}<span style="color:#0284c7;">▎</span></div>',
                unsafe_allow_html=True)

    info_c1, info_c2 = st.columns([1, 1])
    with info_c1:
        src_label = {"trigram": "直前2文字を使用（trigram）",
                     "bigram": "直前1文字を使用（bigram）",
                     "unigram(fallback)": "全体頻度に頼る（フォールバック）"}[source]
        st.markdown(f"""
        <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:12px; font-size:0.9rem;">
        <span style="color:#64748b;">現在の予測の根拠：</span><br>
        <b style="color:#b45309;">{src_label}</b>
        </div>
        """, unsafe_allow_html=True)
        if source == "unigram(fallback)":
            st.caption("見たことがない文脈なので、AIは単純に『よく出てくる言葉』に頼っています。")

    with info_c2:
        st.metric("累計 生成トークン数", f"{st.session_state['llm_gen_count']} 語")

    # --- 候補トップ5の表示（チャート＋凡例＋手動選択） ---
    st.markdown(heading("次トークンの候補トップ5", "bar-chart", level=3), unsafe_allow_html=True)
    if top5:
        chart_col, pick_col = st.columns([1, 1])
        with chart_col:
            fig_top = plot_topk_bars(dist, k=5, title="Next-token probabilities")
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_top)
            plt.close(fig_top)
            # C1..C5 → 実トークン対応
            legend = ""
            palette = ["#b45309", "#0284c7", "#059669", "#c026d3", "#7c3aed"]
            for i, (tok, p) in enumerate(top5):
                disp = tok if tok != "。" else "。(文末)"
                legend += (f'<span><b style="color:{palette[i % len(palette)]};">C{i+1}</b> = '
                           f'{disp} ({p:.1%})</span>')
            st.markdown(f'<div class="llm-legend">{legend}</div>', unsafe_allow_html=True)

        with pick_col:
            if gen_mode.startswith("手動"):
                st.caption("AIの提示した候補から自分で1つ選んで文をのばそう。")
                options = [f"{t}  —  {p:.1%}" for t, p in top5]
                pick = st.radio("追加するトークンを選ぶ", options, key="llm_manual_pick")
                if st.button("選んだ一語を追加",
                             use_container_width=True, key="llm_append_btn",
                             disabled=st.session_state["llm_sentence_done"]):
                    idx = options.index(pick)
                    chosen = top5[idx][0]
                    st.session_state["llm_gen_tokens"].append(chosen)
                    st.session_state["llm_gen_count"] += 1
                    if chosen == "。" or len(st.session_state["llm_gen_tokens"]) >= 25:
                        st.session_state["llm_sentence_done"] = True
                    st.rerun()
            else:
                st.markdown(f"""
                <div style="background:#eef2fb; border:1px solid #cbd5e1; padding:14px; font-size:0.9rem;">
                <span style="color:#b45309;">自動モード：</span><br>
                「次の一語を予測」を押すたびに、AIが<b>温度 {temperature:.1f}</b> に従って
                確率的に1語を選びます。<br><br>
                <span style="color:#64748b;">最有力候補：</span>
                <b style="color:#059669; font-size:1.1rem;">{top5[0][0]}</b>
                （{top5[0][1]:.1%}）
                </div>
                """, unsafe_allow_html=True)

    if st.session_state["llm_sentence_done"]:
        st.success("文が完成しました（文末「。」に到達）。1文を生成し切りました。")
    elif st.session_state["llm_gen_count"] >= 8:
        st.success(f"{st.session_state['llm_gen_count']}語も生成しました。自己回帰生成のリズムをつかめましたね。")

    # --- 温度の説明 ---
    st.markdown(heading("温度（Temperature）で「性格」が変わる", "thermometer", level=3),
                unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        temp_c1, temp_c2 = st.columns([1, 1])
        with temp_c1:
            st.markdown(f"""
            <div class="explanation-box">
            <h3>{svg_icon("thermometer", size=20)} 温度スライダーの正体</h3>
            サイドバーの<b>温度</b>は、確率分布を
            <span style="font-family:monospace; color:#0284c7;">pᵢ^(1/T)</span>
            で作り直してから選ぶ仕組みです。<br>
            <ul>
            <li><b style="color:#0284c7;">低温 (T→0.1)</b>：一番確率の高い語ばかり選ぶ。
                <b>堅実だが反復的・退屈</b>。同じ言葉が勝ち続けます。</li>
            <li><b style="color:#c026d3;">高温 (T→2.0)</b>：低確率の語も選ばれやすい。
                <b>創造的だが支離滅裂</b>になりがち。</li>
            </ul>
            これは本物のLLM API（ChatGPT等）が公開している
            <b>temperature パラメータそのもの</b>です。
            </div>
            """, unsafe_allow_html=True)
        with temp_c2:
            # 温度による分布の変化を可視化
            low = apply_temperature(dist_raw, 0.2)[:5]
            high = apply_temperature(dist_raw, 1.8)[:5]
            fig_t, (axl, axr) = plt.subplots(1, 2, figsize=(6.5, 3.0))
            fig_t.patch.set_facecolor("#ffffff")
            for ax, data, ttl, col in [(axl, low, "T=0.2 (堅実)", "#0284c7"),
                                       (axr, high, "T=1.8 (奔放)", "#c026d3")]:
                ax.set_facecolor("#ffffff")
                yy = np.arange(len(data))
                ax.barh(yy, [p for _, p in data], color=col, edgecolor="#eef2fb")
                ax.set_yticks(yy)
                ax.set_yticklabels([f"C{i+1}" for i in range(len(data))],
                                   color="#000000", fontsize=8)
                ax.invert_yaxis()
                ax.set_title(ttl.split(" ")[0], color=col, fontsize=10)
                ax.tick_params(colors="#000000")
                for sp in ax.spines.values():
                    sp.set_color("#cbd5e1")
                ax.set_xlim(0, 1)
            plt.tight_layout(pad=0.5)
            st.markdown('<span class="sim-chart-marker"></span>', unsafe_allow_html=True)
            st.pyplot(fig_t)
            plt.close(fig_t)
            st.caption("同じ文脈でも、低温は一択に尖り、高温はなだらかに散らばります。")

# ================================================================
# TAB 4: ハルシネーション体験
# ================================================================
with tab4:
    st.markdown(heading("なぜAIは堂々とウソをつくのか", "alert-triangle", level=2),
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="lab-anchor-green"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation-box">
        <h3>{svg_icon("alert-triangle", size=20)} ハルシネーション（幻覚）とは？</h3>
        LLMは常に<b>「次にもっともらしく続く語」</b>を選んでいるだけで、
        <b>「事実かどうか」を確かめる仕組みは持っていません</b>。<br><br>
        だから学習データに答えが無いことを聞かれても、
        AIは「分かりません」と言う代わりに、<b>流暢でそれらしい文章を自信満々に作り出します</b>。
        これがハルシネーション（幻覚）です。<br><br>
        これは<b>バグやサボりではなく、次の一語を予測するという仕組みそのものの当然の帰結</b>。
        だからこそ、AIの出力を<b>ファクトチェックすること</b>が欠かせません。
        </div>
        """, unsafe_allow_html=True)

    st.markdown(heading("答えを知らない質問をぶつけてみる", "bug", level=3), unsafe_allow_html=True)
    st.caption("内蔵コーパス（約20文の日常会話）には、下の質問の答えは一切含まれていません。")

    q_label = st.selectbox(
        "AIに聞く質問（コーパスに答えが無い）",
        list(HALLU_PROMPTS.keys()),
        key="llm_hallu_q",
    )

    if st.button("AIに答えさせる", key="llm_hallu_btn"):
        seed = HALLU_PROMPTS[q_label]
        # シードの末尾から自己回帰的に貪欲生成（トップ確率を採用）
        gen = list(seed)
        confidences = []
        rng = np.random.RandomState(7)
        for _ in range(14):
            d_raw, _src = predict_next(gen)
            d = apply_temperature(d_raw, 0.7)
            if not d:
                break
            # 上位数語から確率的に選ぶ（それらしさを演出）
            k = min(3, len(d))
            toks = [t for t, _ in d[:k]]
            ps = np.array([p for _, p in d[:k]], dtype=float)
            ps = ps / ps.sum()
            ci = rng.choice(k, p=ps)
            chosen = toks[ci]
            confidences.append(d[0][1])   # トップ候補の確率＝見かけの自信度
            gen.append(chosen)
            if chosen == "。":
                break
        answer = "".join(gen[len(list(seed)):]) or "…"
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        st.session_state["llm_hallu_result"] = {
            "q": q_label, "answer": answer, "conf": avg_conf,
        }

    res = st.session_state.get("llm_hallu_result")
    if res:
        st.markdown(f"""
        <div style="background:#eef2fb; border:2px solid #dc2626; border-left:6px solid #b45309;
                    padding:16px; margin-top:6px;">
            <div style="color:#64748b; font-size:0.85rem;">質問</div>
            <div style="color:#000000; font-size:1.05rem; margin-bottom:10px;">{res['q']}</div>
            <div style="color:#64748b; font-size:0.85rem;">AIの回答（自信満々）</div>
            <div style="color:#059669; font-family:monospace; font-size:1.15rem;">
                「{res['answer']}」
            </div>
        </div>
        """, unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        hc1.metric("AIの見かけの自信度", f"{res['conf']:.1%}")
        hc2.metric("コーパス内の実際の根拠", "0 件")

        st.error("この回答は完全なでっち上げです。コーパスには答えが1文字も無いのに、"
                 "AIは高い『自信度』の数字とともに、流暢な文を生成しました。")

        st.markdown(f"""
        <div class="explanation-box" style="margin-top:14px;">
        <h3>{svg_icon("alert-triangle", size=20)} 「自信度が高い＝正しい」ではない</h3>
        いま見たとおり、モデルは<b>答えを持っていないのに、それらしい文と高い確率値</b>を出しました。
        LLMの「自信度」は<b>「言葉のつながりの自然さ」の指標にすぎず、「事実の正しさ」ではありません</b>。<br>
        堂々とした口調も、詳しそうな説明も、正しさの保証には一切なりません。
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("「AIに答えさせる」を押すと、答えを知らないはずのAIが堂々と回答します。")

    st.markdown(f"""
    <div class="explanation-box" style="margin-top:14px;">
    <h3>{svg_icon("shield", size=20)} この教訓をどう活かすか（大人向けまとめ）</h3>
    LLMは<b>下書き・要約・発想・言い換え</b>など、事実の一次ソースが不要な作業では非常に強力です。
    恐れて避ける必要はありません。ただし、次のような場面では<b>必ず裏取り</b>を：
    <ul>
    <li><b style="color:#b45309;">日付・数字・固有名詞</b>（人名・地名・製品名）</li>
    <li><b style="color:#b45309;">引用・出典・URL・法律や医療の判断</b></li>
    <li><b style="color:#b45309;">「知らないはず」の専門的・最新の事実</b></li>
    </ul>
    合言葉は<b>「便利な下書き係として使い、事実は自分で確認する」</b>。
    仕組みを理解すれば、AIは恐れる対象でも盲信する対象でもなく、<b>賢く使いこなす道具</b>になります。
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# SECTION 11: フッター
# ================================================================
st.markdown('''
<div class="custom-footer">
    <p>© 2026 <strong>AI Inquiry Lab.</strong> | AIを恐れない。理解する。</p>
    <p>MISSION 04: LLMの脳内 — 確率とトークンで文章を紡ぐ仕組み</p>
</div>
''', unsafe_allow_html=True)
