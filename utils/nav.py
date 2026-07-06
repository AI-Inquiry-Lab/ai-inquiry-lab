"""Shared top navigation bar for all pages.

Renders a hamburger-style menu:
- Collapsed (default): brand mark + a small toggle button showing the
  classic three-horizontal-bar "menu" glyph in the top-right corner.
- Expanded: the toggle switches to a "close" mark, and a vertical list of
  every page (via st.page_link(), which always navigates in the same tab
  and auto-highlights the current page) appears below it.

Note on icons: Streamlit's built-in `:material/xxx:` icon shortcode looks
like a clean way to add icons to native widgets (st.button/st.page_link
can't render raw SVG), but in this app it renders as literal overlapping
text instead of a glyph. To avoid that bug entirely, native widgets here
use plain text only. The hamburger/close toggle uses the "☰"/"✕"
Unicode symbols (plain monochrome glyphs from the default UI font, not
color emoji) styled via CSS scoped to this button through Streamlit's
`st.container(key=...)` -> `.st-key-<key>` class hook.
"""
import streamlit as st

from utils.icons import svg_icon

PAGES = [
    ("main_app.py",                    "ホーム"),
    ("pages/1_vision.py",              "M01: AIの目"),
    ("pages/2_adversarial.py",         "M02: AI騙し"),
    ("pages/3_training.py",            "M03: AI育成"),
    ("pages/4_llm_mechanism.py",       "M04: LLMの脳内"),
    ("pages/5_cpu_gpu.py",             "M05: CPU対GPU"),
    ("pages/6_physical_digital_ai.py", "M06: AIの二形態"),
    ("pages/7_rl_game.py",             "M07: 育成ゲーム"),
    ("pages/8_machine_learning.py",    "M08: 機械学習"),
]

MENU_GLYPH  = "☰"  # ☰ 三本線
CLOSE_GLYPH = "✕"  # ✕


def render_top_nav(active_key: str = "home") -> None:
    """Render the hamburger nav bar (brand + toggle, and the page list when open)."""
    if "nav_menu_open" not in st.session_state:
        st.session_state.nav_menu_open = False

    # ページ遷移（page_linkクリック）が起きたら自動的に閉じる。
    # session_stateはページ間で共有されるため、直前に開いていたページと
    # active_keyが変わっていれば「リンクをクリックして移動してきた」と判断する。
    if st.session_state.get("nav_last_active_key") != active_key:
        st.session_state.nav_menu_open = False
        st.session_state.nav_last_active_key = active_key

    is_open = st.session_state.nav_menu_open

    brand_col, toggle_col = st.columns([6, 1])

    with brand_col:
        st.markdown(
            f'<p class="nav-brand-inline">{svg_icon("brain", size=15)} AI LAB</p>',
            unsafe_allow_html=True,
        )

    with toggle_col:
        with st.container(key="nav_hamburger_container"):
            glyph = CLOSE_GLYPH if is_open else MENU_GLYPH
            if st.button(glyph, key="nav_hamburger_btn", use_container_width=True):
                # st.button のクリックは既にこのスクリプト全体の再実行そのものなので、
                # ここで st.rerun() を呼ぶと再実行が二重に走ってしまう。
                # is_open をその場で更新し、二重再実行なしで以降の描画に反映させる。
                st.session_state.nav_menu_open = not is_open
                is_open = not is_open

    st.markdown('<div class="nav-bottom-bar"></div>', unsafe_allow_html=True)

    if is_open:
        # position:fixed の実コンテナ（st.container(key=...) が生成する
        # .st-key-<key> クラス）を使い、ページ本文の flow から完全に切り離して
        # 最前面にオーバーレイ表示する。
        # 背景（バックドロップ）はクリックすると閉じる — 実体は画面全面を覆う
        # 透明な st.button（.invisible-button）で、パネルより低いz-indexに
        # 置くことでパネル自体のクリックには反応しない。
        with st.container(key="nav_backdrop"):
            st.markdown('<div class="invisible-button">', unsafe_allow_html=True)
            if st.button("背景", key="nav_backdrop_btn", use_container_width=True):
                st.session_state.nav_menu_open = False
                is_open = False
            st.markdown('</div>', unsafe_allow_html=True)

        if is_open:
            with st.container(key="nav_dropdown_panel"):
                st.markdown(
                    f'<p class="nav-dropdown-title">{svg_icon("compass", size=15)} ページ一覧</p>',
                    unsafe_allow_html=True,
                )
                for path, label in PAGES:
                    st.page_link(path, label=label, use_container_width=True)
