"""Inline SVG icon library — replaces all emoji across AI Inquiry Lab.

Every icon is a minimal stroke-based line drawing on a 24x24 grid,
rendered via ``currentColor`` so it inherits the surrounding text color
(or an explicit color class). This keeps the pixel/cyber theme
consistent without depending on any external font or CDN (required for
Streamlit + offline use).

Usage:
    from utils.icons import icon, icon_html

    st.markdown(f"<h2>{icon('brain', color='var(--color-cyan)')} タイトル</h2>",
                unsafe_allow_html=True)

    # Inside f-string HTML blocks:
    f'<div>{icon("shield")} SHIELD: ONLINE</div>'
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Raw icon bodies (inner SVG markup only — the wrapper adds <svg>).
# All on a 0 0 24 24 viewBox, stroke=currentColor, fill=none unless noted.
# ----------------------------------------------------------------------
_ICONS: dict[str, str] = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9.5h13V10"/><path d="M9.5 19.5V13h5v6.5"/>',
    "eye": '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "mask": '<path d="M4 7c2 -2 5 -2.5 8 -2.5s6 0.5 8 2.5c0 6 -2 12 -8 14 -6 -2 -8 -8 -8 -14Z"/><path d="M9 11v1.2M15 11v1.2"/><path d="M9 15.5c1 1 5 1 6 0"/>',
    "dna": '<path d="M7 3c0 4 10 4 10 8s-10 4-10 8"/><path d="M17 3c0 4-10 4-10 8s10 4 10 8"/><path d="M8 6h8M7.3 9.5h9.4M7.3 14.5h9.4M8 18h8"/>',
    "brain": '<path d="M9 4.5a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3.2 3.2 0 0 0 1.6 5.4A3 3 0 0 0 9 20.5"/><path d="M15 4.5a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3.2 3.2 0 0 1-1.6 5.4A3 3 0 0 1 15 20.5"/><path d="M9 4.5v16M15 4.5v16M9 9h3M9 13.5h3M12 9v9M15 9h-3M15 13.5h-3"/>',
    "flask": '<path d="M9.5 3h5M10 3v6.5L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9.5V3"/><path d="M7.5 14.5h9"/>',
    "shield": '<path d="M12 3.5 5 6v6c0 5 3 8 7 8.5 4-.5 7-3.5 7-8.5V6l-7-2.5Z"/><path d="m9 12 2 2 4-4.5"/>',
    "rocket": '<path d="M13.5 4.5c3 0 5.5 2.5 5.5 5.5-2 5-5 8-8 9-1-2-1-4-1-6 1-3 3-6 8.5-8.5Z"/><path d="M9 15c-2 0-3.5 1-4.5 4 3-1 4-2.5 4-4.5Z"/><circle cx="14.5" cy="9.5" r="1.4"/>',
    "lightbulb": '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.4 1.1 2.2h5c0-.8.4-1.6 1.1-2.2A6 6 0 0 0 12 3Z"/>',
    "zap": '<path d="M12.5 3 5 13.5h5.5L10 21l7.5-10.5H12L12.5 3Z"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.5-4.5"/>',
    "alert-triangle": '<path d="M12 4 2.5 20h19L12 4Z"/><path d="M12 10.5v4M12 17.2v.1"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
    "trending-up": '<path d="M3 17 9.5 10.5 13.5 14.5 21 6"/><path d="M15 6h6v6"/>',
    "trending-down": '<path d="M3 7 9.5 13.5 13.5 9.5 21 18"/><path d="M15 18h6v-6"/>',
    "bar-chart": '<path d="M4 20V10M12 20V4M20 20v-7"/>',
    "wrench": '<path d="M14.5 6.5a4 4 0 0 0-5.4 4.9L4 16.5V20h3.5l5.1-5.1a4 4 0 0 0 4.9-5.4l-2.7 2.7-2-2 2.7-2.7Z"/>',
    "microscope": '<path d="M6 20.5h11M9 20.5v-3.2a4.3 4.3 0 1 1 6 0v3.2"/><path d="M9.2 12 7 6.5l2.4-1 3 7.4"/><path d="M12.5 4.5l3 1.2"/>',
    "wind": '<path d="M3 8h11a2.5 2.5 0 1 0-2.5-2.5"/><path d="M3 12.5h15a2.5 2.5 0 1 1-2.5 2.5"/><path d="M3 17h8a2 2 0 1 1-2 2"/>',
    "sparkles": '<path d="M11 3 12.3 8 17 9.3 12.3 10.6 11 15.6 9.7 10.6 5 9.3 9.7 8 11 3Z"/><path d="M18 14l.7 2.3 2.3.7-2.3.7L18 20l-.7-2.3-2.3-.7 2.3-.7L18 14Z"/>',
    "graduation-cap": '<path d="M2.5 9 12 5l9.5 4-9.5 4-9.5-4Z"/><path d="M7 11.5v4.5c0 1.3 2.2 2.5 5 2.5s5-1.2 5-2.5v-4.5"/><path d="M21.5 9v5.5"/>',
    "trophy": '<path d="M8 4.5h8v5a4 4 0 0 1-8 0v-5Z"/><path d="M8 5.5H5a1 1 0 0 0-1 1c0 2.3 1.6 4 3.6 4.3M16 5.5h3a1 1 0 0 1 1 1c0 2.3-1.6 4-3.6 4.3"/><path d="M12 13.5v3M9 20.5h6M9.5 20.5c0-1.6.7-2.7 2.5-3 1.8.3 2.5 1.4 2.5 3"/>',
    "network": '<circle cx="6" cy="6" r="2.3"/><circle cx="18" cy="6" r="2.3"/><circle cx="6" cy="18" r="2.3"/><circle cx="18" cy="18" r="2.3"/><circle cx="12" cy="12" r="2.3"/><path d="M8 7.5 10.2 10.4M15.9 10.4 18 7.5M8 16.5l2.2-2.9M15.9 13.6 18 16.5"/>',
    "bug": '<path d="M9 8.5h6M8 6l1.5 2M16 6l-1.5 2"/><path d="M8 12a4 4 0 0 1 8 0v3a4 4 0 0 1-8 0v-3Z"/><path d="M4 11h3.2M20 11h-3.2M4 16h3.5M20 16h-3.5M9 20l1-2.3M15 20l-1-2.3"/>',
    "map": '<path d="M9 4.5 4 6.5v13l5-2 6 2 5-2v-13l-5 2-6-2Z"/><path d="M9 4.5v13M15 6.5v13"/>',
    "dumbbell": '<path d="M4 10v4M2.5 9v6M20 10v4M21.5 9v6"/><path d="M7 12h10"/><rect x="4.5" y="8.5" width="3" height="7" rx="0.3"/><rect x="16.5" y="8.5" width="3" height="7" rx="0.3"/>',
    "lock": '<rect x="4.5" y="10.5" width="15" height="10" rx="0.5"/><path d="M7.5 10.5V7a4.5 4.5 0 0 1 9 0v3.5"/><circle cx="12" cy="15" r="1.5"/><path d="M12 16.5v2"/>',
    "megaphone": '<path d="M3 10v4h3l9 5V5L6 10H3Z"/><path d="M18 9.5a3.5 3.5 0 0 1 0 5"/>',
    "dollar-sign": '<path d="M12 2.5v19"/><path d="M16.5 6.5c0-1.7-2-3-4.5-3S7.5 4.8 7.5 6.5c0 3.2 9 2 9 5.5 0 1.7-2 3-4.5 3s-4.5-1.3-4.5-3"/>',
    "printer": '<path d="M6.5 9V4.5h11V9"/><rect x="4.5" y="9" width="15" height="7" rx="0.5"/><path d="M6.5 14.5h11V20h-11v-5.5Z"/>',
    "stethoscope": '<path d="M6 3v6a3.5 3.5 0 0 0 7 0V3M9.5 12.5v2a5 5 0 0 0 10 0v-2.8"/><circle cx="19.5" cy="9" r="1.7"/><circle cx="6" cy="3" r="1"/><circle cx="9.5" cy="3" r="1"/>',
    "glasses": '<circle cx="6.5" cy="14.5" r="3.2"/><circle cx="17.5" cy="14.5" r="3.2"/><path d="M9.7 14.5h4.6M3.3 14.5 2 9M20.7 14.5 22 9M14.3 9H9.7"/>',
    "ruler": '<path d="M3.5 15.5 8.5 20.5 20.5 8.5 15.5 3.5 3.5 15.5Z"/><path d="m7 12 2 2M9.5 9.5l2 2M12 7l2 2"/>',
    "image": '<rect x="3.5" y="4.5" width="17" height="15" rx="1"/><circle cx="8.5" cy="9.5" r="1.6"/><path d="m5 17 5-5 3.5 3.5L18 10l2 2"/>',
    "bird": '<path d="M4 13.5c1-4 4-7.5 9-7.5 3 0 6 1.8 7 5-1 0-2 .3-2.8 1 2 .3 3.3 1.5 3.8 3.3-1.5.5-3-.1-4-1.1.3 2-.6 4-2.5 5.3-3.5-.3-6-2-7-5-2 .2-3.5-.3-4.5-1 .8-.2 1.5-.6 2-1-1.3-.2-2.4-1-3-2 .8.2 1.5.1 2-.3Z"/><circle cx="14.8" cy="10.2" r="0.6" fill="currentColor" stroke="none"/>',
    "gamepad": '<rect x="2.5" y="8" width="19" height="10" rx="4"/><path d="M7 11v4M5 13h4"/><circle cx="15.5" cy="11.5" r="1"/><circle cx="18" cy="14" r="1"/>',
    "droplet": '<path d="M12 3.5s6 6.7 6 10.8a6 6 0 0 1-12 0c0-4.1 6-10.8 6-10.8Z"/>',
    "stop-circle": '<circle cx="12" cy="12" r="9"/><rect x="9" y="9" width="6" height="6"/>',
    "plug": '<path d="M9 3v5M15 3v5"/><path d="M6 8h12v3.5a6 6 0 0 1-12 0V8Z"/><path d="M12 17.5V21"/>',
    "arrow-up-right": '<path d="M6 18 18 6"/><path d="M9 6h9v9"/>',
    "x-circle": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
    "refresh-cw": '<path d="M4 12a8 8 0 0 1 13.7-5.7L20 8.5"/><path d="M20 4v4.5h-4.5"/><path d="M20 12a8 8 0 0 1-13.7 5.7L4 15.5"/><path d="M4 20v-4.5h4.5"/>',
    "contrast": '<circle cx="12" cy="12" r="9"/><path d="M12 3v18a9 9 0 0 0 0-18Z" fill="currentColor" stroke="none"/>',
    "arrow-right": '<path d="M4 12h15.5"/><path d="m13 6 6.5 6-6.5 6"/>',
    "download": '<path d="M12 3.5v11.5"/><path d="m7 10.5 5 5 5-5"/><path d="M4.5 18.5h15v2h-15z" fill="currentColor" stroke="none"/>',
    "scale": '<path d="M12 3v17.5M8 20.5h8"/><path d="M12 5.5 5 7.5l3.3 6.7a3.7 3.7 0 0 0 7.4 0L19 7.5 12 5.5Z"/><path d="M5 7.5 1.7 14.2M19 7.5l3.3 6.7"/>',
    "book": '<path d="M5 4.5h9a3 3 0 0 1 3 3V21H8a3 3 0 0 1-3-3V4.5Z"/><path d="M17 21a3 3 0 0 0 3-3V6"/>',
    "book-open": '<path d="M12 6.5c-1.5-1.5-4-2-8-2v13c4 0 6.5.5 8 2 1.5-1.5 4-2 8-2v-13c-4 0-6.5.5-8 2Z"/><path d="M12 6.5V19.5"/>',
    "moon": '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"/>',
    "flame": '<path d="M12 3s-5.5 5-5.5 10a5.5 5.5 0 0 0 11 0c0-1.7-.8-2.7-1.5-3.7.2 2-1 3-1.8 2C15.5 9 15 6 12 3Z"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.5 4 5.7 4 9s-1.5 6.5-4 9c-2.5-2.5-4-5.7-4-9s1.5-6.5 4-9Z"/>',
    "cpu": '<rect x="7" y="7" width="10" height="10" rx="0.5"/><rect x="10" y="10" width="4" height="4"/><path d="M12 2.5V7M12 17v4.5M2.5 12H7M17 12h4.5M9 2.5V7M15 2.5V7M9 17v4.5M15 17v4.5M2.5 9H7M2.5 15H7M17 9h4.5M17 15h4.5"/>',
    "gpu": '<rect x="3" y="6" width="18" height="12" rx="0.5"/><rect x="5.5" y="8.5" width="3" height="3"/><rect x="10.5" y="8.5" width="3" height="3"/><rect x="15.5" y="8.5" width="3" height="3"/><rect x="5.5" y="13" width="3" height="3"/><rect x="10.5" y="13" width="3" height="3"/><rect x="15.5" y="13" width="3" height="3"/>',
    "layers": '<path d="M12 3.5 21 8l-9 4.5L3 8l9-4.5Z"/><path d="m3 12 9 4.5 9-4.5M3 16l9 4.5L21 16"/>',
    "bot": '<rect x="5" y="9" width="14" height="10" rx="2"/><path d="M12 5.5v3.5M12 3.5a1.2 1.2 0 1 0 0 2.4 1.2 1.2 0 0 0 0-2.4Z" fill="currentColor" stroke="none"/><circle cx="9" cy="14" r="1.3"/><circle cx="15" cy="14" r="1.3"/><path d="M9 17.5h6M2.5 12.5v3M21.5 12.5v3"/>',
    "atom": '<circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><ellipse cx="12" cy="12" rx="9" ry="3.6"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(120 12 12)"/>',
    "puzzle": '<path d="M9 4.5h4v2.3a1.8 1.8 0 0 0 3.5 0V4.5H20v4.2h-2.3a1.8 1.8 0 0 0 0 3.5H20V16h-3.5v-2.3a1.8 1.8 0 0 0-3.5 0V16H9v-3.5H6.7a1.8 1.8 0 0 1 0-3.5H9V4.5Z"/><path d="M9 16H4.5v-4"/>',
    "thermometer": '<path d="M12 3.5a2 2 0 0 0-2 2v9.3a4 4 0 1 0 4 0V5.5a2 2 0 0 0-2-2Z"/><path d="M12 9v6.5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.5l4 2.3"/>',
    "server": '<rect x="3.5" y="4" width="17" height="6.5" rx="0.5"/><rect x="3.5" y="13.5" width="17" height="6.5" rx="0.5"/><path d="M7 7.2h.01M7 16.7h.01" stroke-width="2.6"/>',
    "database": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><path d="m15 9-4.5 2.5L8 16l4.5-2.5L15 9Z"/>',
    "crosshair": '<circle cx="12" cy="12" r="8.5"/><path d="M12 2.5V6M12 18v3.5M2.5 12H6M18 12h3.5"/>',
    "heart": '<path d="M12 20.5s-7.5-4.6-9.7-9A5.4 5.4 0 0 1 12 6a5.4 5.4 0 0 1 9.7 5.5c-2.2 4.4-9.7 9-9.7 9Z"/>',
    "star": '<path d="M12 3.5 14.6 9l6 .9-4.3 4.2 1 6-5.3-2.8-5.3 2.8 1-6L3.4 9.9l6-.9L12 3.5Z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6" stroke-width="2.4"/><circle cx="12" cy="7.5" r="0.4" fill="currentColor" stroke="currentColor" stroke-width="2.2"/>',
    "help-circle": '<circle cx="12" cy="12" r="9"/><path d="M9.3 9.3a2.7 2.7 0 1 1 3.9 2.4c-.8.5-1.2 1-1.2 1.9"/><circle cx="12" cy="17" r="0.4" fill="currentColor" stroke="currentColor" stroke-width="2.2"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="m7.5 12.5 3 3 6-6.5"/>',
    "footprints": '<path d="M8 3.5c1.8 0 3 1.3 3 3.3 0 1.7-.7 2.5-1.5 3.5S8 12.5 8 14.3c0 1.7 1.2 2.7 1.2 2.7"/><ellipse cx="9" cy="19.5" rx="2.2" ry="1.4"/><path d="M16 8.5c1.8 0 3 1.3 3 3.3 0 1.7-.7 2.5-1.5 3.5s-1.5 2.2-1.5 4c0 1.7 1.2 2.7 1.2 2.7"/><ellipse cx="17" cy="4.4" rx="2.2" ry="1.4"/>',
    "cloud": '<path d="M7 18.5a4.2 4.2 0 0 1-.5-8.4 5.5 5.5 0 0 1 10.7-1.8 4.3 4.3 0 0 1-.7 10.2H7Z"/>',
    "message-circle": '<path d="M4 12a8 8 0 1 1 3.5 6.6L4 20l1.2-3.6A7.9 7.9 0 0 1 4 12Z"/>',
    "send": '<path d="M21 3 3 10.5l7 2.5 2.5 7L21 3Z"/><path d="M10.5 13 21 3"/>',
    "sliders": '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h13M21 18h0"/><circle cx="13" cy="6" r="2"/><circle cx="7" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    "play": '<path d="M6.5 4.5v15l13-7.5-13-7.5Z"/>',
    "pause": '<rect x="6.5" y="5" width="4" height="14"/><rect x="13.5" y="5" width="4" height="14"/>',
    "key": '<circle cx="8" cy="15.5" r="4"/><path d="M11 12.5 18.5 5M15.5 8.5 18 6M17.5 10.5 20 8"/>',
    "link": '<path d="M9.5 14.5 14.5 9.5"/><path d="M11 7.5 13.5 5a3.5 3.5 0 1 1 5 5l-2.5 2.5M13 16.5 10.5 19a3.5 3.5 0 1 1-5-5l2.5-2.5"/>',
    "users": '<circle cx="9" cy="9" r="3.3"/><path d="M3.5 19c0-3 2.5-5 5.5-5s5.5 2 5.5 5"/><path d="M16 6a3 3 0 0 1 0 6M18.5 19c0-2.5-1.7-4.4-4-4.9"/>',
    "check": '<path d="m4.5 12.5 5 5L20 6.5"/>',
    "x": '<path d="m5 5 14 14M19 5 5 19"/>',
    "plus": '<path d="M12 4.5v15M4.5 12h15"/>',
    "minus": '<path d="M4.5 12h15"/>',
    "chevron-right": '<path d="m9 5 7 7-7 7"/>',
    "battery": '<rect x="2.5" y="8" width="17" height="8" rx="1"/><path d="M21.5 10.5v3"/><rect x="5" y="10" width="10" height="4" fill="currentColor" stroke="none"/>',
    "shuffle": '<path d="m3 6 5 12M20 6h-5.5l-2 4M3 18h5.5l2-4M17 4l3 2-3 2M17 16l3 2-3 2"/>',
    "walk": '<circle cx="14" cy="4.5" r="1.6"/><path d="M12 8l3 2 1 4-2 6M15 10l4 1M9 22l3-5.5-1.5-4L14 9"/>',
    "briefcase": '<rect x="3" y="7.5" width="18" height="12" rx="1.5"/><path d="M8.5 7.5V5.5a1.5 1.5 0 0 1 1.5-1.5h4a1.5 1.5 0 0 1 1.5 1.5v2"/><path d="M3 12.5h18"/>',
    "menu-lines": '<path d="M4 6.5h16M4 12h16M4 17.5h16"/>',
    "tag": '<path d="M11.5 3.5H5a1.5 1.5 0 0 0-1.5 1.5v6.5a1.5 1.5 0 0 0 .44 1.06l8 8a1.5 1.5 0 0 0 2.12 0l6.5-6.5a1.5 1.5 0 0 0 0-2.12l-8-8a1.5 1.5 0 0 0-1.06-.44Z"/><circle cx="8.2" cy="8.2" r="1.3"/>',
    "cluster": '<circle cx="6" cy="7" r="2.1"/><circle cx="6.8" cy="12.5" r="1.7"/><circle cx="9.5" cy="4.5" r="1.5"/><circle cx="17.5" cy="8" r="2.3"/><circle cx="18.5" cy="13.5" r="1.6"/><circle cx="15" cy="16.5" r="1.4"/><circle cx="11.5" cy="18.5" r="2"/>',
}


def svg_icon(
    name: str,
    size: int = 18,
    color: str = "currentColor",
    stroke_width: float = 2.0,
    css_class: str = "",
) -> str:
    """Return a standalone inline <svg> markup string for the given icon name."""
    body = _ICONS.get(name)
    if body is None:
        # フォールバック: 不明なアイコン名は円で代替（無音の失敗にしない）
        body = '<circle cx="12" cy="12" r="8"/>'
    cls = f' icon-inline {css_class}'.rstrip()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{size}" height="{size}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="square" '
        f'stroke-linejoin="miter" class="{cls}" aria-hidden="true">{body}</svg>'
    )


# 短縮エイリアス
def icon(name: str, size: int = 18, color: str = "currentColor", css_class: str = "") -> str:
    return svg_icon(name, size=size, color=color, css_class=css_class)


def icon_html(
    name: str,
    label: str,
    size: int = 16,
    color: str = "currentColor",
    gap: str = "0.45em",
) -> str:
    """アイコン＋ラベルの横並びspanを返す（見出しやバッジ用）。"""
    return (
        f'<span style="display:inline-flex; align-items:center; gap:{gap};">'
        f'{svg_icon(name, size=size, color=color)}<span>{label}</span></span>'
    )


def heading(
    text: str,
    icon_name: str | None = None,
    level: int = 2,
    color: str = "currentColor",
    size: int | None = None,
) -> str:
    """st.markdown(..., unsafe_allow_html=True) にそのまま渡せる見出しHTMLを返す。

    既存CSS（.stApp h2 / h3 など）がそのまま適用されるよう、通常のh要素として出力する。
    """
    isize = size or (22 if level == 1 else 18 if level == 2 else 16)
    ic = svg_icon(icon_name, size=isize, color=color) if icon_name else ""
    return f'<h{level}>{ic} {text}</h{level}>'


def badge(
    text: str,
    icon_name: str | None = None,
    color: str = "var(--color-cyan)",
    size: int = 14,
) -> str:
    """math-badge / step-badge のようなインラインバッジ用。呼び出し側でclassを付与して使う。"""
    ic = svg_icon(icon_name, size=size, color=color) if icon_name else ""
    return f'<span style="display:inline-flex; align-items:center; gap:0.4em;">{ic}{text}</span>'
