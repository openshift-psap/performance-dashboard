"""Dashboard Styles Module.

Contains all CSS styling functions for the LLM Performance Dashboard.
"""

import streamlit as st


def get_app_css():
    """Get the main CSS styles for the application."""
    return """
    <style>
    /* Force light color scheme regardless of browser/OS dark mode preference */
    :root, html {
        color-scheme: light only !important;
    }

    /* Reduce top and side margins/padding */
    .main > div {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0 !important;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0 !important;
        max-width: none !important;
    }

    /* Reduce header gap */
    .main .block-container {
        padding-top: 0.5rem !important;
    }

    /* Reduce top padding on main content */
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
    }

    /* Streamlit top toolbar — let the custom title own the top surface */
    [data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }

    /* Main dashboard title card */
    .dashboard-titlebar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.9rem;
        padding: 1rem 1.25rem;
        margin: -0.25rem -0.35rem 0.9rem -0.35rem;
        border: 1px solid #e4e8ef;
        border-radius: 16px;
        background:
            radial-gradient(circle at top left, rgba(204, 0, 0, 0.08), transparent 28%),
            linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    .dashboard-title-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 3rem;
        height: 3rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #fff7f7 0%, #ffffff 100%);
        border: 1px solid rgba(204, 0, 0, 0.12);
        box-shadow: 0 4px 12px rgba(204, 0, 0, 0.08);
        flex-shrink: 0;
    }

    .dashboard-title-logo img {
        height: 28px;
        width: auto;
        display: block;
    }

    .dashboard-title-text {
        font-size: clamp(1.9rem, 2.6vw, 2.55rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        color: #111827;
        text-align: center;
        margin: 0;
    }

    /* Remove top margin from title */
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        color: #1a1f36 !important;
    }
    h2 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #1a1f36 !important;
        background: #f0f2f6;
        padding: 0.55rem 0.9rem !important;
        border-radius: 8px;
        border-bottom: 2px solid #e1e5e9;
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1a1f36 !important;
        background: #f5f6f8;
        padding: 0.4rem 0.75rem !important;
        border-radius: 6px;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }

    /* Moderate gap between header and its subtitle text */
    [data-testid="stHeading"] {
        padding-bottom: 0 !important;
        margin-bottom: 0.15rem !important;
    }
    [data-testid="stHeading"] + div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    div:has(> [data-testid="stHeading"]) {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div:has(> [data-testid="stHeading"]) + div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Reduce gap above first element */
    .element-container:first-child {
        margin-top: 0 !important;
    }

    @media (max-width: 768px) {
        .dashboard-titlebar {
            gap: 0.7rem;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
        }

        .dashboard-title-logo {
            width: 2.6rem;
            height: 2.6rem;
        }
    }

    /* Metric container spacing */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border: 1px solid #e8ecf1;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Filter dropdown borders and styling */
    [data-testid="stSelectbox"] > div > div {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: #b0b8c4 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stMultiSelect"] > div > div:hover {
        border-color: #b0b8c4 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    }

    /* Filter dropdown expanded menu borders */
    [data-baseweb="popover"] {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }

    [data-baseweb="select"] [data-baseweb="menu"] {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        height: auto !important; /* Allow the item's background to grow */
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span[title] {
        white-space: normal; /* Allow the text to wrap */
        max-width: 100%;
        display: inline-block;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4a9eff;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a1f36;
        margin: 0;
    }
    .kpi-subtitle {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* ── Overview Section ── */
    .overview-card {
        background:
            radial-gradient(circle at top left, rgba(204, 0, 0, 0.06), transparent 30%),
            linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e4e8ef;
        margin: 0.4rem 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .overview-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.09);
    }
    .overview-card details { cursor: pointer; }
    .overview-card details summary {
        list-style: none;
        display: flex;
        flex-direction: column;
    }
    .overview-card details summary::-webkit-details-marker { display: none; }
    .overview-card details summary::after {
        content: "ℹ️";
        position: absolute;
        top: 0.8rem;
        right: 1rem;
        font-size: 0.75rem;
        opacity: 0.45;
        transition: opacity 0.15s;
    }
    .overview-card { position: relative; }
    .overview-card:hover details summary::after { opacity: 0.8; }
    .overview-card details[open] summary::after { opacity: 1; }
    .overview-card-detail {
        margin-top: 0.7rem;
        padding-top: 0.6rem;
        border-top: 1px solid #e7ebf2;
        font-size: 0.82rem;
        line-height: 1.55;
        color: #4b5563;
    }
    .overview-card-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }
    .overview-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1f36;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .overview-card-value .icon {
        font-size: 1.3rem;
    }
    .val-green,
    .overview-card-value.val-green,
    .vllm-stat-value.val-green,
    b.val-green { color: #15803d !important; }
    .val-red,
    .overview-card-value.val-red,
    .vllm-stat-value.val-red,
    b.val-red { color: #b91c1c !important; }
    .val-amber,
    .overview-card-value.val-amber,
    .vllm-stat-value.val-amber,
    b.val-amber { color: #ca8a04 !important; }
    .val-blue,
    .overview-card-value.val-blue,
    .vllm-stat-value.val-blue,
    b.val-blue { color: #1d4ed8 !important; }

    /* Health traffic-light dots */
    .health-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .health-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-grey   { background: #d1d5db; }
    .dot-green  { background: #16a34a; }
    .dot-amber  { background: #d97706; }
    .dot-red    { background: #dc2626; }

    /* Accelerator / model-family cards */
    .overview-family-card {
        background: #ffffff;
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 0.35rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        min-height: 160px;
        transition: box-shadow 0.15s ease;
        position: relative;
    }
    .overview-family-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .overview-family-card details { cursor: pointer; }
    .overview-family-card details summary { list-style: none; }
    .overview-family-card details summary::-webkit-details-marker { display: none; }
    .overview-family-card details summary::after {
        content: "ℹ️";
        position: absolute;
        bottom: 0.6rem;
        right: 0.8rem;
        font-size: 0.7rem;
        opacity: 0.4;
        transition: opacity 0.15s;
    }
    .overview-family-card:hover details summary::after { opacity: 0.8; }
    .overview-family-card details[open] summary::after { opacity: 1; }
    .overview-family-card.status-healthy {
        border-left: 4px solid #16a34a;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    }
    .overview-family-card.status-warning {
        border-left: 4px solid #d97706;
        background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
    }
    .overview-family-card.status-regression {
        border-left: 4px solid #dc2626;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
    }
    .family-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .family-card-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1f36;
    }
    .family-card-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 9999px;
        letter-spacing: 0.02em;
    }
    .badge-healthy    { background: #dcfce7; color: #166534; }
    .badge-warning    { background: #fef3c7; color: #92400e; }
    .badge-regression { background: #fee2e2; color: #991b1b; }
    .family-card-stat {
        font-size: 0.82rem;
        color: #4b5563;
        margin: 0.15rem 0;
    }

    /* vLLM competitive scorecard */
    .vllm-scorecard {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .vllm-scorecard.vllm-hue-green {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
    }
    .vllm-scorecard.vllm-hue-green .vllm-scorecard-title { color: #166534; }
    .vllm-scorecard.vllm-hue-yellow {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border: 1px solid #fde68a;
    }
    .vllm-scorecard.vllm-hue-yellow .vllm-scorecard-title { color: #854d0e; }
    .vllm-scorecard.vllm-hue-red {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 1px solid #fca5a5;
    }
    .vllm-scorecard.vllm-hue-red .vllm-scorecard-title { color: #991b1b; }
    .vllm-scorecard-title {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .vllm-stat-row {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
        align-items: baseline;
    }
    .vllm-stat {
        text-align: center;
    }
    .vllm-stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .vllm-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1f36;
    }
    .vllm-scorecard { position: relative; }
    .vllm-scorecard details { cursor: pointer; }
    .vllm-scorecard details summary { list-style: none; }
    .vllm-scorecard details summary::-webkit-details-marker { display: none; }
    .vllm-scorecard details summary::after {
        content: "ℹ️";
        position: absolute;
        top: 0.8rem;
        right: 1rem;
        font-size: 0.75rem;
        opacity: 0.4;
        transition: opacity 0.15s;
    }
    .vllm-scorecard:hover details summary::after { opacity: 0.8; }
    .vllm-scorecard details[open] summary::after { opacity: 1; }

    .vllm-parity-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
        font-size: 0.82rem;
    }
    .vllm-parity-table th {
        padding: 0.4rem 0.6rem;
        border-bottom: 2px solid #d1d5db;
        font-weight: 600;
        white-space: nowrap;
    }
    .vllm-parity-table td {
        padding: 0.35rem 0.6rem;
        border-bottom: 1px solid #e5e7eb;
        white-space: nowrap;
    }
    .vllm-parity-table tbody tr:hover { background: rgba(0,0,0,0.03); }

    /* Regression heatmap */
    .heatmap-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.88rem;
    }
    .heatmap-table th {
        background: #f9fafb;
        color: #6b7280;
        font-weight: 600;
        padding: 0.7rem 1rem;
        text-align: center;
        border-bottom: 2px solid #e5e7eb;
    }
    .heatmap-table th:first-child {
        text-align: left;
    }
    .heatmap-table td {
        padding: 0.65rem 1rem;
        text-align: center;
        font-weight: 600;
        border-bottom: 1px solid #f3f4f6;
    }
    .heatmap-table td:first-child {
        text-align: left;
        font-weight: 600;
        color: #1a1f36;
    }
    .heatmap-table tr:last-child td {
        border-bottom: none;
    }
    .hm-cell {
        border-radius: 6px;
        padding: 0.45rem 0.8rem;
        display: inline-block;
        min-width: 70px;
    }
    .hm-improve-strong { background: #bbf7d0; color: #14532d; }
    .hm-similar        { background: #fef3c7; color: #92400e; }
    .hm-neutral        { background: #f3f4f6; color: #6b7280; }
    .hm-regress-strong { background: #fca5a5; color: #7f1d1d; }

    /* Extras: New-in-release callout */
    .new-release-callout {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }
    .new-release-callout-title {
        font-weight: 700;
        color: #166534;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .new-release-item {
        font-size: 0.82rem;
        color: #4b5563;
        padding: 0.1rem 0;
    }

    /* Expander card styling */
    [data-testid="stExpander"] {
        border: 1px solid #e8ecf1 !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 1rem !important;
        overflow: hidden !important;
        transition: box-shadow 0.15s ease !important;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }

    /* Chart container card styling */
    [data-testid="stPlotlyChart"] {
        border: 1px solid #e8ecf1;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.75rem;
        background: #ffffff;
        overflow: visible;
    }

    /* Center Plotly modebar icons within their buttons */
    .modebar-btn {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .modebar-btn svg {
        display: block !important;
        margin: auto !important;
    }

    /* DataFrame container styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #e8ecf1 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    /* Increase font size of collapsible section headings */
    .stExpander > details > summary {
        font-size: 2.34rem !important;
        font-weight: 600 !important;
        padding: 15px 16px !important;
        line-height: 1.4 !important;
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* In-page section navigation: sticky column, just below the divider line */
    [data-testid="stColumn"]:has(.section-nav-marker) > div {
        position: sticky;
        top: 3rem;
        height: fit-content;
        align-self: flex-start;
        margin-top: -1.5rem;
    }


    /* Section nav: container — transparent bg with visible border */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] {
        gap: 0 !important;
        border: 1px solid rgba(151, 166, 195, 0.25);
        border-radius: 6px;
        overflow: hidden;
        background: transparent;
    }

    /* Section nav: each row */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label {
        display: flex !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding: 0.75rem 1rem !important;
        margin: 0 !important;
        border-radius: 0 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, border-left 0.15s ease !important;
        border-bottom: 1px solid rgba(151, 166, 195, 0.18) !important;
        border-left: 3px solid transparent !important;
    }

    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label:last-child {
        border-bottom: none !important;
    }

    /* Hide radio circle indicator — target only the div that does NOT contain text */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label > div:not(:has([data-testid="stMarkdownContainer"])):not([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }

    /* Text container: fill width and lay out for chevron */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label [data-testid="stMarkdownContainer"] {
        display: flex !important;
        flex: 1;
        align-items: center;
        justify-content: space-between;
    }

    /* Chevron on the right side */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label [data-testid="stMarkdownContainer"]::after {
        content: "›";
        font-size: 1.2rem;
        opacity: 0.4;
        padding-left: 0.5rem;
        flex-shrink: 0;
    }

    /* Hover state */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label:hover {
        background-color: rgba(151, 166, 195, 0.08) !important;
    }

    /* Selected/active nav item — cover both possible Streamlit attributes */
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label[data-checked="true"],
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label[aria-checked="true"],
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label:has(input:checked) {
        border-left: 3px solid #cc0000 !important;
        background-color: rgba(204, 0, 0, 0.06) !important;
        font-weight: 600 !important;
    }

    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label[data-checked="true"] [data-testid="stMarkdownContainer"]::after,
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label[aria-checked="true"] [data-testid="stMarkdownContainer"]::after,
    [data-testid="stColumn"]:has(.section-nav-marker) [role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"]::after {
        opacity: 0.7;
    }

    /* ── Hamburger button (sidebar expand & collapse) ── */
    button.hamburger-btn {
        width: 3.2rem !important;
        height: 3.2rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    button.hamburger-btn > * {
        display: none !important;
    }
    button.hamburger-btn::after {
        content: "\\2630" !important;
        font-size: 2rem !important;
        color: #374151 !important;
        line-height: 1 !important;
    }

    /* ── Sidebar navigation styling ── */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
        border-right: 1px solid #e8ecf1 !important;
    }

    /* Hide zero-height components.html iframes and their wrappers */
    [data-testid="stCustomComponentV1"]:has(iframe[height="0"]),
    [data-testid="element-container"]:has([data-testid="stCustomComponentV1"]),
    div:has(> [data-testid="stCustomComponentV1"]:only-child) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide Streamlit bottom container and footer decoration */
    [data-testid="stBottom"],
    [data-testid="stDecoration"],
    footer {
        display: none !important;
        height: 0 !important;
    }

    /* ── Seamless section / filter / view transitions ── */

    /* Prevent content area from collapsing to zero during Streamlit reruns */
    .block-container {
        min-height: 100vh;
    }

    /* Fade-in animation for freshly rendered content */
    @keyframes contentFadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    [data-testid="stMain"] > .block-container {
        animation: contentFadeIn 0.2s ease-out;
    }

    /* Dim stale elements quickly during reruns so the progressive
       rebuild is barely visible; a short transition avoids a harsh
       snap-to-invisible that reads as a "flash". */
    [data-stale="true"] {
        opacity: 0 !important;
        transition: opacity 0.06s ease-out !important;
    }

    /* Batch-reveal filter rows: widgets render one-by-one during a rerun.
       Hide the row during that build-up, then fade the whole row in at
       once so individual widgets don't "flash" into view sequentially. */
    [data-testid="stHorizontalBlock"]:has(
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"]
    ) {
        animation: contentFadeIn 0.18s ease-out 0.08s both;
    }

    /* Prevent individual widget containers from causing cascading reflow
       while their siblings are still being mounted by Streamlit. */
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"] {
        contain: layout style;
    }

    /* Scrollable sidebar content */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow-y: auto !important;
        padding: 0.4rem 0.75rem 2rem 0.75rem !important;
    }

    /* Tighten vertical gaps between sidebar elements */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.15rem !important;
    }

    [data-testid="stSidebar"] .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.25rem 0 0.5rem 0;
        border-bottom: 1px solid #e1e5e9;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] .sidebar-logo img {
        height: 32px;
    }

    [data-testid="stSidebar"] .sidebar-logo .sidebar-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1f36;
        line-height: 1.2;
    }

    [data-testid="stSidebar"] .nav-group-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        padding: 0.6rem 0 1.3rem 0.25rem;
        margin: 0;
        line-height: 1.3;
        border-top: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] .nav-group-header:first-of-type {
        border-top: none;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #4b5563 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        padding: 0.5rem 0.6rem !important;
        margin: 0 !important;
        transition: background-color 0.15s ease !important;
        box-shadow: none !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:focus {
        box-shadow: none !important;
        border: none !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(204, 0, 0, 0.06) !important;
        border: none !important;
        border-left: 3px solid #cc0000 !important;
        border-radius: 0 8px 8px 0 !important;
        color: #1a1f36 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.6rem 0.5rem 0.75rem !important;
        margin: 0 !important;
        box-shadow: none !important;
        min-height: 0 !important;
        line-height: 1.4 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: rgba(204, 0, 0, 0.10) !important;
        border: none !important;
        border-left: 3px solid #cc0000 !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"]:focus {
        box-shadow: none !important;
        border: none !important;
        border-left: 3px solid #cc0000 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] p,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] span {
        color: #1a1f36 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] span {
        color: #4b5563 !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0 0.6rem 0 !important;
        border-color: #e1e5e9 !important;
    }

    </style>
    """


def get_auto_mode_css():
    """Get CSS for auto theme mode."""
    return """
    <style>
    /* Auto mode - respect browser's color scheme preference */

    /* Light mode (default) */
    .stApp {
        background-color: #fafbfd;
        color: #1a1f36;
    }

    .main {
        background-color: #fafbfd;
    }

    .main-title {
        color: #1a1f36;
    }

    .main-subtitle {
        color: #6b7280;
    }

    .kpi-card {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
        color: #1a1f36;
        border-left-color: #4a9eff;
    }
    .kpi-title {
        color: #4b5563;
    }
    .kpi-value {
        color: #1a1f36;
    }
    .kpi-subtitle {
        color: #6b7280;
    }

    /* Dark mode when browser prefers dark */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }

        .main {
            background-color: #0e1117 !important;
        }

        .main-title {
            color: #58a6ff !important;
        }

        .main-subtitle {
            color: #c9d1d9 !important;
        }

        .kpi-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #21262d 100%) !important;
            color: #c9d1d9 !important;
            border-left-color: #58a6ff !important;
        }
        .kpi-title {
            color: #c9d1d9 !important;
        }
        .kpi-value {
            color: #58a6ff !important;
        }
        .kpi-subtitle {
            color: #8b949e !important;
        }

        h1, h2, h3 {
            color: #58a6ff !important;
        }

        p {
            color: #c9d1d9 !important;
        }
    }

    /* Auto mode styling for error banner */
    .no-data-error-banner {
        border: 2px solid #ff9500;
        background: rgba(255, 149, 0, 0.1);
        color: #333;
    }
    .no-data-error-banner h2, .no-data-error-banner h3 {
        color: #333;
    }
    </style>
    """


def get_dark_mode_css():
    """Get CSS for forced dark theme mode."""
    return """
    <style>
    /* Force dark mode */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    .main {
        background-color: #0e1117 !important;
    }

    .main-title {
        color: #58a6ff !important;
    }

    .main-subtitle {
        color: #c9d1d9 !important;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #21262d 100%) !important;
        color: #c9d1d9 !important;
        border-left-color: #58a6ff !important;
    }
    .kpi-title {
        color: #c9d1d9 !important;
    }
    .kpi-value {
        color: #58a6ff !important;
    }
    .kpi-subtitle {
        color: #8b949e !important;
    }

    h1, h2, h3 {
        color: #58a6ff !important;
    }

    p {
        color: #c9d1d9 !important;
    }

    /* Dark mode styling for error banner */
    .no-data-error-banner {
        border: 2px solid #ffd43b !important;
        background: rgba(255, 212, 59, 0.1) !important;
        color: #e3e8ee !important;
    }
    .no-data-error-banner h2, .no-data-error-banner h3 {
        color: #e3e8ee !important;
    }

    /* Filter dropdown borders for dark mode */
    [data-testid="stSelectbox"] > div > div {
        border: 2px solid #4a5568 !important;
        border-radius: 4px !important;
        background-color: #2d3748 !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        border: 2px solid #4a5568 !important;
        border-radius: 4px !important;
        background-color: #2d3748 !important;
    }

    /* Dark mode expanded menu borders */
    [data-baseweb="popover"] {
        border: 2px solid #4a5568 !important;
        background-color: #2d3748 !important;
        color: #fafafa !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }

    [data-baseweb="select"] [data-baseweb="menu"] {
        border: 2px solid #4a5568 !important;
        background-color: #2d3748 !important;
        color: #fafafa !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }

    /* Comprehensive dark mode input styling */
    /* Override all dropdown and input elements */
    [data-testid="stSelectbox"] div {
        background-color: #2d3748 !important;
        color: #fafafa !important;
    }

    [data-testid="stMultiSelect"] div {
        background-color: #2d3748 !important;
        color: #fafafa !important;
    }

    /* Override Streamlit containers for dark mode */
    .main {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    .block-container {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    /* Override all text elements */
    p, div, span, label {
        color: #fafafa !important;
    }

    /* Override Streamlit specific text elements */
    .stMarkdown, .stText {
        color: #fafafa !important;
    }

    /* Override buttons for dark mode */
    button {
        background-color: #2d3748 !important;
        color: #fafafa !important;
        border: 1px solid #4a5568 !important;
    }

    button:hover {
        background-color: #1a202c !important;
        color: #fafafa !important;
    }

    /* Override remaining text in main app area */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li {
        color: #fafafa !important;
    }

    /* Re-apply specific colors for important elements */
    .main-title, h1 {
        color: #58a6ff !important;
    }

    .kpi-value {
        color: #58a6ff !important;
    }
    </style>
    """


def get_light_mode_css():
    """Get CSS for forced light theme mode."""
    return """
    <style>
    /* Force light mode - comprehensive overrides */
    .stApp {
        background-color: #fafbfd !important;
        color: #1a1f36 !important;
    }

    .main {
        background-color: #fafbfd !important;
        color: #1a1f36 !important;
    }

    /* Override all Streamlit containers */
    .block-container {
        background-color: #fafbfd !important;
        color: #1a1f36 !important;
    }

    /* Override sidebar if present */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }

    /* Override all text elements */
    .main-title {
        color: #1a1f36 !important;
    }

    .main-subtitle {
        color: #6b7280 !important;
    }

    /* Override all headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1f36 !important;
    }
    h2 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
        background: #f0f2f6 !important;
        padding: 0.55rem 0.9rem !important;
        border-radius: 8px !important;
        border-bottom: 2px solid #e1e5e9 !important;
    }
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        background: #f5f6f8 !important;
        padding: 0.4rem 0.75rem !important;
        border-radius: 6px !important;
    }

    /* Override all text */
    p, div, span, label {
        color: #1a1f36 !important;
    }

    /* Override Streamlit specific text elements */
    .stMarkdown, .stText {
        color: #1a1f36 !important;
    }

    /* Override buttons - comprehensive targeting */
    .stButton > button {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target all button variants */
    button[data-testid="baseButton-secondary"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    button[data-testid="baseButton-primary"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Override all button elements */
    button {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Button hover states */
    button:hover {
        background-color: #f8f9fa !important;
        color: #1a1f36 !important;
    }

    /* Specific targeting for filter control buttons */
    button[title*="Reset"], button[title*="Clear"], button[title*="Share"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Override metrics */
    .stMetric {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
    }

    .kpi-card {
        background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%) !important;
        color: #1a1f36 !important;
        border-left: 4px solid #4a9eff !important;
        border: 1px solid #e1e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }
    .kpi-title {
        color: #4b5563 !important;
    }
    .kpi-value {
        color: #1a1f36 !important;
    }
    .kpi-subtitle {
        color: #6b7280 !important;
    }

    /* Light mode styling for error banner (default) */
    .no-data-error-banner {
        border: 2px solid #ff9500 !important;
        background: rgba(255, 149, 0, 0.1) !important;
        color: #333 !important;
    }
    .no-data-error-banner h2, .no-data-error-banner h3 {
        color: #333 !important;
    }

    /* Filter dropdown borders for light mode */
    [data-testid="stSelectbox"] > div > div {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }

    /* Light mode expanded menu borders */
    [data-baseweb="popover"] {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }

    [data-baseweb="select"] [data-baseweb="menu"] {
        border: 1px solid #dde1e8 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }

    /* Override all dropdown and input elements */
    [data-testid="stSelectbox"] div {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
    }

    [data-baseweb="tag"] {
        background-color: #cc0000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
    }

    [data-baseweb="tag"] span,
    [data-baseweb="tag"] span[title] {
        color: #ffffff !important;
    }

    [data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Override expander styling */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #e8ecf1 !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stExpander"] > div,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details > div,
    [data-testid="stExpander"] details[open] > div,
    [data-testid="stExpander"] details[open] > div > div {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
    }

    /* Override dataframe */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #e8ecf1 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    /* Override plots and charts within expanded sections */
    [data-testid="stPlotlyChart"] {
        background-color: #ffffff !important;
    }

    /* Override plot containers */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }

    /* Override any plot backgrounds */
    .plot-container {
        background-color: #ffffff !important;
    }

    /* Fix Plotly chart text and controls for light mode */
    .plotly .modebar {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        flex-direction: row !important;
    }

    .plotly .modebar-group {
        display: flex !important;
        flex-wrap: nowrap !important;
    }

    .plotly .modebar-btn {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        fill: #1a1f36 !important;
    }

    .plotly .modebar-btn:hover {
        background-color: #f8f9fa !important;
        color: #1a1f36 !important;
        fill: #1a1f36 !important;
    }

    /* Fix chart axis labels and text */
    .plotly text {
        fill: #1a1f36 !important;
        color: #1a1f36 !important;
    }

    .plotly .xtick text, .plotly .ytick text {
        fill: #1a1f36 !important;
        color: #1a1f36 !important;
    }

    /* Fix chart legends */
    .plotly .legend {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
    }

    .plotly .legend text {
        fill: #1a1f36 !important;
        color: #1a1f36 !important;
    }

    /* Fix chart titles and axis titles */
    .plotly .gtitle text, .plotly .xtitle text, .plotly .ytitle text {
        fill: #1a1f36 !important;
        color: #1a1f36 !important;
    }

    /* Fix hover labels */
    .plotly .hoverlayer .hovertext {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Fix SVG text elements in charts */
    .js-plotly-plot svg text {
        fill: #1a1f36 !important;
    }

    .js-plotly-plot .plotly-notifier {
        color: #1a1f36 !important;
    }

    /* Fix chart grid and background */
    .js-plotly-plot .bg {
        fill: #ffffff !important;
    }

    .js-plotly-plot .gridlayer .crisp {
        stroke: #e1e5e9 !important;
    }

    /* Override any remaining chart elements */
    .user-select-none {
        background-color: transparent !important;
    }

    /* Override info boxes and alerts */
    [data-testid="stAlert"][data-baseweb="notification"][kind="info"],
    [data-testid="stNotification"][kind="info"] {
        background-color: #e7f3ff !important;
        color: #1a1f36 !important;
        border: 1px solid #b3d9ff !important;
        border-radius: 6px !important;
    }

    [data-testid="stAlert"][kind="success"],
    [data-testid="stNotification"][kind="success"] {
        background-color: #d1f2d1 !important;
        color: #1a1f36 !important;
        border: 1px solid #a3d977 !important;
        border-radius: 6px !important;
    }

    [data-testid="stAlert"][kind="warning"],
    [data-testid="stNotification"][kind="warning"] {
        background-color: #fff3cd !important;
        color: #1a1f36 !important;
        border: 1px solid #ffd93d !important;
        border-radius: 6px !important;
    }

    [data-testid="stAlert"][kind="error"],
    [data-testid="stNotification"][kind="error"] {
        background-color: #f8d7da !important;
        color: #1a1f36 !important;
        border: 1px solid #f1a7aa !important;
        border-radius: 6px !important;
    }

    /* Override code blocks */
    [data-testid="stCode"] {
        background-color: #f8f9fa !important;
        color: #1a1f36 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }

    pre {
        background-color: #f8f9fa !important;
        color: #1a1f36 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }

    code {
        background-color: #f8f9fa !important;
        color: #1a1f36 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 3px !important;
        padding: 2px 4px !important;
    }

    /* Override all containers and content areas */
    [data-testid="stElementContainer"],
    .element-container {
        background-color: transparent !important;
    }

    /* Override remaining text in main app area */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li {
        color: #1a1f36 !important;
    }

    /* Additional button targeting - catch all variations */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    button[kind="primary"] {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target buttons by their container classes */
    .stButton button {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target buttons by their parent element */
    div[data-testid="stColumn"] button {
        background-color: #ffffff !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Ensure button text is visible */
    .stButton button span {
        color: #1a1f36 !important;
    }

    /* Re-apply specific colors for important elements */
    .main-title, h1 {
        color: #1a1f36 !important;
    }

    .kpi-value {
        color: #1a1f36 !important;
    }

    /* ── IntelliConfig wizard styles ── */

    /*
     * Clickable cards.  Each card is a st.container(border=True, key="ic_...").
     * Streamlit adds class "st-key-<key>" to the outermost wrapper, so
     * [class*="st-key-ic_"] reliably targets every IntelliConfig card.
     */
    [class*="st-key-ic_ic_"] {
        transition: border-color 0.15s, box-shadow 0.15s !important;
        cursor: pointer !important;
        position: relative !important;
        height: 100% !important;
    }
    /* Equal-height cards within a row */
    [class*="st-key-ic_grid_"] [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [class*="st-key-ic_grid_"] [data-testid="stColumn"] > [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 !important;
    }
    /* Stretch the button over the entire card so clicking anywhere selects */
    [class*="st-key-ic_btn_"] button {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 1 !important;
    }
    [class*="st-key-ic_btn_"] {
        position: static !important;
    }
    /* Selected card – blue border */
    [class*="st-key-ic_ic_"]:has(.ic-state-selected) {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
    }
    /* Disabled card */
    [class*="st-key-ic_ic_"]:has(.ic-state-disabled) {
        background: #f3f4f6 !important;
        opacity: 0.55 !important;
        cursor: not-allowed !important;
    }
    /* Hover – only on non-selected, non-disabled cards */
    [class*="st-key-ic_ic_"]:not(:has(.ic-state-selected)):not(:has(.ic-state-disabled)):hover {
        border-color: #b4bfed !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.10) !important;
    }
    .ic-card-header {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.25rem;
    }
    .ic-card-icon {
        font-size: 1.25rem;
    }
    .ic-card-name {
        font-weight: 650;
        font-size: 1rem;
        color: #1a1f36;
    }
    .ic-card-radio {
        margin-left: auto;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid #d1d5db;
        display: inline-block;
    }
    .ic-card-radio-selected {
        margin-left: auto;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid #4f46e5;
        background: #4f46e5;
        display: inline-block;
        position: relative;
    }
    .ic-card-radio-selected::after {
        content: '';
        position: absolute;
        top: 3px;
        left: 3px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #fff;
    }
    .ic-card-desc {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .ic-disabled-note {
        color: #9ca3af;
        font-style: italic;
    }
    .ic-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
    }
    .ic-badge-muted {
        display: inline-block;
        background: #f3f4f6;
        color: #9ca3af;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
    }

    /* Step sidebar labels (rendered below the button, purely visual) */
    .ic-step, .ic-step-active {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.65rem 0.8rem;
        border-radius: 8px;
        font-size: 0.92rem;
        color: #6b7280;
        pointer-events: none;
    }
    .ic-step-active {
        background: #f0f0ff;
        color: #1a1f36;
        font-weight: 600;
    }
    .ic-step-num, .ic-step-num-active {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .ic-step-num {
        border: 2px solid #d1d5db;
        color: #9ca3af;
    }
    .ic-step-num-active {
        background: #4f46e5;
        color: #fff;
        border: none;
    }

    /* Clickable step nav: button covers the row, styled to match */
    [class*="st-key-ic_stepnav_"] {
        position: relative !important;
    }
    [class*="st-key-ic_stepnav_"] button {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        z-index: 2 !important;
        border: none !important;
        background: transparent !important;
    }
    [class*="st-key-ic_stepnav_"]:hover .ic-step {
        background: #f9fafb;
    }

    /* ── Export step polish ── */

    /* Selection summary chips */
    .ic-export-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .ic-export-chip {
        display: inline-block;
        background: #f0f0ff;
        color: #3730a3;
        font-size: 0.82rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #e0e0f7;
    }

    /* Styled tabs */
    [class*="st-key-ic_export_tabs"] [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 2px solid #e5e7eb !important;
        background: transparent !important;
    }
    [class*="st-key-ic_export_tabs"] [data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.25rem !important;
        color: #6b7280 !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        background: transparent !important;
        transition: color 0.15s, border-color 0.15s !important;
    }
    [class*="st-key-ic_export_tabs"] [data-baseweb="tab"]:hover {
        color: #4f46e5 !important;
    }
    [class*="st-key-ic_export_tabs"] [aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
    [class*="st-key-ic_export_tabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    [class*="st-key-ic_export_tabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Code blocks within export */
    [class*="st-key-ic_export_tabs"] [data-testid="stCode"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }

    /* Action buttons */
    [class*="st-key-ic_export_actions"] button {
        background: #fff !important;
        color: #4f46e5 !important;
        border: 1.5px solid #4f46e5 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: background 0.15s, box-shadow 0.15s !important;
    }
    [class*="st-key-ic_export_actions"] button:hover {
        background: #f0f0ff !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.12) !important;
    }

    /* Cancel button: red outline style */
    [class*="st-key-ic_cancel_wrap"] button {
        background: #fff !important;
        color: #dc2626 !important;
        border: 1.5px solid #dc2626 !important;
    }
    [class*="st-key-ic_cancel_wrap"] button:hover {
        background: #fef2f2 !important;
    }

    /* Grid section spacing */
    [class*="st-key-ic_grid_"] {
        margin-bottom: 0.5rem;
    }

    /* Responsive card grid: collapse to 2 columns on narrower viewports */
    @media (max-width: 1200px) {
        [class*="st-key-ic_grid_"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [class*="st-key-ic_grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 0 0 calc(50% - 0.5rem) !important;
            min-width: calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
        }
    }

    </style>
    """


def apply_theme_css():
    """Apply light theme CSS unconditionally."""
    st.markdown(get_light_mode_css(), unsafe_allow_html=True)


def get_mlperf_dashboard_css():
    """Get MLPerf dashboard specific CSS for tabs, tooltips, and filters.

    Returns:
        CSS string for MLPerf dashboard styling
    """
    tooltip_css = """
        /* Force light mode for all tooltips and popovers */
        [role="tooltip"],
        [data-testid="stTooltipContent"],
        .stTooltipIcon,
        [data-baseweb="tooltip"],
        div[class*="tooltip"],
        div[class*="Tooltip"] {
            background-color: white !important;
            color: #1a1f36 !important;
            border: 1px solid #d1d5db !important;
        }

        /* Nested divs in tooltips */
        [role="tooltip"] *,
        [data-testid="stTooltipContent"] *,
        [data-baseweb="tooltip"] * {
            color: #1a1f36 !important;
        }
        """

    return f"""
    <style>
    /* Make MLPerf tabs bigger */
    .stTabs [data-baseweb="tab-list"] button {{
        font-size: 2rem;
        padding: 1.5rem 3rem;
        font-weight: 700;
        height: auto;
        min-height: 70px;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        font-size: 2.1rem;
    }}

    {tooltip_css}

    /* Fix dropdown/multiselect visibility in light mode */
    [data-baseweb="popover"] {{
        background-color: white !important;
        color: #1a1f36 !important;
    }}

    [data-baseweb="select"] [data-baseweb="menu"] {{
        background-color: white !important;
        color: #1a1f36 !important;
    }}

    [data-baseweb="menu"] li {{
        background-color: white !important;
        color: #1a1f36 !important;
    }}

    [data-baseweb="menu"] li:hover {{
        background-color: #f0f2f6 !important;
        color: #1a1f36 !important;
    }}

    /* Fix multiselect tags visibility */
    [data-baseweb="tag"] {{
        background-color: #cc0000 !important;
        color: #ffffff !important;
    }}

    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}
    </style>
    """


def get_mlperf_table_tooltip_css():
    """Get CSS for MLPerf detailed results table tooltips (light mode)."""
    return """
    <style>
    [role="tooltip"],
    [data-testid="stTooltipContent"],
    [data-baseweb="tooltip"],
    .stTooltip,
    div[class*="tooltip"],
    div[class*="Tooltip"],
    div[data-baseweb="tooltip"],
    body [role="tooltip"] {
        background: white !important;
        background-color: white !important;
        color: #1a1f36 !important;
        border: 1px solid #d1d5db !important;
    }

    [role="tooltip"] *,
    [data-testid="stTooltipContent"] *,
    [data-baseweb="tooltip"] *,
    div[class*="tooltip"] *,
    div[class*="Tooltip"] * {
        background: transparent !important;
        color: #1a1f36 !important;
    }
    </style>
    """


def generate_color_palette(n_colors: int) -> list[str]:
    """Generate a list of maximally distinct colors for visualization.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color strings
    """
    base_colors = [
        "#e6194b",  # Red
        "#3cb44b",  # Green
        "#ffe119",  # Yellow
        "#4363d8",  # Blue
        "#f58231",  # Orange
        "#46f0f0",  # Cyan
        "#f032e6",  # Magenta
        "#bcf60c",  # Lime
        "#9a6324",  # Brown
        "#800000",  # Maroon
        "#000075",  # Navy
        "#808080",  # Grey
        "#ff6347",  # Tomato
        "#ee82ee",  # Violet
        "#00ced1",  # DarkTurquoise
        "#ff1493",  # DeepPink
        "#1e90ff",  # DodgerBlue
        "#ff69b4",  # HotPink
        "#cd5c5c",  # IndianRed
        "#4b0082",  # Indigo
        "#7cfc00",  # LawnGreen
        "#add8e6",  # LightBlue
        "#ff00ff",  # Magenta2
        "#800080",  # Purple2
        "#ff0000",  # Red2
        "#fa8072",  # Salmon
        "#2e8b57",  # SeaGreen
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    return base_colors


def initialize_session_state():
    """Initialize session state variables for styling."""
    # Initialize theme state with auto-detection
    if "theme_initialized" not in st.session_state:
        st.session_state.theme_initialized = True
        st.session_state.theme_mode = "light"  # Options: "light", "dark", "auto"


def initialize_streamlit_config():
    """Initialize Streamlit configuration."""
    st.set_page_config(
        page_title="Staging Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
