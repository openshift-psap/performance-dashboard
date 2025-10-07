"""Dashboard Styles Module.

Contains all CSS styling functions for the LLM Performance Dashboard.
"""

import streamlit as st


def get_app_css():
    """Get the main CSS styles for the application."""
    return """
    <style>
    /* Reduce top and side margins/padding */
    .main > div {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2.5rem !important;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2.5rem !important;
        max-width: none !important;
    }

    /* Reduce header gap */
    .main .block-container {
        padding-top: 2.5rem !important;
    }

    /* Reduce top padding on main content */
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
    }

    /* Remove top margin from title */
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Reduce gap above first element */
    .element-container:first-child {
        margin-top: 0 !important;
    }

    /* Filter dropdown borders and styling */
    [data-testid="stSelectbox"] > div > div {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
    }

    /* Filter dropdown expanded menu borders */
    [data-baseweb="popover"] {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    [data-baseweb="select"] [data-baseweb="menu"] {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Dark mode filter dropdown borders */
    @media (prefers-color-scheme: dark) {
        [data-testid="stSelectbox"] > div > div {
            border: 2px solid #4a5568 !important;
            background-color: #2d3748 !important;
        }

        [data-testid="stMultiSelect"] > div > div {
            border: 2px solid #4a5568 !important;
            background-color: #2d3748 !important;
        }

        /* Dark mode expanded menu borders */
        [data-baseweb="popover"] {
            border: 2px solid #4a5568 !important;
            background-color: #2d3748 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        }

        [data-baseweb="select"] [data-baseweb="menu"] {
            border: 2px solid #4a5568 !important;
            background-color: #2d3748 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        }
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .kpi-title {
        font-size: 0.9rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
    }
    .kpi-subtitle {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }

    /* Increase font size of collapsible section headings */
    .stExpander > details > summary {
        font-size: 2.34rem !important;
        font-weight: 700 !important;
        padding: 15px 10px !important;
        line-height: 1.4 !important;
        min-height: 60px !important;
        display: flex !important;
        align-items: center !important;
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
        background-color: #ffffff;
        color: #262730;
    }

    .main {
        background-color: #ffffff;
    }

    .main-title {
        color: #1f77b4;
    }

    .main-subtitle {
        color: #666;
    }

    .kpi-card {
        background-color: #f0f2f6;
        color: #262730;
        border-left-color: #1f77b4;
    }
    .kpi-title {
        color: #333;
    }
    .kpi-value {
        color: #1f77b4;
    }
    .kpi-subtitle {
        color: #666;
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
            background-color: #21262d !important;
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
        background-color: #21262d !important;
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

    /* Override any remaining dark text */
    * {
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
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .main {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Override all Streamlit containers */
    .block-container {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Override sidebar if present */
    .css-1d391kg {
        background-color: #f8f9fa !important;
    }

    /* Override all text elements */
    .main-title {
        color: #1f77b4 !important;
    }

    .main-subtitle {
        color: #666 !important;
    }

    /* Override all headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1f77b4 !important;
    }

    /* Override all text */
    p, div, span, label {
        color: #262730 !important;
    }

    /* Override Streamlit specific text elements */
    .stMarkdown, .stText {
        color: #262730 !important;
    }

    /* Override buttons - comprehensive targeting */
    .stButton > button {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target all button variants */
    button[data-testid="baseButton-secondary"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    button[data-testid="baseButton-primary"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Override all button elements */
    button {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Button hover states */
    button:hover {
        background-color: #f8f9fa !important;
        color: #262730 !important;
    }

    /* Specific targeting for filter control buttons */
    button[title*="Reset"], button[title*="Clear"], button[title*="Share"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Override metrics */
    .stMetric {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .kpi-card {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        border-left-color: #1f77b4 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 8px !important;
    }
    .kpi-title {
        color: #333 !important;
    }
    .kpi-value {
        color: #1f77b4 !important;
    }
    .kpi-subtitle {
        color: #666 !important;
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
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
    }

    /* Light mode expanded menu borders */
    [data-baseweb="popover"] {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        color: #262730 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    [data-baseweb="select"] [data-baseweb="menu"] {
        border: 2px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        color: #262730 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Override all dropdown and input elements */
    [data-testid="stSelectbox"] div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    [data-testid="stMultiSelect"] div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Override expander headers and content - comprehensive */
    .streamlit-expander {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Target expander content area when expanded */
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Target expander details content */
    details[open] > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Target all expander-related divs */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 8px !important;
        margin-bottom: 16px !important;
    }

    [data-testid="stExpander"] > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Target expander summary and content */
    .stExpander > details {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .stExpander > details > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Override any dark backgrounds in expanded sections - force normal state */
    .stExpander details[open] {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .stExpander details[open] > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .stExpander details[open] > div > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    /* Force all nested content in expanded sections */
    .stExpander details[open] * {
        background-color: transparent !important;
        color: #262730 !important;
    }

    /* But maintain white backgrounds for main containers */
    .stExpander details[open] > div,
    .stExpander details[open] > div > div {
        background-color: #ffffff !important;
    }

    /* Override dataframe */
    .stDataFrame {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }

    /* Override plots and charts within expanded sections */
    .stPlotlyChart {
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
        color: #262730 !important;
    }

    .plotly .modebar-btn {
        background-color: #ffffff !important;
        color: #262730 !important;
        fill: #262730 !important;
    }

    .plotly .modebar-btn:hover {
        background-color: #f8f9fa !important;
        color: #262730 !important;
        fill: #262730 !important;
    }

    /* Fix chart axis labels and text */
    .plotly text {
        fill: #262730 !important;
        color: #262730 !important;
    }

    .plotly .xtick text, .plotly .ytick text {
        fill: #262730 !important;
        color: #262730 !important;
    }

    /* Fix chart legends */
    .plotly .legend {
        background-color: #ffffff !important;
        color: #262730 !important;
    }

    .plotly .legend text {
        fill: #262730 !important;
        color: #262730 !important;
    }

    /* Fix chart titles and axis titles */
    .plotly .gtitle text, .plotly .xtitle text, .plotly .ytitle text {
        fill: #262730 !important;
        color: #262730 !important;
    }

    /* Fix hover labels */
    .plotly .hoverlayer .hovertext {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Fix SVG text elements in charts */
    .js-plotly-plot svg text {
        fill: #262730 !important;
    }

    .js-plotly-plot .plotly-notifier {
        color: #262730 !important;
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

    /* Override info boxes and alerts within expanded sections */
    .stInfo {
        background-color: #e7f3ff !important;
        color: #262730 !important;
        border: 1px solid #b3d9ff !important;
        border-radius: 6px !important;
    }

    .stSuccess {
        background-color: #d1f2d1 !important;
        color: #262730 !important;
        border: 1px solid #a3d977 !important;
        border-radius: 6px !important;
    }

    .stWarning {
        background-color: #fff3cd !important;
        color: #262730 !important;
        border: 1px solid #ffd93d !important;
        border-radius: 6px !important;
    }

    .stError {
        background-color: #f8d7da !important;
        color: #262730 !important;
        border: 1px solid #f1a7aa !important;
        border-radius: 6px !important;
    }

    /* Override code blocks */
    .stCode {
        background-color: #f8f9fa !important;
        color: #262730 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }

    pre {
        background-color: #f8f9fa !important;
        color: #262730 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 6px !important;
    }

    code {
        background-color: #f8f9fa !important;
        color: #262730 !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 3px !important;
        padding: 2px 4px !important;
    }

    /* Override all containers and content areas */
    .element-container {
        background-color: transparent !important;
    }

    /* Override any remaining dark text */
    * {
        color: #262730 !important;
    }

    /* Additional button targeting - catch all variations */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    button[kind="primary"] {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target buttons by their container classes */
    .stButton button {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Target buttons by their parent element */
    div[data-testid="column"] button {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Ensure button text is visible */
    .stButton button span {
        color: #262730 !important;
    }

    /* Re-apply specific colors for important elements */
    .main-title, h1 {
        color: #1f77b4 !important;
    }

    .kpi-value {
        color: #1f77b4 !important;
    }
    </style>
    """


def apply_theme_css():
    """Apply theme-specific CSS based on current theme mode."""
    theme_mode = st.session_state.get("theme_mode", "auto")

    if theme_mode == "auto":
        st.markdown(get_auto_mode_css(), unsafe_allow_html=True)
    elif theme_mode == "dark":
        st.markdown(get_dark_mode_css(), unsafe_allow_html=True)
    else:  # light mode
        st.markdown(get_light_mode_css(), unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables for styling."""
    # Initialize theme state with auto-detection
    if "theme_initialized" not in st.session_state:
        st.session_state.theme_initialized = True
        st.session_state.theme_mode = "auto"  # Options: "light", "dark", "auto"


def initialize_streamlit_config():
    """Initialize Streamlit configuration."""
    st.set_page_config(
        page_title="LLM Inference Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
