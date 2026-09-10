"""Custom dropdown component with hover tooltips for profile selection."""

import streamlit as st
import streamlit.components.v1 as components
from profile_config import PROFILE_DETAILS


def render_profile_dropdown(label: str, options: list, key: str = None) -> str:
    """Render a custom dropdown with hover tooltips for each profile.

    Args:
        label: Dropdown label
        options: List of profile keys (e.g., ["1000/1000", "512/2048"])
        key: Streamlit key for state management

    Returns:
        Selected profile key, or None if nothing selected
    """

    # Build tooltip HTML for each option
    options_html = ""
    for opt in options:
        details = PROFILE_DETAILS.get(opt, {})
        if not details:
            # For options not in profile config, just show the key
            tooltip = opt
        else:
            # Build tooltip with profile details
            tooltip_lines = [
                f"<strong>{details['name']}</strong>",
                f"Input: {details['prompt_tokens']}",
                f"Output: {details['output_tokens']}",
            ]
            if details.get("samples"):
                tooltip_lines.append(f"Samples: {details['samples']}")
            if details.get("turns"):
                tooltip_lines.append(f"Turns: {details['turns']}")
            if details.get("prefix_tokens"):
                tooltip_lines.append(f"Prefix: {details['prefix_tokens']}")
            tooltip_lines.append(f"<em>{details['description']}</em>")
            tooltip = "<br>".join(tooltip_lines)

        options_html += f'''
            <div class="dropdown-item" data-value="{opt}" data-tooltip="{tooltip}">
                {opt}
            </div>
        '''

    # Generate unique ID for this dropdown
    dropdown_id = f"profile_dropdown_{key}" if key else "profile_dropdown"

    html = f'''
    <div class="custom-profile-dropdown" id="{dropdown_id}">
        <div class="dropdown-label">{label}</div>
        <button class="dropdown-toggle" id="{dropdown_id}_toggle">
            Select profile ▼
        </button>
        <div class="dropdown-menu" id="{dropdown_id}_menu">
            {options_html}
        </div>
    </div>

    <style>
    .custom-profile-dropdown {{
        position: relative;
        display: inline-block;
        width: 100%;
        margin: 10px 0;
    }}

    .dropdown-label {{
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #262730;
    }}

    .dropdown-toggle {{
        width: 100%;
        padding: 10px 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 0.875rem;
        text-align: left;
        transition: border-color 0.2s;
    }}

    .dropdown-toggle:hover {{
        border-color: #999;
    }}

    .dropdown-toggle.active {{
        border-color: #0066cc;
        background: #f0f7ff;
    }}

    .dropdown-menu {{
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border: 1px solid #ccc;
        border-top: none;
        border-radius: 0 0 4px 4px;
        max-height: 300px;
        overflow-y: auto;
        z-index: 1000;
        display: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    .dropdown-menu.open {{
        display: block;
    }}

    .dropdown-item {{
        padding: 12px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
        position: relative;
        transition: background-color 0.15s;
    }}

    .dropdown-item:last-child {{
        border-bottom: none;
    }}

    .dropdown-item:hover {{
        background-color: #f5f5f5;
    }}

    /* Tooltip on hover */
    .dropdown-item[data-tooltip]::after {{
        content: attr(data-tooltip);
        position: absolute;
        left: 100%;
        top: 50%;
        transform: translateY(-50%);
        margin-left: 10px;
        background: #1f2937;
        color: #fff;
        font-size: 0.75rem;
        padding: 8px 12px;
        border-radius: 4px;
        white-space: nowrap;
        z-index: 1001;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s;
        white-space: normal;
        max-width: 250px;
    }}

    .dropdown-item:hover[data-tooltip]::after {{
        opacity: 1;
    }}

    /* Arrow for tooltip */
    .dropdown-item[data-tooltip]::before {{
        content: '';
        position: absolute;
        left: calc(100% - 5px);
        top: 50%;
        transform: translateY(-50%);
        width: 0;
        height: 0;
        border-left: 5px solid #1f2937;
        border-top: 5px solid transparent;
        border-bottom: 5px solid transparent;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s;
    }}

    .dropdown-item:hover[data-tooltip]::before {{
        opacity: 1;
    }}
    </style>

    <script>
    (function() {{
        const dropdownId = "{dropdown_id}";
        const toggle = document.getElementById(dropdownId + "_toggle");
        const menu = document.getElementById(dropdownId + "_menu");
        const items = menu.querySelectorAll(".dropdown-item");
        let selectedValue = null;

        // Toggle menu
        toggle.addEventListener("click", function() {{
            menu.classList.toggle("open");
            toggle.classList.toggle("active");
        }});

        // Close menu when clicking outside
        document.addEventListener("click", function(e) {{
            if (!e.target.closest("#" + dropdownId)) {{
                menu.classList.remove("open");
                toggle.classList.remove("active");
            }}
        }});

        // Handle item selection
        items.forEach(function(item) {{
            item.addEventListener("click", function() {{
                selectedValue = item.dataset.value;
                toggle.textContent = selectedValue + " ✓";
                menu.classList.remove("open");
                toggle.classList.remove("active");

                // Send data to Streamlit
                window.parent.postMessage({{
                    type: "streamlit:setComponentValue",
                    sessionID: "{{key}}",
                    value: selectedValue
                }}, "*");
            }});
        }});
    }})();
    </script>
    '''

    return components.html(html, height=400)
