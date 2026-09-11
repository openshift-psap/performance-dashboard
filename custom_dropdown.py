"""Inject hover tooltips onto Streamlit selectbox options for profile dropdowns."""

import json

import streamlit.components.v1 as components

from profile_config import PROFILE_DETAILS


def inject_profile_tooltips() -> None:
    """Inject JS that adds hover tooltips to all ISL/OSL profile selectbox options.

    Uses st.components.v1.html() to run JavaScript that accesses the parent
    Streamlit document via window.parent.document. A MutationObserver watches
    for [role="option"] elements and attaches tooltip behavior on hover.
    """
    tooltip_map = {}
    for key, details in PROFILE_DETAILS.items():
        lines = [details["name"]]
        lines.append(f"Input: {details['prompt_tokens']}")
        lines.append(f"Output: {details['output_tokens']}")
        if details.get("samples"):
            lines.append(f"Samples: {details['samples']}")
        if details.get("turns"):
            lines.append(f"Turns: {details['turns']}")
        if details.get("prefix_tokens"):
            lines.append(f"Prefix Token: {details['prefix_tokens']}")
        if details.get("prefix_count"):
            lines.append(f"Prefix Count: {details['prefix_count']}")
        if details.get("description"):
            lines.append(details["description"])
        tooltip_map[key] = "\n".join(lines)

    for key in list(tooltip_map):
        for variant in _generate_display_variants(key):
            if variant not in tooltip_map:
                tooltip_map[variant] = tooltip_map[key]

    js_code = f"""
    <script>
    (function() {{
        const doc = window.parent.document;
        if (doc.__profileTooltipInjected) return;
        doc.__profileTooltipInjected = true;

        const tooltips = {json.dumps(tooltip_map)};

        const extractKey = (text) => {{
            text = text.trim();
            if (tooltips[text]) return text;
            const m = text.match(/\\(([^)]+)\\)\\s*$/);
            if (m) {{
                let pair = m[1];
                pair = pair.replace(/(\\d+)k/gi, (_, n) => String(Number(n) * 1000));
                if (tooltips[pair]) return pair;
            }}
            let norm = text.replace(/(\\d+)k/gi, (_, n) => String(Number(n) * 1000));
            if (tooltips[norm]) return norm;
            return null;
        }};

        const tip = doc.createElement('div');
        tip.id = 'profile-tooltip';
        tip.style.cssText = `
            position: fixed; background: #ffffff; color: #1a1a1a;
            padding: 10px 14px; border-radius: 6px; font-size: 0.8rem;
            font-family: "Source Sans Pro", sans-serif;
            max-width: 340px; z-index: 100001; pointer-events: none;
            opacity: 0; transition: opacity 0.15s; white-space: pre-line;
            line-height: 1.5; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border: 1px solid #e0e0e0;
        `;
        doc.body.appendChild(tip);

        const show = (el, key) => {{
            tip.textContent = tooltips[key];
            const rect = el.getBoundingClientRect();
            let left = rect.right + 12;
            if (left + 350 > doc.documentElement.clientWidth) {{
                left = rect.left - 350 - 12;
                if (left < 8) left = 8;
            }}
            tip.style.left = left + 'px';
            tip.style.top = Math.max(8, rect.top) + 'px';
            tip.style.opacity = '1';
        }};

        const hide = () => {{ tip.style.opacity = '0'; }};

        const bind = (opt) => {{
            if (opt.dataset.ttBound) return;
            const key = extractKey(opt.textContent);
            if (!key) return;
            opt.dataset.ttBound = '1';
            opt.addEventListener('mouseenter', () => show(opt, key));
            opt.addEventListener('mouseleave', hide);
        }};

        new MutationObserver(() => {{
            doc.querySelectorAll('[role="option"]').forEach(bind);

            // Tag ISL/OSL tooltip icons with profile-info-icon class
            doc.querySelectorAll('[data-testid="stWidgetLabel"]').forEach(label => {{
                const text = label.textContent || '';
                if (text.includes('ISL') || text.includes('OSL') || text.includes('Sequence Length')) {{
                    const container = label.closest('[data-testid="stSelectbox"], [data-testid="stMultiSelect"]');
                    if (!container) return;
                    const icon = container.querySelector('.stTooltipIcon');
                    if (icon && !icon.classList.contains('profile-info-icon')) {{
                        icon.classList.add('profile-info-icon');
                    }}
                }}
            }});
        }}).observe(doc.body, {{ childList: true, subtree: true }});
    }})();
    </script>
    """
    components.html(js_code, height=0)


def _generate_display_variants(token_pair: str) -> list[str]:
    """Generate common display name variants for a token pair like '1000/1000'."""
    variants = []
    parts = token_pair.split("/")
    if len(parts) != 2:
        return variants

    def to_k(val: str) -> str:
        try:
            n = int(val)
            if n >= 1000 and n % 1000 == 0:
                return f"{n // 1000}k"
        except ValueError:
            pass
        return val

    k_input = to_k(parts[0])
    k_output = to_k(parts[1])
    k_pair = f"{k_input}/{k_output}"
    if k_pair != token_pair:
        variants.append(k_pair)
    variants.append(f"({k_pair})")

    profile_names = {
        "1000/1000": [
            "Profile A: Balanced (1k/1k)",
        ],
        "512/2048": [
            "Profile B: Variable Workload (512/2k)",
            "(512/2k)",
        ],
        "2048/128": [
            "Profile C: Prompt-Heavy (2k/128)",
            "(2k/128)",
        ],
        "8000/1000": [
            "Profile D: Long Context (8k/1k)",
            "(8k/1k)",
        ],
        "100000/1000": [
            "Profile E: Extreme Context (100k/1k)",
            "(100k/1k)",
        ],
        "8000/800": [
            "Profile F: Heavy Heterogeneous (8k/800)",
            "(8k/800)",
        ],
        "128/128": [
            "Profile G: Multi-turn (128/128)",
            "(128/128)",
            "Multi-turn",
        ],
    }
    for name in profile_names.get(token_pair, []):
        if name not in variants:
            variants.append(name)

    return variants
