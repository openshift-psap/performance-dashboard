"""Inject hover tooltips onto Streamlit selectbox options for profile dropdowns."""

import json

import streamlit.components.v1 as components

PROFILE_DETAILS = {
    "1000/1000": {
        "name": "Balanced Profile",
        "prompt_tokens": "1000",
        "output_tokens": "1000",
    },
    "512/2048": {
        "name": "Heterogeneous",
        "prompt_tokens": "512 (stdev=128, min=1, max=1024)",
        "output_tokens": "2048 (stdev=512, min=1, max=4096)",
    },
    "2048/128": {
        "name": "Short Prefill-Heavy",
        "prompt_tokens": "2048",
        "output_tokens": "128",
    },
    "8000/1000": {
        "name": "Prefill-Heavy",
        "prompt_tokens": "8000",
        "output_tokens": "1000",
        "samples": "50",
    },
    "100000/1000": {
        "name": "Long Context Prefill-Heavy",
        "prompt_tokens": "100000",
        "output_tokens": "1000",
        "samples": "10",
    },
    "8000/800": {
        "name": "Heavy Heterogeneous",
        "prompt_tokens": "8000 (stdev=8500, min=50, max=30000)",
        "output_tokens": "800 (stdev=1500, min=20, max=8000)",
        "description": "Simulates realistic chat patterns with large prompts and smaller outputs. Samples 450 seconds of traffic.",
    },
    "128/128": {
        "name": "Multi-turn Profile",
        "prompt_tokens": "128",
        "output_tokens": "128",
        "turns": "5",
        "prefix_tokens": "512",
        "prefix_count": "10,000",
        "description": "Multi-turn conversation benchmark with 5 turns and 512-token prefix.",
        "aliases": ["Multi-turn"],
    },
}


def get_profile_details(profile_name: str) -> dict:
    """Get profile details by name, normalizing k-notation (e.g. 1k/1k -> 1000/1000)."""
    if profile_name in PROFILE_DETAILS:
        return PROFILE_DETAILS[profile_name]

    if not profile_name:
        return {}

    if "(" in profile_name and ")" in profile_name:
        start = profile_name.rfind("(") + 1
        end = profile_name.rfind(")")
        token_pair = profile_name[start:end].strip()
    else:
        token_pair = profile_name

    parts = token_pair.split("/")
    normalized = []
    for part in parts:
        if "k" in part.lower():
            num = float(part.lower().replace("k", "")) * 1000
            normalized.append(str(int(num)))
        else:
            normalized.append(part)

    key = "/".join(normalized)
    return PROFILE_DETAILS.get(key, {})


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
    """Generate display name variants for a token pair like '1000/1000'.

    Auto-generates k-notation variants (e.g. 1000/1000 -> 1k/1k, (1k/1k)).
    Additional aliases (e.g. 'Multi-turn') come from the 'aliases' field in PROFILE_DETAILS.
    """
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

    # The dropdown may show rounded k-notation (e.g. 2048 as "2k") which the JS
    # decodes as 2000. Add that decoded form so the lookup still hits.
    def k_decoded(val: str) -> str:
        try:
            n = int(val)
            if n >= 1000:
                return str((n // 1000) * 1000)
        except ValueError:
            pass
        return val

    decoded = f"{k_decoded(parts[0])}/{k_decoded(parts[1])}"
    if decoded != token_pair and decoded not in variants:
        variants.append(decoded)

    for alias in PROFILE_DETAILS.get(token_pair, {}).get("aliases", []):
        if alias not in variants:
            variants.append(alias)

    return variants
