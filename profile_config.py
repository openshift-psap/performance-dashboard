"""Shared guidellm profile configurations for all dashboards."""

PROFILE_DETAILS = {
    "1000/1000": {
        "name": "Profile 1 - Balanced",
        "prompt_tokens": "1000",
        "output_tokens": "1000",
        "description": "Balanced workload with equal input and output token counts.",
    },
    "512/2048": {
        "name": "Profile 2 - Variable Distribution",
        "prompt_tokens": "512 (stdev=128, min=1, max=1024)",
        "output_tokens": "2048 (stdev=512, min=1, max=4096)",
        "description": "Variable token distributions simulating diverse real-world prompts and responses. Tests performance with unpredictable token patterns.",
    },
    "2048/128": {
        "name": "Profile 3 - Prompt-Heavy",
        "prompt_tokens": "2048",
        "output_tokens": "128",
        "description": "Prompt-heavy workload with long context queries and short responses (16:1 ratio). Tests context processing efficiency.",
    },
    "8000/1000": {
        "name": "Profile 4 - Long Context",
        "prompt_tokens": "8000",
        "output_tokens": "1000",
        "samples": "50",
        "description": "Extended context with 8K prompt tokens and 1K output tokens. Samples 50 requests. Tests performance on document-based tasks.",
    },
    "100000/1000": {
        "name": "Profile 5 - Extreme Context",
        "prompt_tokens": "100000",
        "output_tokens": "1000",
        "samples": "10",
        "description": "Extreme long-context workload with 100K+ prompt tokens (e.g., full documents). Samples 10 requests. Tests maximum context window.",
    },
    "8000/800": {
        "name": "Profile 7 - Heterogeneous",
        "prompt_tokens": "8000 (stdev=8500, min=50, max=30000)",
        "output_tokens": "800 (stdev=1500, min=20, max=8000)",
        "description": "Long context with variable token distributions. Simulates realistic chat patterns with large prompts and smaller outputs. Samples 450 seconds of traffic.",
    },
    "128/128": {
        "name": "Profile 6 - Multi-turn",
        "prompt_tokens": "1000",
        "output_tokens": "1000",
        "turns": "5",
        "prefix_tokens": "512",
        "description": "Multi-turn conversation benchmark with 5 turns and 512-token prefix (10 copies). Tests context management in iterative conversations.",
    },
}


def extract_token_pair_from_profile_name(profile_name: str) -> str:
    """Extract token pair (e.g., '512/2048') from profile display name.

    Examples:
        "Profile B: Variable Workload (512/2k)" -> "512/2048"
        "1k/1k" -> "1000/1000"
        "8k/1k" -> "8000/1000"
        "512/2k" -> "512/2048"
    """
    if not profile_name:
        return ""

    # Extract token counts from parentheses, e.g., (512/2k)
    if "(" in profile_name and ")" in profile_name:
        start = profile_name.rfind("(") + 1
        end = profile_name.rfind(")")
        token_pair = profile_name[start:end].strip()
    else:
        token_pair = profile_name

    # Normalize 'k' notation: 1k -> 1000, 2k -> 2048, etc.
    parts = token_pair.split("/")
    normalized = []
    for part in parts:
        if "k" in part.lower():
            # Convert 1k to 1000, 2k to 2048, etc.
            num = float(part.lower().replace("k", "")) * 1000
            normalized.append(str(int(num)))
        else:
            normalized.append(part)

    return "/".join(normalized)


def get_profile_details(profile_name: str) -> dict:
    """Get profile details from profile name, extracting token pair if needed."""
    # First try direct lookup
    if profile_name in PROFILE_DETAILS:
        return PROFILE_DETAILS[profile_name]

    # Try to extract token pair and look up
    token_pair = extract_token_pair_from_profile_name(profile_name)
    if token_pair in PROFILE_DETAILS:
        return PROFILE_DETAILS[token_pair]

    return {}


def get_profile_tooltip(profile_key: str) -> str:
    """Return detailed tooltip text for a guidellm profile."""
    details = get_profile_details(profile_key)
    if not details:
        return ""

    lines = [
        f"**{details['name']}**",
        f"• Input tokens: {details['prompt_tokens']}",
        f"• Output tokens: {details['output_tokens']}",
    ]

    if details.get("samples"):
        lines.append(f"• Samples: {details['samples']}")
    if details.get("turns"):
        lines.append(f"• Turns: {details['turns']}")
    if details.get("prefix_tokens"):
        lines.append(f"• Prefix tokens: {details['prefix_tokens']}")

    lines.append(f"• {details['description']}")
    return "\n".join(lines)
