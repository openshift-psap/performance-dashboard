"""IntelliConfig: Guided Optimization Wizard.

A 4-step wizard that helps users select model, accelerator, workload profile,
and deployment platform, then generates ready-to-use deployment configurations.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Option catalogs (hardcoded for now; will be replaced with dynamic mapping)
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "description": "120B parameter open-source model",
        "icon": "🤖",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-0528",
        "name": "DeepSeek R1",
        "description": "DeepSeek R1 reasoning model",
        "icon": "🧠",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama 3.3 70B",
        "description": "Meta Llama 3.3, 70B parameters",
        "icon": "🦙",
    },
    {
        "id": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        "name": "Llama 3.3 70B FP8",
        "description": "RedHat AI optimized, FP8 quantization",
        "icon": "🦙",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B",
        "description": "8B parameter model, fast inference",
        "icon": "🦙",
    },
]

ACCELERATORS = [
    {
        "id": "H200",
        "name": "NVIDIA H200",
        "description": "Hopper architecture, 141 GB HBM3e",
        "icon": "🟩",
    },
    {
        "id": "B200",
        "name": "NVIDIA B200",
        "description": "Blackwell architecture, next-gen GPU",
        "icon": "🟩",
    },
    {
        "id": "B300",
        "name": "NVIDIA B300",
        "description": "Blackwell Ultra, high-memory GPU",
        "icon": "🟩",
    },
    {
        "id": "MI300X",
        "name": "AMD MI300X",
        "description": "192 GB HBM3, ROCm stack",
        "icon": "⬛",
    },
    {
        "id": "TPU",
        "name": "Google TPU",
        "description": "Tensor Processing Unit, Google Cloud",
        "icon": "🔵",
    },
]

WORKLOAD_PROFILES = [
    {
        "id": "latency",
        "name": "Latency-Sensitive",
        "description": "Minimize TTFT and ITL for real-time apps",
        "icon": "⏱",
    },
    {
        "id": "throughput",
        "name": "Throughput-Optimized",
        "description": "Maximize tokens / second for batch jobs",
        "icon": "⏱",
    },
    {
        "id": "balanced",
        "name": "Balanced",
        "description": "Trade-off between latency and throughput",
        "icon": "⏱",
    },
]

PLATFORMS = [
    {
        "id": "openshift",
        "name": "Red Hat OpenShift",
        "description": "Deploy via InferenceService CRD on OpenShift AI",
        "icon": "🖥",
    },
    {
        "id": "bare_metal",
        "name": "Bare-metal",
        "description": "Direct CLI deployment on bare-metal servers",
        "icon": "🖥",
    },
]

STEP_LABELS = {1: "Context", 2: "Recommendations", 3: "Simulator", 4: "Export"}

# ---------------------------------------------------------------------------
# Template config parameters keyed by (workload_profile, accelerator)
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "max_batch_size": 64,
    "tensor_parallel_size": 2,
    "max_num_seqs": 256,
    "gpu_memory_utilization": 0.92,
    "extra_flags": ["--enable-speculative-decoding"],
}

_PARAM_OVERRIDES = {
    "latency": {
        "max_batch_size": 32,
        "max_num_seqs": 128,
        "extra_flags": ["--enable-chunked-prefill"],
    },
    "throughput": {
        "max_batch_size": 128,
        "max_num_seqs": 512,
        "extra_flags": ["--enable-speculative-decoding"],
    },
    "balanced": {
        "max_batch_size": 64,
        "max_num_seqs": 256,
        "extra_flags": ["--enable-speculative-decoding"],
    },
}

_ACCEL_TP = {
    "H200": 4,
    "B200": 8,
    "B300": 1,
    "MI300X": 4,
    "TPU": 4,
}


def _get_config_params(workload_id: str, accel_id: str) -> dict:
    params = dict(_DEFAULT_PARAMS)
    overrides = _PARAM_OVERRIDES.get(workload_id, {})
    params.update(overrides)
    params["tensor_parallel_size"] = _ACCEL_TP.get(accel_id, 2)
    return params


# ---------------------------------------------------------------------------
# Config template generators
# ---------------------------------------------------------------------------


def _generate_bare_metal_cli(model_id: str, params: dict) -> str:
    extra = " \\\n  ".join(params["extra_flags"])
    return (
        f"vllm serve {model_id} \\\n"
        f"  --max-batch-size {params['max_batch_size']} \\\n"
        f"  --tensor-parallel-size {params['tensor_parallel_size']} \\\n"
        f"  --max-num-seqs {params['max_num_seqs']} \\\n"
        f"  --gpu-memory-utilization {params['gpu_memory_utilization']} \\\n"
        f"  {extra} \\\n"
        f"  --host 0.0.0.0 \\\n"
        f"  --port 8000"
    )


def _generate_openshift_yaml(model_id: str, params: dict) -> str:
    slug = model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    args_lines = "\n".join(
        [
            f"          - --max-batch-size={params['max_batch_size']}",
            f"          - --tensor-parallel-size={params['tensor_parallel_size']}",
            f"          - --max-num-seqs={params['max_num_seqs']}",
            f"          - --gpu-memory-utilization={params['gpu_memory_utilization']}",
        ]
        + [f"          - {f}" for f in params["extra_flags"]]
    )
    return (
        f"apiVersion: serving.kserve.io/v1beta1\n"
        f"kind: InferenceService\n"
        f"metadata:\n"
        f"  name: {slug}-optimized\n"
        f"  namespace: psap-inference\n"
        f"spec:\n"
        f"  predictor:\n"
        f"    model:\n"
        f"      modelFormat:\n"
        f"        name: vLLM\n"
        f"      runtime: vllm-runtime\n"
        f"      args:\n"
        f"{args_lines}"
    )


def _generate_helm_values(model_id: str, params: dict) -> str:
    extra = "\n".join(f"    - {f}" for f in params["extra_flags"])
    return (
        f"# Helm values for xKS / AKS deployment\n"
        f"model:\n"
        f"  name: {model_id}\n"
        f"  runtime: vllm\n"
        f"server:\n"
        f"  replicas: 1\n"
        f"  args:\n"
        f"    maxBatchSize: {params['max_batch_size']}\n"
        f"    tensorParallelSize: {params['tensor_parallel_size']}\n"
        f"    maxNumSeqs: {params['max_num_seqs']}\n"
        f"    gpuMemoryUtilization: {params['gpu_memory_utilization']}\n"
        f"  extraArgs:\n"
        f"{extra}\n"
        f"resources:\n"
        f"  gpu:\n"
        f"    count: {params['tensor_parallel_size']}\n"
        f"  memory: 80Gi"
    )


# ---------------------------------------------------------------------------
# Benchmark run counting
# ---------------------------------------------------------------------------


def _count_benchmark_runs(df, model_id: str) -> int:
    """Count rows matching a model id (substring match on the 'model' column)."""
    if df is None or "model" not in df.columns:
        return 0
    short_name = model_id.split("/")[-1]
    mask = df["model"].str.contains(short_name, case=False, na=False)
    return int(mask.sum())


def _count_accel_runs(df, accel_id: str) -> int:
    if df is None or "accelerator" not in df.columns:
        return 0
    mask = df["accelerator"] == accel_id
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Card rendering helpers
# ---------------------------------------------------------------------------


def _render_selection_card(
    item: dict,
    selected_id: str | None,
    session_key: str,
    benchmark_count: int | None = None,
    disabled: bool = False,
):
    """Render a single clickable card.

    Uses st.container(key=...) so we can target the wrapper via the
    CSS class ``.st-key-<key>`` that Streamlit adds automatically.
    Inside the container: card HTML + an invisible button covering it.
    """
    is_selected = item["id"] == selected_id

    badge_html = ""
    if benchmark_count is not None:
        badge_cls = "ic-badge-muted" if benchmark_count == 0 else "ic-badge"
        badge_html = (
            f'<span class="{badge_cls}">'
            f"{benchmark_count} benchmark runs available</span>"
        )

    desc = item["description"]
    if disabled:
        desc = '<span class="ic-disabled-note">Not supported for selected model</span>'

    radio_cls = "ic-card-radio-selected" if is_selected else "ic-card-radio"

    state_cls = ""
    if is_selected:
        state_cls = "ic-state-selected"
    elif disabled:
        state_cls = "ic-state-disabled"

    content_html = (
        f'<div class="ic-card-content {state_cls}">'
        f'<div class="ic-card-header">'
        f'<span class="ic-card-icon">{item["icon"]}</span>'
        f'<span class="ic-card-name">{item["name"]}</span>'
        f'<span class="{radio_cls}"></span>'
        f"</div>"
        f'<div class="ic-card-desc">{desc}</div>'
        f"{badge_html}"
        f"</div>"
    )

    safe_id = item["id"].replace("/", "-").replace(".", "-")
    container_key = f"ic_{session_key}_{safe_id}"
    btn_key = f"ic_btn_{session_key}_{safe_id}"

    card = st.container(border=True, key=container_key)
    with card:
        st.markdown(content_html, unsafe_allow_html=True)
        if not disabled and st.button(
            item["name"],
            key=btn_key,
            use_container_width=True,
        ):
            st.session_state[session_key] = item["id"]
            st.rerun()


def _render_card_grid(
    title: str,
    items: list[dict],
    session_key: str,
    df=None,
    count_fn=None,
    disabled_ids: set | None = None,
):
    """Render a titled responsive grid of selection cards."""
    st.markdown(f"**{title}**")
    disabled_ids = disabled_ids or set()
    n_cols = min(len(items), 4)
    grid_key = f"ic_grid_{session_key}"
    with st.container(key=grid_key):
        cols = st.columns(n_cols)
        for idx, item in enumerate(items):
            with cols[idx % n_cols]:
                bench_count = count_fn(df, item["id"]) if count_fn else None
                _render_selection_card(
                    item=item,
                    selected_id=st.session_state.get(session_key),
                    session_key=session_key,
                    benchmark_count=bench_count,
                    disabled=item["id"] in disabled_ids,
                )


# ---------------------------------------------------------------------------
# Step renderers
# ---------------------------------------------------------------------------


def _render_step_context(df):
    st.markdown(
        "Choose your target model, accelerator, workload profile, and deployment "
        "platform."
    )

    _render_card_grid(
        "Model", MODELS, "ic_model", df=df, count_fn=_count_benchmark_runs
    )
    _render_card_grid(
        "Accelerator",
        ACCELERATORS,
        "ic_accelerator",
        df=df,
        count_fn=_count_accel_runs,
    )
    _render_card_grid("Workload Profile", WORKLOAD_PROFILES, "ic_workload")
    _render_card_grid("Target Platform", PLATFORMS, "ic_platform")


def _render_step_recommendations():
    st.markdown("---")
    st.info(
        "**Recommendations** — coming soon.\n\n"
        "This step will surface optimal configurations based on your benchmark data.",
        icon="🔮",
    )


def _render_step_simulator():
    st.markdown("---")
    st.info(
        "**Simulator** — coming soon.\n\n"
        "This step will let you model performance under different configurations.",
        icon="🧪",
    )


def _render_step_export():
    model_id = st.session_state.get("ic_model") or MODELS[0]["id"]
    accel_id = st.session_state.get("ic_accelerator") or ACCELERATORS[0]["id"]
    workload_id = st.session_state.get("ic_workload") or WORKLOAD_PROFILES[0]["id"]
    platform_id = st.session_state.get("ic_platform") or PLATFORMS[0]["id"]

    model_name = next((m["name"] for m in MODELS if m["id"] == model_id), model_id)
    accel_name = next(
        (a["name"] for a in ACCELERATORS if a["id"] == accel_id), accel_id
    )
    workload_name = next(
        (w["name"] for w in WORKLOAD_PROFILES if w["id"] == workload_id), workload_id
    )
    platform_name = next(
        (p["name"] for p in PLATFORMS if p["id"] == platform_id), platform_id
    )
    params = _get_config_params(workload_id, accel_id)

    summary_html = (
        '<div class="ic-export-summary">'
        f'<span class="ic-export-chip">Model: <b>{model_name}</b></span>'
        f'<span class="ic-export-chip">Accelerator: <b>{accel_name}</b></span>'
        f'<span class="ic-export-chip">Workload: <b>{workload_name}</b></span>'
        f'<span class="ic-export-chip">Platform: <b>{platform_name}</b></span>'
        "</div>"
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    st.markdown(
        "Copy the generated configuration below. "
        "The default tab matches your selected platform."
    )

    openshift_cfg = _generate_openshift_yaml(model_id, params)
    helm_cfg = _generate_helm_values(model_id, params)
    cli_cfg = _generate_bare_metal_cli(model_id, params)

    with st.container(key="ic_export_tabs"):
        tab_labels = [
            "🟥  OpenShift YAML",
            "⎈  Helm Values",
            "▶  Bare Metal CLI",
        ]
        tabs = st.tabs(tab_labels)

        configs = [
            ("yaml", openshift_cfg),
            ("yaml", helm_cfg),
            ("bash", cli_cfg),
        ]
        for i, tab in enumerate(tabs):
            with tab:
                lang, cfg_text = configs[i]
                st.code(cfg_text, language=lang)

    with st.container(key="ic_export_actions"):
        btn_cols = st.columns([1, 1, 2])
        with btn_cols[0]:
            st.button(
                "🔗  Share Config Link",
                key="ic_share",
                use_container_width=True,
            )
        with btn_cols[1]:
            st.button(
                "📊  View Benchmark Evidence",
                key="ic_evidence",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_intelliconfig_section(df):
    """Render the IntelliConfig guided optimization wizard."""
    for key, default in [
        ("ic_step", 1),
        ("ic_model", None),
        ("ic_accelerator", None),
        ("ic_workload", None),
        ("ic_platform", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(
        '<h2 style="margin-bottom:0.2em;">IntelliConfig: Guided Optimization</h2>',
        unsafe_allow_html=True,
    )

    step = st.session_state.ic_step

    # --- Layout: step sidebar + content ---
    step_col, content_col = st.columns([1, 4], gap="large")

    with step_col:
        for num, label in STEP_LABELS.items():
            is_active = num == step
            cls = "ic-step-active" if is_active else "ic-step"
            marker = (
                f'<span class="ic-step-num-active">{num}</span>'
                if is_active
                else f'<span class="ic-step-num">{num}</span>'
            )
            key = f"ic_stepnav_{num}"
            with st.container(key=key):
                if st.button(label, key=f"ic_stepbtn_{num}", use_container_width=True):
                    st.session_state.ic_step = num
                    st.rerun()
                st.markdown(
                    f'<div class="{cls}">{marker} {label}</div>',
                    unsafe_allow_html=True,
                )

    with content_col:
        if step == 1:
            _render_step_context(df)
        elif step == 2:
            _render_step_recommendations()
        elif step == 3:
            _render_step_simulator()
        elif step == 4:
            _render_step_export()

    # --- Navigation buttons (right-aligned) ---
    st.markdown("---")
    nav_cols = st.columns([2, 1, 1, 1])
    with nav_cols[1]:
        if step > 1 and st.button("Previous", key="ic_prev", use_container_width=True):
            st.session_state.ic_step = step - 1
            st.rerun()
    with nav_cols[2]:
        if step < 4:
            if st.button(
                "Next", key="ic_next", type="primary", use_container_width=True
            ):
                st.session_state.ic_step = step + 1
                st.rerun()
        else:
            if st.button(
                "Apply Configuration",
                key="ic_apply",
                type="primary",
                use_container_width=True,
            ):
                st.success("Configuration applied! (placeholder)")
    with nav_cols[3]:
        with st.container(key="ic_cancel_wrap"):
            if st.button("Cancel", key="ic_cancel", use_container_width=True):
                st.session_state.ic_step = 1
                st.session_state.ic_model = None
                st.session_state.ic_accelerator = None
                st.session_state.ic_workload = None
                st.session_state.ic_platform = None
                st.rerun()
