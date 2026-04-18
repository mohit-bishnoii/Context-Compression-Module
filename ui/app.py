# ui/app.py
#
# Side-by-side Gradio UI: Baseline Agent (left) vs CCM Agent (right)
# Mirrors the Arena AI layout shown in the design reference.
#
# Run from project root:
#   python ui/app.py
#
# Requirements:
#   pip install gradio>=4.42.0
#   GROQ_API_KEY must be set in .env

import sys
import os
import json
import time
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

# ── Lazy agent imports (avoids heavy load on startup) ────────────
_baseline_agent = None
_ccm_agent = None


def get_agents():
    global _baseline_agent, _ccm_agent
    if _baseline_agent is None:
        from travel_agent.baseline_agent import BaselineAgent
        from travel_agent.agent import CCMAgent
        _baseline_agent = BaselineAgent()
        _ccm_agent = CCMAgent(use_reranking=False)
    return _baseline_agent, _ccm_agent


# ── State helpers ────────────────────────────────────────────────

def _fmt_memory(mem_state: dict) -> str:
    """Format CCM memory state for display panel."""
    lines = []

    wm = mem_state.get("working_memory", "")
    if wm and wm.strip() != "[NO USER PREFERENCES CAPTURED YET]":
        lines.append("### 🧠 Working Memory")
        lines.append("```")
        lines.append(wm.strip())
        lines.append("```")

    ep = mem_state.get("episodic_entries", [])
    if ep:
        lines.append(f"\n### 📼 Episodic Memory ({len(ep)} entries)")
        for e in ep[:4]:
            txt = e.get("text", "")[:120]
            lines.append(f"- {txt}…")

    sem = mem_state.get("semantic_entries", [])
    if sem:
        lines.append(f"\n### 🗃️ Semantic Memory ({len(sem)} entries)")
        for s in sem[:4]:
            tool = s.get("tool_name", "tool")
            txt = s.get("text", "")[:100]
            lines.append(f"- **[{tool}]** {txt}…")

    tok = mem_state.get("token_metrics", {})
    if tok:
        lines.append("\n### 📊 Token Metrics")
        lines.append(f"- Baseline tokens: **{tok.get('baseline_tokens_used', 0):,}**")
        lines.append(f"- CCM tokens: **{tok.get('ccm_tokens_used', 0):,}**")
        ratio = tok.get('compression_ratio', 0)
        saved = tok.get('tokens_saved', 0)
        lines.append(f"- Compression ratio: **{ratio}x**")
        lines.append(f"- Tokens saved: **{saved:,}**")

    return "\n".join(lines) if lines else "_No memory captured yet._"


def _build_metrics_table(baseline_history, ccm_history, b_tokens, c_tokens):
    """Build a comparison metrics markdown table."""
    rows = []
    n = max(len(baseline_history), len(ccm_history))

    for i in range(n):
        b_tok = b_tokens[i] if i < len(b_tokens) else 0
        c_tok = c_tokens[i] if i < len(c_tokens) else 0
        ratio = f"{b_tok / max(c_tok, 1):.1f}x" if b_tok and c_tok else "—"
        rows.append(f"| Turn {i+1} | {b_tok:,} | {c_tok:,} | {ratio} |")

    if not rows:
        return "_Send a message to see metrics._"

    header = "| Turn | Baseline Tokens | CCM Tokens | Reduction |\n|------|----------------|------------|-----------|"
    return header + "\n" + "\n".join(rows)


# ── Core chat function ───────────────────────────────────────────

def chat(
    user_message: str,
    baseline_history: list,
    ccm_history: list,
    b_token_log: list,
    c_token_log: list,
    session_active: bool,
):
    if not user_message.strip():
        yield (
            baseline_history, ccm_history,
            b_token_log, c_token_log,
            "", "",
            _build_metrics_table(baseline_history, ccm_history, b_token_log, c_token_log),
            session_active,
        )
        return

    baseline_agent, ccm_agent = get_agents()

    # --- NEW: Use dictionaries instead of lists ---
    baseline_history = baseline_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ _Thinking…_"}
    ]
    ccm_history = ccm_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ _Thinking…_"}
    ]


    yield (
        baseline_history, ccm_history,
        b_token_log, c_token_log,
        "", "",
        _build_metrics_table(baseline_history, ccm_history, b_token_log, c_token_log),
        session_active,
    )

    # ── Run Baseline ─────────────────────────────────────────────
    # ... (Keep your agent logic as is) ...
    b_result = baseline_agent.chat(user_message)
    b_response = b_result["response"]
    
    # --- NEW: Update the last message content ---
    baseline_history[-1]["content"] = b_response
    b_token_log = b_token_log + [b_result["tokens_in_context"]]

    yield (
        baseline_history, ccm_history,
        b_token_log, c_token_log,
        "", "",
        _build_metrics_table(baseline_history, ccm_history, b_token_log, c_token_log),
        session_active,
    )

    # ── Run CCM ───────────────────────────────────────────────────
    # ... (Keep your agent logic as is) ...
    c_result = ccm_agent.chat(user_message)
    c_response = c_result["response"]
    
    # --- NEW: Update the last message content ---
    ccm_history[-1]["content"] = c_response
    c_token_log = c_token_log + [c_result["tokens_in_context"]]
    
    mem_md = _fmt_memory(c_result.get("memory_state", {}))
    metrics_md = _build_metrics_table(baseline_history, ccm_history, b_token_log, c_token_log)

    yield (
        baseline_history, ccm_history,
        b_token_log, c_token_log,
        mem_md, metrics_md,
        metrics_md,
        True,
    )


def reset_conversation():
    """Reset both agents and all state."""
    try:
        baseline_agent, ccm_agent = get_agents()
        baseline_agent.reset()
        ccm_agent.reset()
    except Exception:
        pass
    return [], [], [], [], "", "_No memory captured yet._", "_Send a message to see metrics._", False


# ── CSS ──────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Base & fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg-primary:    #0d0f14;
    --bg-secondary:  #13161e;
    --bg-card:       #181c26;
    --bg-input:      #1e2330;
    --border:        #2a2f3d;
    --border-accent: #3d4461;
    --text-primary:  #e8eaf0;
    --text-secondary:#8b90a8;
    --text-muted:    #555c78;
    --accent-left:   #5b8dee;
    --accent-right:  #22d3a0;
    --accent-warn:   #f5a623;
    --accent-err:    #ef4444;
    --tag-bg:        rgba(91,141,238,0.12);
    --radius:        12px;
    --radius-sm:     8px;
    --shadow:        0 4px 24px rgba(0,0,0,0.4);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Header ───────────────────────────────────────────── */
.ccm-header {
    text-align: center;
    padding: 28px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.ccm-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
    background: linear-gradient(135deg, var(--accent-left), var(--accent-right));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ccm-header p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin: 0;
    font-weight: 300;
}

/* ── Column headers ───────────────────────────────────── */
.col-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 0 0;
    margin-bottom: 0;
}
.col-header-left  { border-bottom: 2px solid var(--accent-left); }
.col-header-right { border-bottom: 2px solid var(--accent-right); }

.col-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    letter-spacing: 0.5px;
}
.badge-left  { background: rgba(91,141,238,0.18); color: var(--accent-left); }
.badge-right { background: rgba(34,211,160,0.18); color: var(--accent-right); }

.col-model-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
}

/* ── Chatbot ──────────────────────────────────────────── */
.chatbot-left  { border-top: none !important; border-radius: 0 0 var(--radius) var(--radius) !important; }
.chatbot-right { border-top: none !important; border-radius: 0 0 var(--radius) var(--radius) !important; }

.chatbot-left .message.user,
.chatbot-right .message.user {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

.chatbot-left .message.bot {
    background: rgba(91,141,238,0.08) !important;
    border: 1px solid rgba(91,141,238,0.2) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

.chatbot-right .message.bot {
    background: rgba(34,211,160,0.08) !important;
    border: 1px solid rgba(34,211,160,0.2) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Input row ────────────────────────────────────────── */
.input-row {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px 8px !important;
}

.input-row textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
    resize: none !important;
}
.input-row textarea::placeholder { color: var(--text-muted) !important; }
.input-row textarea:focus { outline: none !important; box-shadow: none !important; }

/* ── Buttons ──────────────────────────────────────────── */
.btn-send {
    background: linear-gradient(135deg, var(--accent-left), #7b6cff) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.btn-send:hover { opacity: 0.85 !important; }

.btn-reset {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 8px 18px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.btn-reset:hover {
    border-color: var(--accent-err) !important;
    color: var(--accent-err) !important;
}

/* ── Tabs ─────────────────────────────────────────────── */
.tabs-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    transition: all 0.2s !important;
}
.tab-nav button:hover { color: var(--text-primary) !important; }
.tab-nav button.selected {
    color: var(--accent-right) !important;
    border-bottom-color: var(--accent-right) !important;
    background: rgba(34,211,160,0.06) !important;
}

/* ── Markdown panels ──────────────────────────────────── */
.memory-panel, .metrics-panel {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.83rem !important;
    line-height: 1.7 !important;
    padding: 16px !important;
}

.memory-panel code, .metrics-panel code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    color: var(--accent-right) !important;
}

.memory-panel pre, .metrics-panel pre {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px !important;
    overflow-x: auto !important;
}

.memory-panel table, .metrics-panel table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.82rem !important;
}
.memory-panel th, .metrics-panel th {
    background: var(--bg-input) !important;
    color: var(--text-secondary) !important;
    padding: 8px 12px !important;
    text-align: left !important;
    border-bottom: 1px solid var(--border) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.3px !important;
}
.memory-panel td, .metrics-panel td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
.memory-panel tr:last-child td, .metrics-panel tr:last-child td {
    border-bottom: none !important;
}
.memory-panel tr:hover td, .metrics-panel tr:hover td {
    background: rgba(255,255,255,0.03) !important;
}

/* ── Token bars ───────────────────────────────────────── */
.token-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
}
.token-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    min-width: 60px;
}
.token-bar {
    height: 6px;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.token-bar-b { background: var(--accent-left); }
.token-bar-c { background: var(--accent-right); }

/* ── Status pill ──────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--text-muted);
    padding: 4px 0;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
}
.status-dot.active { background: var(--accent-right); box-shadow: 0 0 6px var(--accent-right); }

/* ── Scrollbars ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 99px; }

/* ── Gradio overrides ─────────────────────────────────── */
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gr-group { border: none !important; background: transparent !important; }
label.svelte-1b6s6s { color: var(--text-secondary) !important; font-size: 0.78rem !important; }
"""

# ── HTML helpers ─────────────────────────────────────────────────

HEADER_HTML = """
<div class="ccm-header">
  <h1>Context Compression Module — Arena</h1>
  <p>Baseline (naive full-history) vs CCM (compressed memory) — same input, side by side</p>
</div>
"""

def col_header_html(label: str, model: str, side: str) -> str:
    badge_cls = "badge-left" if side == "left" else "badge-right"
    header_cls = f"col-header-{side}"
    return f"""
    <div class="col-header {header_cls}">
      <span class="col-badge {badge_cls}">{label}</span>
      <span class="col-model-name">{model}</span>
    </div>
    """

BASELINE_HEADER = col_header_html("BASELINE", "llama-3.1-8b-instant · no compression", "left")
CCM_HEADER = col_header_html("CCM", "llama-3.1-8b-instant · context compression module", "right")


# ── Build UI ─────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="CCM Arena — Context Compression") as demo:

        # ── Hidden state ─────────────────────────────────────────
        baseline_history = gr.State([])
        ccm_history      = gr.State([])
        b_token_log      = gr.State([])
        c_token_log      = gr.State([])
        session_active   = gr.State(False)

        # ── Header ───────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Main chat columns — SIDE BY SIDE ─────────────────────
        with gr.Row():

            # LEFT Column — Baseline
            with gr.Column(scale=1):
                gr.HTML(BASELINE_HEADER)
                baseline_chat = gr.Chatbot(
                label="",
                height=480,
                elem_classes=["chatbot-left"],
                show_label=False,
                # type="messages", # Use messages format
                render_markdown=True,
            )

            # RIGHT Column — CCM
            with gr.Column(scale=1):
                gr.HTML(CCM_HEADER)
                ccm_chat = gr.Chatbot(
                label="",
                height=480,
                elem_classes=["chatbot-right"],
                show_label=False,
                # type="messages", # Use messages format
                render_markdown=True,
            )

        # ── Input row ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                user_input = gr.Textbox(
                    placeholder="Ask your travel agent anything…",
                    lines=1,
                    max_lines=3,
                    show_label=False,
                    elem_classes=["input-row"],
                    container=False,
                )
            with gr.Column(scale=1, min_width=100):
                send_btn = gr.Button("Send →", elem_classes=["btn-send"])
            with gr.Column(scale=1, min_width=100):
                reset_btn = gr.Button("↺ Reset", elem_classes=["btn-reset"])

        # ── Bottom panel — tabs ───────────────────────────────────
        with gr.Row():
            with gr.Column():
                with gr.Tabs(elem_classes=["tabs-panel"]):
                    with gr.Tab("🧠 CCM Memory State"):
                        memory_panel = gr.Markdown(
                            value="_No memory captured yet._",
                            elem_classes=["memory-panel"],
                        )
                    with gr.Tab("📊 Token Comparison"):
                        metrics_panel = gr.Markdown(
                            value="_Send a message to see token metrics._",
                            elem_classes=["metrics-panel"],
                        )

                    # Tab 3 — Quick Guide
                    with gr.Tab("📖 How It Works"):
                        gr.Markdown(
                            """
### Context Compression Module (CCM)

The CCM sits between the user and the LLM. Instead of sending the full
conversation history every turn, it maintains **three memory tiers**:

| Tier | Storage | Access | Purpose |
|------|---------|--------|---------|
| **Working Memory** | JSON file on disk | Always injected | Critical facts (allergies, budget caps) — never forgotten |
| **Episodic Memory** | ChromaDB vectors | Retrieved by similarity | Conversation summaries — what happened in past turns |
| **Semantic Memory** | ChromaDB vectors | Retrieved by similarity | Compressed tool results — research archive |

### Why this beats naive context stuffing

| | Baseline | CCM |
|---|---|---|
| Turn 5 tokens | ~2,400 | ~800 |
| Turn 15 tokens | ~7,800 → 💥 crash | ~1,400 ✅ |
| Allergy remembered at turn 15 | ❌ Often forgotten | ✅ Always present |
| Budget tracking | ❌ Lost in noise | ✅ Tracked in working memory |
| Stale context (Bali→Switzerland) | ❌ Bleeds through | ✅ Marked stale & removed |

### Try these test scenarios

1. **Forgotten Allergy** — Start with *"I'm severely allergic to shellfish, budget $3000"*, chat for several turns, then ask *"find restaurants near Tsukiji"*
2. **Budget Tracking** — Set a $2500 budget, "book" flights for $800 then hotels for $750, then ask for an Amalfi hotel
3. **The Pivot** — Plan a Bali trip, then say *"scratch Bali, let's do Switzerland instead"*, then ask for a summary
                            """,
                            elem_classes=["memory-panel"],
                        )

        # ── Suggested prompts ─────────────────────────────────────
        gr.HTML("""
        <div style="padding: 12px 0 0; display:flex; gap:8px; flex-wrap:wrap;">
          <span style="font-size:0.75rem; color:#555c78; align-self:center;">Try:</span>
        </div>
        """)

        with gr.Row():
            p1 = gr.Button(
                "🇯🇵 I'm planning Tokyo trip, $3000 budget, severely allergic to shellfish",
                size="sm",
            )
            p2 = gr.Button("✈️ Find flights from New York to Tokyo in June", size="sm")
            p3 = gr.Button("🏨 Find hotels in Shinjuku area", size="sm")

        with gr.Row():
            p4 = gr.Button("🍽️ Find dinner restaurants near Tsukiji fish market", size="sm")
            p5 = gr.Button("🔄 Scratch Tokyo, let's do Paris instead — forget everything", size="sm")
            p6 = gr.Button("📋 Summarize my trip plan so far", size="sm")

        # ── Wire up events ────────────────────────────────────────

        send_outputs = [
            baseline_chat, ccm_chat,
            b_token_log, c_token_log,
            memory_panel, metrics_panel,
            metrics_panel,
            session_active,
        ]

        send_inputs = [
            user_input,
            baseline_history, ccm_history,
            b_token_log, c_token_log,
            session_active,
        ]

        def submit_and_clear(msg, bh, ch, bt, ct, sa):
            results = None
            for step in chat(msg, bh, ch, bt, ct, sa):
                results = step
                yield step
            # Clear input after done
            if results:
                final = list(results)
                yield tuple(final)

        # Textbox submit
        user_input.submit(
            fn=submit_and_clear,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(
            fn=lambda: "",
            outputs=user_input,
        )

        # Button send
        send_btn.click(
            fn=submit_and_clear,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(
            fn=lambda: "",
            outputs=user_input,
        )

        # Reset
        reset_btn.click(
            fn=reset_conversation,
            outputs=[
                baseline_chat, ccm_chat,
                b_token_log, c_token_log,
                memory_panel, metrics_panel,
                metrics_panel,
                session_active,
            ],
        )

        # Suggested prompt buttons — fill input
        p1.click(fn=lambda: "I'm planning a 5-day trip to Tokyo and Kyoto. Budget is $3,000 total. I am severely allergic to shellfish — this is a serious medical allergy.", outputs=user_input)
        p2.click(fn=lambda: "Find flights from New York to Tokyo in June", outputs=user_input)
        p3.click(fn=lambda: "What hotels are available in Shinjuku area Tokyo?", outputs=user_input)
        p4.click(fn=lambda: "Find me the best dinner spots in the Tsukiji area", outputs=user_input)
        p5.click(fn=lambda: "Actually, scratch Tokyo entirely. Forget everything about Tokyo. Let's do Paris instead — I want culture, museums, not fish markets.", outputs=user_input)
        p6.click(fn=lambda: "Give me a complete summary of my trip plan so far", outputs=user_input)

    return demo


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("CCM Arena UI")
    print("="*55)

    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set. Add it to your .env file.")
        print("   Get a free key at https://console.groq.com")
        sys.exit(1)

    print("✅ GROQ_API_KEY found")
    print("🚀 Starting Gradio server…")
    print("   Open: http://localhost:7860\n")

    demo = build_ui()
    demo.queue(max_size=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,  # Move it here
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Sora"),
        ), # Move it here
    )
def get_agents():
    global _baseline_agent, _ccm_agent
    if _baseline_agent is None:
        print("🤖 Initializing agents (this may take a minute on first run)...")
        from travel_agent.baseline_agent import BaselineAgent
        from travel_agent.agent import CCMAgent
        
        print("📡 Loading Baseline Agent...")
        _baseline_agent = BaselineAgent()
        
        print("🧠 Loading CCM Agent & ChromaDB...")
        _ccm_agent = CCMAgent(use_reranking=False)
        print("✅ Agents ready!")
    return _baseline_agent, _ccm_agent