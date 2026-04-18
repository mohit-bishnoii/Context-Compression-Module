# ui/app.py
#
# Side-by-side Gradio UI: Baseline Agent (left) vs CCM Agent (right)
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
import gc
import time
import shutil
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

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
MEMORY_PATH = "data/working_memory.json"

# ── One-time reset (mirrors reset_all_storage from test.py) ──────

def reset_all_storage():
    """
    Wipe ALL persisted state once at the start of a new conversation.
    Mirrors reset_all_storage() in test.py exactly.
    """
    os.makedirs("data", exist_ok=True)

    # Reset budget tool state
    try:
        from travel_agent.tools import reset_budget
        reset_budget()
    except Exception as exc:
        print(f"[Reset] Budget reset warning: {exc}")

    # Blank out working memory
    blank = {
        "facts": {"critical": [], "important": [], "contextual": []},
        "decisions":  [],
        "cancelled":  [],
        "turn_count": 0,
        "conversation_id": "",
        "last_updated": "",
    }
    with open(MEMORY_PATH, "w") as f:
        json.dump(blank, f, indent=2)

    # Delete ChromaDB collections via a fresh client
    if os.path.exists(CHROMA_PATH):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            for col in client.list_collections():
                try:
                    name = col.name if hasattr(col, "name") else str(col)
                    client.delete_collection(name)
                    print(f"[Reset] Deleted collection: {name}")
                except Exception as exc:
                    print(f"[Reset] Could not delete collection: {exc}")
            del client
        except Exception as exc:
            print(f"[Reset] ChromaDB client error: {exc}")

    gc.collect()
    time.sleep(0.5)
    print("[Reset] Storage reset complete\n")


# ── Lazy agent imports ────────────────────────────────────────────
_baseline_agent = None
_ccm_agent = None


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


# ── Core chat function ────────────────────────────────────────────

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
            session_active,
        )
        return

    baseline_agent, ccm_agent = get_agents()

    # ── One-time reset on first turn ─────────────────────────────
    # If session_active is False, this is the first message.
    # Reset all storage once before the agents process anything.
    if not session_active:
        print("[App] First turn detected — resetting all storage...")
        reset_all_storage()
        baseline_agent.reset()
        ccm_agent.reset()

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
        session_active,
    )

    # ── Run Baseline ─────────────────────────────────────────────
    b_result = baseline_agent.chat(user_message)
    b_response = b_result["response"]
    baseline_history[-1]["content"] = b_response
    b_token_log = b_token_log + [b_result["tokens_in_context"]]

    yield (
        baseline_history, ccm_history,
        b_token_log, c_token_log,
        session_active,
    )

    # ── Run CCM ──────────────────────────────────────────────────
    c_result = ccm_agent.chat(user_message)
    c_response = c_result["response"]
    ccm_history[-1]["content"] = c_response
    c_token_log = c_token_log + [c_result["tokens_in_context"]]

    yield (
        baseline_history, ccm_history,
        b_token_log, c_token_log,
        True,   # mark session as active after first turn completes
    )


def reset_conversation():
    """Reset both agents and all state (manual reset button)."""
    try:
        baseline_agent, ccm_agent = get_agents()
        reset_all_storage()
        baseline_agent.reset()
        ccm_agent.reset()
    except Exception:
        pass
    return [], [], [], [], False


# ── CSS ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
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

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gr-group { border: none !important; background: transparent !important; }
label.svelte-1b6s6s { color: var(--text-secondary) !important; font-size: 0.78rem !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 99px; }
"""

# ── HTML helpers ──────────────────────────────────────────────────

HEADER_HTML = """
<div class="ccm-header">
  <h1>Context Compression Module — Arena</h1>
  <p>Baseline (naive full-history) vs CCM (compressed memory) — same input, side by side</p>
</div>
"""

def col_header_html(label: str, model: str, side: str) -> str:
    badge_cls  = "badge-left" if side == "left" else "badge-right"
    header_cls = f"col-header-{side}"
    return f"""
    <div class="col-header {header_cls}">
      <span class="col-badge {badge_cls}">{label}</span>
      <span class="col-model-name">{model}</span>
    </div>
    """

BASELINE_HEADER = col_header_html("BASELINE", "llama-3.3-70b · no compression", "left")
CCM_HEADER      = col_header_html("CCM",      "llama-3.3-70b · context compression module", "right")


# ── Build UI ──────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="CCM Arena — Context Compression") as demo:

        # ── Hidden state ──────────────────────────────────────────
        baseline_history = gr.State([])
        ccm_history      = gr.State([])
        b_token_log      = gr.State([])
        c_token_log      = gr.State([])
        session_active   = gr.State(False)

        # ── Header ────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Main chat columns ─────────────────────────────────────
        with gr.Row():

            # LEFT — Baseline
            with gr.Column(scale=1):
                gr.HTML(BASELINE_HEADER)
                baseline_chat = gr.Chatbot(
                    label="",
                    height=480,
                    elem_classes=["chatbot-left"],
                    show_label=False,
                    render_markdown=True,
                )

            # RIGHT — CCM
            with gr.Column(scale=1):
                gr.HTML(CCM_HEADER)
                ccm_chat = gr.Chatbot(
                    label="",
                    height=480,
                    elem_classes=["chatbot-right"],
                    show_label=False,
                    render_markdown=True,
                )

        # ── Input row ─────────────────────────────────────────────
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
            session_active,
        ]

        send_inputs = [
            user_input,
            baseline_history, ccm_history,
            b_token_log, c_token_log,
            session_active,
        ]

        def submit_and_clear(msg, bh, ch, bt, ct, sa):
            for step in chat(msg, bh, ch, bt, ct, sa):
                yield step

        user_input.submit(
            fn=submit_and_clear,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=user_input)

        send_btn.click(
            fn=submit_and_clear,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=user_input)

        reset_btn.click(
            fn=reset_conversation,
            outputs=[
                baseline_chat, ccm_chat,
                b_token_log, c_token_log,
                session_active,
            ],
        )

        # Suggested prompt buttons
        p1.click(fn=lambda: "I'm planning a 5-day trip to Tokyo and Kyoto. Budget is $3,000 total. I am severely allergic to shellfish — this is a serious medical allergy.", outputs=user_input)
        p2.click(fn=lambda: "Find flights from New York to Tokyo in June", outputs=user_input)
        p3.click(fn=lambda: "What hotels are available in Shinjuku area Tokyo?", outputs=user_input)
        p4.click(fn=lambda: "Find me the best dinner spots in the Tsukiji area", outputs=user_input)
        p5.click(fn=lambda: "Actually, scratch Tokyo entirely. Forget everything about Tokyo. Let's do Paris instead — I want culture, museums, not fish markets.", outputs=user_input)
        p6.click(fn=lambda: "Give me a complete summary of my trip plan so far", outputs=user_input)

    return demo


# ── Entry point ───────────────────────────────────────────────────

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
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Sora"),
        ),
    )