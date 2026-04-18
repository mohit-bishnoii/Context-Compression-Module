# README.md


# Context Compression Module (CCM) — Travel Agent

> Make every token count. A plug-and-play context compression system
> that teaches AI agents what to remember, what to forget, and what
> to re-inject.

---

## Overview

Modern AI agents break when conversations get long. They forget,
lose track and recommend things the user already cancelled.
— not because the model is bad, but because the
context window fills up with noise.

**This project builds a Context Compression Module (CCM)** that sits
between the agent and its growing history, keeping only what matters.

It is demonstrated through a **Multi-City Travel Planning Agent** that
handles flights, hotels, restaurants, budgets, and user preferences
across 20+ turn conversations.

---

## How Memory Works

![Tiered Memory Flow](images/memory.png)

Three memory tiers feed the LLM — instead of dumping everything:

| Tier | Storage | What it holds | Always sent? |
|---|---|---|---|
| 🔴 Working Memory | JSON file | Allergies, budget, dates | ✅ Yes |
| 🟡 Episodic Memory | ChromaDB | Summarized decisions | RAG only |
| 🔵 Archived Memory | ChromaDB | Compressed tool results | RAG only |

---

## How the Pipeline Works

![CCM Director Pipeline](images/pipeline.png)

Every user message goes through 4 steps before the LLM sees it:

```
User Message
     │
     ▼
1. MEMORY EXTRACTOR     → pull facts (allergy, budget, destination)
     │
     ▼
2. STALE DETECTOR       → mark cancelled info as invalid
     │
     ▼
3. RAG RETRIEVAL        → fetch only relevant memories
     │
     ▼
4. CONTEXT ASSEMBLER    → pack into 2,000 token budget
     │
     ▼
  LLM Agent (Llama 3.1 8B)
```

**Result:** Baseline sends ~11,200 tokens. CCM sends ~1,400 tokens.
Same task. Better answers.

---

## Prerequisites

```bash
Python 3.9+
A free Groq API key → https://console.groq.com
```

```bash
git clone https://github.com/mohit-bishnoii/Context-Compression-Module.git
cd Context-Compression-Module
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=gsk_your_key_here
```

Verify everything works:
```bash
python test_setup.py
```

---

## Run It

```bash
# Chat UI
python ui/app.py

# Evaluation (baseline vs CCM comparison)
python evaluation/run_evaluation.py
```

---

## Key Results

| Metric | Baseline | CCM |
|---|---|---|
| Tokens at turn 16 | ~11,200 | ~1,400 |
| Allergy remembered | ❌ | ✅ |
| Budget tracked | ❌ | ✅ |
| Stale info removed | ❌ | ✅ |
| Response latency | 8.2s | 2.1s |

---

## Stack

| What | Tool |
|---|---|
| LLM | Llama 3.1 8B via Groq (free) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (local) |
| UI | Gradio |
| Training needed | None |

---

> The deliverable is the **CCM module** — not the chat interface.
> Drop it in front of any agent that suffers from long conversations.
```
