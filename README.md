# Context Compression Module

A drop-in Python module that reduces LLM context size by compressing tool results and managing agent memory across three tiers.

## Overview

The Context Compression Module (CCM) sits between any user interface and LLM agent, handling:

- **Fact extraction** from user messages → WorkingMemory
- **Tool result compression** (~600 tokens → ~80 tokens)
- **Stale context detection** and cleanup
- **Memory retrieval** from EpisodicMemory and SemanticMemory
- **Context assembly** into ~1400-token packets

## Architecture

Three memory tiers:

| Tier | Purpose | Storage |
|------|---------|---------|
| WorkingMemory | Current facts, constraints, user preferences | In-memory dict |
| EpisodicMemory | Summarized conversation turns (every 4 turns) | ChromaDB |
| SemanticMemory | Compressed tool results | ChromaDB |

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

## Usage

```python
from ccm.ccm_core import ContextCompressionModule

ccm = ContextCompressionModule()

# Before LLM call
context = ccm.process_user_message(user_message)

# After tool execution
compressed = ccm.process_tool_result(tool_name, raw_result, query)

# After agent response
ccm.process_agent_response(user_message, agent_response, tool_calls)
```

## Components

- `extractor.py` - LLM-based fact extraction
- `compressor.py` - Tool result compression
- `stale_detector.py` - Detects stale/overridden context
- `retriever.py` - ChromaDB retrieval with optional re-ranking
- `assembler.py` - Builds context packets

## Demo

```bash
python -m ui.app
```

## Testing

```bash
python evaluation/run_evaluation.py
```