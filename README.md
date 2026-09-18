# Adaptive Universal Prompt Optimizer & APEX Quality Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://modelcontextprotocol.io/)

An intelligent, multi-stage prompt-engineering framework and **Model Context Protocol (MCP)** server built with **FastMCP**. It dynamically transforms raw, underspecified user queries into structured, production-ready, and verifiable prompt templates with continuous **APEX** quality scoring.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dynamic Archetypes & Domain Classification](#dynamic-archetypes--domain-classification)
- [APEX Evaluation Framework](#apex-evaluation-framework)
- [Prerequisites & Setup](#prerequisites--setup)
- [Usage Guide](#usage-guide)
  - [1. Standalone CLI Fast Direct Mode](#1-standalone-cli-fast-direct-mode)
  - [2. Interactive Mode (Adaptive Q&A)](#2-interactive-mode-adaptive-qa)
  - [3. Verification Test Suite](#3-verification-test-suite)
  - [4. MCP Client Integration (Claude Desktop / Cursor)](#4-mcp-client-integration-claude-desktop--cursor)
- [Complete MCP Tool Reference](#complete-mcp-tool-reference)
  - [Fast-Path & 1-Click Tools](#fast-path--1-click-tools)
  - [11-Stage Granular Pipeline Tools](#11-stage-granular-pipeline-tools)
  - [Session & Diagnostic Tools](#session--diagnostic-tools)
- [Pipeline Execution Flow](#pipeline-execution-flow)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Most prompt generators either rely on rigid, hardcoded domain templates or produce generic instructions without considering user depth, specific components, or evaluation guardrails.

This **Adaptive Prompt Optimizer** overcomes these limitations by:
1. **Classifying tasks dynamically** across software engineering, academic research, conceptual learning (pedagogy), information synthesis, strategic decision-making, and general expert consultation.
2. **Interactively clarifying user intent** using low-friction questions to capture preferred depth (Overview, Balanced, Deep Dive), required elements (analogies, use cases, implementation, trade-offs), and target audience.
3. **Evaluating prompt quality continuously** using a multi-dimensional **APEX** (Assessment, Performance, Execution, Excellence) scoring rubric with an automated refinement loop (OPRO).
4. **Providing dual operation modes**: A standalone command-line interface (CLI) with zero external API dependencies, and a standard **Model Context Protocol (MCP)** server over `stdio` for Claude Desktop, Cursor, and Claude Code.

---

## Key Features

- **Universal Task Adaptation**: Dynamically maps any prompt to specialized expert personas, custom structured blueprints, and quality requirements.
- **Adaptive Clarification (Non-Hardbound)**: Offers 2–3 low-friction questions (Depth, Elements, Audience) with recommended defaults for rapid customization.
- **Continuous, Multi-Dimensional APEX Evaluator**:
  - Evaluates both System Prompt and User Prompt against raw query fidelity.
  - Scores across Assessment, Performance, Execution, and Excellence ($0$ to $100$ scale).
  - Automatically identifies flaws and triggers OPRO (Optimization by PROmpting) iterative refinement until quality threshold ($\ge 85.0$) is met.
- **Dual Operating Modes**:
  - **Standalone Terminal CLI**: Run directly from PowerShell, Terminal, or Command Prompt with instant output.
  - **Standard MCP Server**: Exposes complete 1-click tools and granular 11-stage pipeline tools via FastMCP over `stdio`.

---

## System Architecture

```
                       Raw User Query (e.g., "tell me about DSA")
                                          │
                                          ▼
                         [Dynamic Intent & Domain Classifier]
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
   [Software Eng.]               [Academic Research]              [STEM Pedagogy]
   [Info Synthesis]             [Strategic Analysis]           [General Consultation]
                                          │
                                          ▼
                         [Adaptive Clarification Gate]
                          ├── Depth: Overview | Balanced | Deep Dive
                          ├── Elements: Analogies | Examples | Code | Pitfalls
                          └── Audience: Students | Developers | Executives
                                          │
                                          ▼
                             [Prompt Synthesizer Engine]
                          ├── System Prompt (Expert Persona & Policy)
                          └── User Prompt Template (Scaffolding & Directives)
                                          │
                                          ▼
                         [APEX Quality Evaluation Gate]
                          ├── Assessment (0-100)
                          ├── Performance (0-100)
                          ├── Execution (0-100)
                          └── Excellence (0-100)
                                          │
                        ┌─────────────────┴─────────────────┐
                        │ Average Score >= 85.0?            │
                        ├─────────────────┬─────────────────┤
                       YES               NO (and iters < 3)
                        │                                   │
                        ▼                                   ▼
          [Finalized Production Artifact]       [OPRO Refinement Loop]
```

---

## Dynamic Archetypes & Domain Classification

The classifier automatically detects task context and assigns specialized personas and structural archetypes:

| Task Category | Domain | Expert Persona Assigned | Key Structural Focus |
| :--- | :--- | :--- | :--- |
| **Software Product Engineering** | Computer Science & Software | Principal Software Architect & Full-Stack Product Engineer | Architecture, modular blueprint, API/data models, edge cases, verification strategy |
| **Academic Research Analysis** | Research & Academia | Senior Academic Researcher & Literature Methodologist | Executive summary, root causes, research gaps, testable hypotheses, validation framework |
| **Technical & Conceptual Explanation** | Education & Pedagogy | Master Computer Science & STEM Educator (Feynman Technique) | Intuitive hook & analogy, first principles, walkthrough & complexity trade-offs, misconceptions, verification problem |
| **Information Synthesis** | Technology Intelligence | Senior Technology Intelligence Analyst & Research Specialist | Emerging developments, technical mechanics, industry impact, limitations, verification sources |
| **Strategic Analysis** | Strategic Consulting | Principal Strategic Consultant & Decision Analyst | Context & objectives, comparative analysis, critical trade-offs, actionable recommendations |
| **General Expert Consultation** | General Inquiry | World-Class Domain Specialist & Rigorous Analytical Consultant | Objective breakdown, structured deliverable, constraints & edge cases, actionable summary |

---

## APEX Evaluation Framework

The APEX rubric evaluates prompt quality across four distinct pillars ($0$ to $100$ continuous scoring):

| Pillar | Focus | Evaluation Criteria |
| :--- | :--- | :--- |
| **A — Assessment** | Task Clarity & Scope | Query keyword retention ratio, explicit objective framing, and appropriate length/density (60–350 words). |
| **P — Performance** | Authority & Persona | Senior/expert persona definition, domain-specific methodology (e.g. Feynman, first-principles), and audience calibration. |
| **E — Execution** | Structure & Flow | Numbered progressive scaffolding ($\ge 4$ steps), structural section headers, and explicit markdown output formatting. |
| **X — eXcellence** | Guardrails & Verification | Anti-hallucination constraints, edge-case and failure mode analysis, and self-verification / testing criteria. |

$$\text{Composite APEX Score} = \frac{\text{Assessment} + \text{Performance} + \text{Execution} + \text{Excellence}}{4}$$

- **Pass Threshold**: $\ge 85.0$ / $100$
- **Refinement**: If score is $< 85.0$, the optimizer performs automated iterative OPRO refinement (up to 3 iterations).

---

## Prerequisites & Setup

### Requirements
* **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12)
* Windows PowerShell, macOS Terminal, or Linux Bash

### Installation

1. Navigate to the project directory:
   ```bash
   cd Prompt_Optimizer
   ```

2. Activate the existing virtual environment or create a new one:
   ```powershell
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   source .venv/bin/activate
   ```

3. If setting up in a clean environment, install dependencies:
   ```bash
   pip install mcp pydantic
   ```

---

## Usage Guide

### 1. Standalone CLI Fast Direct Mode

Pass any prompt as command-line arguments to get an instant optimized system prompt, user prompt template, and APEX score breakdown:

```powershell
# Conceptual / Educational Prompt
python server.py "tell me about DSA"

# Software Architecture Prompt
python server.py "Make an app for students to study better"

# Research Analysis Prompt
python server.py "Research why startups fail and give me useful information"

# Information Intelligence Briefing
python server.py "Find information about recent technology news"
```

You can also use the explicit `--query` (or `-q`) flag:
```powershell
python server.py -q "Design a distributed caching system with Redis"
```

---

### 2. Interactive Mode (Adaptive Q&A)

Use `-i` (or `--interactive`) to customize the depth, target audience, and output components:

```powershell
python server.py -i "tell me about DSA"
```

Or run without arguments to enter your prompt dynamically:
```powershell
python server.py -i
```

**Interactive Q&A Steps:**
1. **Depth**: Select `[1]` Quick Overview, `[2]` Balanced Practical Guide (Default), or `[3]` Exhaustive Deep Dive.
2. **Focus Elements**: Select `[1]` Definitions & Analogies, `[2]` Real-World Use Cases, `[3]` Implementation Guide, `[4]` Trade-offs & Pitfalls, or `[5]` All of the above (Default).
3. **Audience / Notes**: Enter a custom audience (e.g., *"Junior developers"* or *"Graduate students"*) or press **Enter** to use domain defaults.

---

### 3. Verification Test Suite

Run the built-in offline test suite to verify classification and evaluation across multiple domains:

```powershell
python server.py --test
```

**Output:**
```text
=================================================================
UNIVERSAL PROMPT OPTIMIZER VERIFICATION SUITE
=================================================================

[TEST 1] Testing query: 'Find information about recent technology news.'...
  Detected Task : information_synthesis
  Expert Persona: Senior Technology Intelligence Analyst & Research Specialist
  Quality Score : 94.0/100
  APEX Breakdown: Assessment=90.0 | Performance=86.0 | Execution=100.0 | Excellence=100.0
  [PASS] Verified successfully.

[TEST 2] Testing query: 'Research why startups fail and give me useful information.'...
  Detected Task : academic_research_analysis
  Expert Persona: Senior Academic Researcher & Literature Methodologist
  Quality Score : 100.0/100
  APEX Breakdown: Assessment=100.0 | Performance=100.0 | Execution=100.0 | Excellence=100.0
  [PASS] Verified successfully.

[TEST 3] Testing query: 'Make an app for students to study better.'...
  Detected Task : software_product_engineering
  Expert Persona: Principal Software Architect & Full-Stack Product Engineer
  Quality Score : 92.0/100
  APEX Breakdown: Assessment=100.0 | Performance=86.0 | Execution=100.0 | Excellence=82.0
  [PASS] Verified successfully.

[TEST 4] Testing query: 'tell me about DSA'...
  Detected Task : technical_conceptual_explanation
  Expert Persona: Master Computer Science & STEM Educator (Feynman Technique)
  Quality Score : 88.2/100
  APEX Breakdown: Assessment=85.0 | Performance=100.0 | Execution=86.0 | Excellence=82.0
  [PASS] Verified successfully.

ALL MULTI-DOMAIN VERIFICATION TESTS PASSED.
=================================================================
```

---

### 4. MCP Client Integration (Claude Desktop / Cursor)

When executed without CLI flags, `server.py` starts a standard FastMCP server over `stdio`.

#### Claude Desktop Configuration
Add the server entry to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "prompt-optimizer": {
      "command": "C:\\path\\to\\Prompt_Optimizer\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\Prompt_Optimizer\\server.py"
      ]
    }
  }
}
```

> **Note**: Replace `C:\path\to\Prompt_Optimizer` with the actual path to your repository and virtual environment. On macOS/Linux, use `.../.venv/bin/python` and forward slashes.

---

## Complete MCP Tool Reference

### Fast-Path & 1-Click Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `optimize_prompt` | `user_query: str`<br>`depth: str = "2"`<br>`elements: list[str] = ["5"]`<br>`target_audience: str = "general"` | 1-click universal optimizer that classifies the query, applies preferences, synthesizes system + user prompts, and performs APEX scoring. |
| `optimize_explanation_prompt` | `concept_or_topic: str` | Backwards-compatible alias for 1-click optimization of explanations. |
| `get_adaptive_clarification_questions` | `user_query: str` | Returns adaptive, low-friction clarification questions for host agents/UIs to present to the user. |

---

### 11-Stage Granular Pipeline Tools

For full multi-agent orchestration, the server maintains state across an 11-stage pipeline saved in `.sessions/<session_id>.json`:

| Stage # | Stage Name | MCP Tool | Purpose & Parameters |
| :---: | :--- | :--- | :--- |
| 1 | `input` | `start_pipeline(raw_query)` | Initializes session state and starts pipeline. Returns `session_id`. |
| 2 | `intent` | `submit_intent_analysis(session_id, goal, task_type, keywords)` | Records parsed user goal, classification, and keywords. |
| 3 | `context` | `submit_context_extraction(session_id, audience, tech_stack, constraints, features)` | Captures constraints, target audience, and environment features. |
| 4 | `domain` | `submit_domain_classification(session_id, domain, confidence)` | Records domain mapping and classification confidence score. |
| 5 | `strategy_apet` | `submit_strategy_apet(session_id, action, persona, execution, target)` | Establishes Action-Persona-Execution-Target framework. |
| 6 | `structure_picco` | `submit_structure_picco(session_id, problem, input_, context, constraint, output_format)` | Enforces Problem-Input-Context-Constraint-Output structural blueprint. |
| 7 | `system_prompt_metaspo` | `submit_system_prompt_metaspo(session_id, system_role, policy_rules, output_steering, safety_guardrails)` | Builds the Meta-SPO system prompt with guardrails and policies. |
| 8 | `candidate_prompt` | `submit_candidate_prompt(session_id, candidate_prompt)` | Submits the draft user prompt for quality evaluation. |
| 9 | `apex_evaluation` | `submit_apex_evaluation(session_id, assessment_score?, performance_score?, execution_score?, excellence_score?, flaws?)` | Runs APEX evaluation. Routes to `final` if $\ge 85.0$, otherwise advances to `opro_refinement`. |
| 10 | `opro_refinement` | `submit_opro_refinement(session_id, refined_prompt?, changes_made?)` | Automatically or manually addresses APEX flaws and loops back to evaluation. |
| 11 | `final` | `finalize_prompt(session_id)` | Returns final system prompt, user prompt, APEX score, and complete methodology trail. |

---

### Session & Diagnostic Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `get_pipeline_status` | `session_id: str` | Returns current stage, progress step ($X/11$), and OPRO iteration count. |
| `list_sessions` | *(none)* | Lists all active and saved optimization sessions stored in `.sessions/`. |

---

## Pipeline Execution Flow

```
1. start_pipeline
       │
2. submit_intent_analysis
       │
3. submit_context_extraction
       │
4. submit_domain_classification
       │
5. submit_strategy_apet
       │
6. submit_structure_picco
       │
7. submit_system_prompt_metaspo
       │
8. submit_candidate_prompt
       │
9. submit_apex_evaluation ◄──────────────┐
       │                                 │
       ├──[Score < 85.0 & iters < 3]     │
       │   └──> 10. submit_opro_refinement
       │
       └──[Score >= 85.0 or iters >= 3]
           └──> 11. finalize_prompt
```

---

## Project Structure

```text
Prompt_Optimizer2/
├── README.md             # Complete documentation and user guide
├── LICENSE               # MIT License
└── Prompt_Optimizer/
    ├── server.py         # FastMCP server, synthesizer, evaluation engine & CLI
    ├── .sessions/        # JSON session state persistence directory
    └── .venv/            # Python virtual environment
```

---

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2026 Arpit Kale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
