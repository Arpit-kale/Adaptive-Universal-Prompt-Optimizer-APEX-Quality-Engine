"""
Adaptive Universal Prompt Optimizer MCP Server & Standalone Verification Engine
================================================================================

"Prompt Optimizer" implements an intelligent, adaptive prompt-engineering pipeline
that dynamically optimizes ANY user prompt — whether it's an educational explanation,
academic research, code generation, technical tutorial, strategic analysis, or general problem solving.

Modes of Operation:
  1. Standard MCP Server: Speaks JSON-RPC over stdio for Claude Desktop, Claude Code, Cursor, etc.
  2. Standalone CLI: Run `python server.py "your prompt here"` to optimize directly in terminal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Suppress pydantic / FastMCP internal lifespan warnings for clean terminal output
warnings.filterwarnings("ignore")

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APEX_PASS_THRESHOLD = 85.0        # Average score required to pass quality gate
MAX_OPRO_ITERATIONS = 3         # Max refinement iterations before forcing finalization

STATE_DIR = Path(__file__).resolve().parent / ".sessions"
STATE_DIR.mkdir(exist_ok=True)

STAGE_ORDER = [
    "input",
    "intent",
    "context",
    "domain",
    "strategy_apet",
    "structure_picco",
    "system_prompt_metaspo",
    "candidate_prompt",
    "apex_evaluation",
    "opro_refinement",
    "final",
]

mcp = FastMCP(
    "Prompt Optimizer",
    instructions=(
        "This server dynamically optimizes ANY prompt for maximum LLM performance, "
        "clarity, and rigor. You can call 'optimize_prompt' directly for an immediate "
        "1-click optimization on any topic, or execute stage-by-stage starting with 'start_pipeline'."
    ),
)


# ---------------------------------------------------------------------------
# Session State Model
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id: str
    raw_query: str
    created_at: float
    stage: str = "input"
    opro_iterations: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def log(self, stage: str, payload: dict[str, Any]) -> None:
        self.history.append({"stage": stage, "payload": payload, "t": time.time()})

    def save(self) -> None:
        path = STATE_DIR / f"{self.session_id}.json"
        temp_path = STATE_DIR / f"{self.session_id}.tmp"
        temp_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temp_path.replace(path)


def _load(session_id: str) -> Session:
    path = STATE_DIR / f"{session_id}.json"
    if not path.exists():
        raise ValueError(f"Unknown session_id '{session_id}'. Call start_pipeline to begin.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Session(**raw)


def _require_stage(sess: Session, expected: str) -> None:
    loop_ok = expected in ("apex_evaluation", "opro_refinement") and sess.stage in (
        "apex_evaluation",
        "opro_refinement",
    )
    if sess.stage != expected and not loop_ok:
        raise ValueError(
            f"Stage mismatch: session is at '{sess.stage}', expected '{expected}'."
        )


def _nonempty(*vals: str) -> bool:
    return all(isinstance(v, str) and v.strip() for v in vals)


# ---------------------------------------------------------------------------
# Dynamic Intent & Domain Classifier (Works with ANY Prompt)
# ---------------------------------------------------------------------------

def classify_prompt_dynamically(query: str) -> dict[str, Any]:
    """Dynamically detects task category, expert persona, and ideal prompt structure for ANY query."""
    q = query.lower().strip()

    # 1. Software Engineering / App Development
    if any(k in q for k in ["app", "software", "code", "python", "javascript", "script", "build a", "make an app", "program", "api", "backend", "frontend"]):
        return {
            "task_type": "software_product_engineering",
            "domain": "computer_science_and_software",
            "persona": "Principal Software Architect & Full-Stack Product Engineer",
            "structure_archetype": "software",
            "quality_focus": "System architecture, modular implementation, edge-case handling, and verification",
        }

    # 2. Academic / Research / Literature Gap Analysis
    elif any(k in q for k in ["research gap", "literature review", "academic", "research paper", "research why", "hypothesis", "thesis", "empirical study"]):
        return {
            "task_type": "academic_research_analysis",
            "domain": "research_and_academia",
            "persona": "Senior Academic Researcher & Literature Methodologist",
            "structure_archetype": "research",
            "quality_focus": "Methodological rigor, state-of-the-art context, and actionable gap identification",
        }

    # 3. Technical & Conceptual Explanation (DSA, Concepts, Pedagogy)
    elif any(k in q for k in ["dsa", "algorithm", "data structure", "explain", "understand", "what is", "how does", "teach", "concept", "intuition"]):
        return {
            "task_type": "technical_conceptual_explanation",
            "domain": "education_and_pedagogy",
            "persona": "Master Computer Science & STEM Educator (Feynman Technique)",
            "structure_archetype": "pedagogy",
            "quality_focus": "Grounding analogies, first-principles scaffolding, and misconception refutation",
        }

    # 4. Information Retrieval / Intelligence Briefing / Trends
    elif any(k in q for k in ["news", "find information", "latest", "recent", "updates", "trends", "market overview"]):
        return {
            "task_type": "information_synthesis",
            "domain": "technology_intelligence",
            "persona": "Senior Technology Intelligence Analyst & Research Specialist",
            "structure_archetype": "information",
            "quality_focus": "Signal-to-noise ratio, verified recent milestones, and disruption impact",
        }

    # 5. Strategic / Business / Comparison
    elif any(k in q for k in ["compare", "vs", "versus", "strategy", "business plan", "market analysis", "pros and cons", "trade-offs"]):
        return {
            "task_type": "strategic_analysis",
            "domain": "strategic_consulting",
            "persona": "Principal Strategic Consultant & Decision Analyst",
            "structure_archetype": "analytical",
            "quality_focus": "Balanced trade-off evaluation, objective criteria, and actionable recommendations",
        }

    # 6. General Expert Consultation
    else:
        return {
            "task_type": "expert_consultation",
            "domain": "general_inquiry",
            "persona": "World-Class Domain Specialist & Rigorous Analytical Consultant",
            "structure_archetype": "general",
            "quality_focus": "Precision, structural completeness, step-by-step reasoning, and zero hallucination",
        }


# ---------------------------------------------------------------------------
# Universal Continuous APEX Quality Evaluator (A-P-E-X Framework)
# ---------------------------------------------------------------------------

def evaluate_prompt_universally(raw_query: str, system_prompt: str, user_prompt: str, archetype: str = "general") -> dict[str, Any]:
    """Continuous, multi-dimensional APEX evaluation across Assessment, Performance, Execution, and Excellence."""
    user_lower = user_prompt.lower()
    system_lower = system_prompt.lower()
    full_lower = f"{system_lower}\n{user_lower}"
    flaws = []

    # ---------------------------------------------------------
    # 1. ASSESSMENT (0 - 100): Task Clarity, Scope, Query Fidelity
    # ---------------------------------------------------------
    a_score = 60.0

    # Query specificity & keyword preservation
    query_words = [w for w in re.findall(r'\b\w+\b', raw_query.lower()) if len(w) > 3 and w not in ["about", "give", "make", "find", "tell", "need"]]
    if query_words:
        retained = sum(1 for w in query_words if w in user_lower)
        retention_ratio = retained / len(query_words)
        a_score += min(20.0, retention_ratio * 20.0)
    else:
        a_score += 15.0

    # Explicit objective & task framing
    if any(k in user_lower for k in ["objective", "task", "inquiry", "requirements", "deliver", "goal", "specification"]):
        a_score += 10.0
    else:
        flaws.append("Assessment: Lacks an explicit statement of core objective or output expectation.")

    # Length and density appropriateness
    words = len(user_prompt.split())
    if 60 <= words <= 350:
        a_score += 10.0
    elif words < 50:
        a_score -= 10.0
        flaws.append("Assessment: The prompt is too brief and lacks task specificity.")

    a_score = min(100.0, max(45.0, a_score))

    # ---------------------------------------------------------
    # 2. PERFORMANCE (0 - 100): Persona Grounding & Domain Authority
    # ---------------------------------------------------------
    p_score = 55.0

    # Senior / expert persona definition
    if any(k in system_lower for k in ["senior", "principal", "master", "specialist", "expert", "architect", "lead"]):
        p_score += 18.0
    elif any(k in system_lower for k in ["you are", "act as", "role"]):
        p_score += 10.0
    else:
        flaws.append("Performance: Lacks an authoritative expert persona / role definition.")

    # Domain-specific methodology guidelines
    if any(k in system_lower for k in ["rigorous", "feynman", "methodology", "first principles", "idiomatic", "evidence", "empirical"]):
        p_score += 14.0

    # Target audience and stakeholder calibration
    if any(k in user_lower for k in ["audience", "reader", "user", "student", "beginner", "practitioner", "researcher", "enterprises"]):
        p_score += 13.0
    else:
        flaws.append("Performance: Lacks explicit audience or stakeholder calibration.")

    p_score = min(100.0, max(45.0, p_score))

    # ---------------------------------------------------------
    # 3. EXECUTION (0 - 100): Structure, Formatting & Progressive Flow
    # ---------------------------------------------------------
    e_score = 50.0

    # Numbered progressive scaffolding (1., 2., 3., 4., etc.)
    numbered = len(re.findall(r'^\s*\d+\.', user_prompt, re.MULTILINE))
    if numbered >= 4:
        e_score += 22.0
    elif numbered >= 2:
        e_score += 15.0
    else:
        flaws.append("Execution: Missing numbered step-by-step scaffolding.")

    # Structural delimiters (sections, requirements, bold headers)
    if any(k in user_lower for k in ["sections:", "requirements:", "##", "**"]):
        e_score += 14.0

    # Output format specifications
    if any(k in user_lower for k in ["format", "structure", "markdown", "summary", "breakdown", "blueprint", "briefing"]):
        e_score += 14.0
    else:
        flaws.append("Execution: Missing explicit output format instructions.")

    e_score = min(100.0, max(45.0, e_score))

    # ---------------------------------------------------------
    # 4. EXCELLENCE (0 - 100): Guardrails, Anti-Hallucination & Edge Cases
    # ---------------------------------------------------------
    x_score = 50.0

    # Quality standards & anti-hallucination guardrails
    if any(k in full_lower for k in ["avoid", "do not", "distinguish", "unsupported", "assumptions", "rigorous", "accurate", "limitations"]):
        x_score += 18.0
    else:
        flaws.append("Excellence: Missing explicit anti-hallucination or negative constraints.")

    # Edge cases / failure modes / trade-offs
    if any(k in user_lower for k in ["edge case", "limitation", "failure", "bottleneck", "trade-off", "pitfall", "misconception", "risk"]):
        x_score += 18.0
    else:
        flaws.append("Excellence: Does not instruct model to address edge cases, pitfalls, or limitations.")

    # Verification / validation / testing
    if any(k in user_lower for k in ["verification", "validate", "test", "measure", "criteria", "evaluation", "check"]):
        x_score += 14.0
    else:
        flaws.append("Excellence: Missing self-verification or validation criteria.")

    x_score = min(100.0, max(45.0, x_score))

    # Composite APEX average
    overall = round((a_score + p_score + e_score + x_score) / 4.0, 1)

    return {
        "assessment_score": round(a_score, 1),
        "performance_score": round(p_score, 1),
        "execution_score": round(e_score, 1),
        "excellence_score": round(x_score, 1),
        "average": overall,
        "flaws": flaws,
    }


# ---------------------------------------------------------------------------
# Adaptive Non-Hardbound Clarification Questions (Universal & Low-Friction)
# ---------------------------------------------------------------------------

def generate_adaptive_clarification_questions(raw_query: str) -> list[dict[str, Any]]:
    """Generates universal, low-friction clarification questions adapted to the user's query."""
    clean_q = raw_query.strip().rstrip(".?!")
    return [
        {
            "id": "depth",
            "question": f"What level of depth do you need for '{clean_q}'?",
            "options": {
                "1": "Quick Overview & Core Intuition (Fast, high-level summary)",
                "2": "Balanced Practical Guide (Clear step-by-step breakdown - Recommended)",
                "3": "Exhaustive Deep Dive (Technical rigor, advanced edge cases & details)",
            },
            "default": "2",
        },
        {
            "id": "elements",
            "question": "Which specific elements should the response focus on?",
            "options": {
                "1": "Clear Definitions & Intuitive Analogies",
                "2": "Concrete Real-World Examples & Use Cases",
                "3": "Step-by-Step Implementation or Actionable Strategy",
                "4": "Trade-offs, Common Pitfalls & Edge Cases",
                "5": "All of the above (Comprehensive)",
            },
            "default": "5",
        },
        {
            "id": "custom_focus",
            "question": "Any specific target audience or custom preferences? (Press Enter to skip)",
            "options": None,
            "default": "General / domain-appropriate",
        },
    ]


# ---------------------------------------------------------------------------
# Universal Prompt Synthesizer
# ---------------------------------------------------------------------------

def synthesize_optimized_prompt(raw_query: str, preferences: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dynamically builds a high-performance prompt tailored to ANY query and optional user preferences."""
    meta = classify_prompt_dynamically(raw_query)
    archetype = meta["structure_archetype"]
    prefs = preferences or {}

    if archetype == "software":
        system_prompt = (
            f"You are a {meta['persona']}. You design elegant, scalable software architectures "
            f"and provide robust, production-ready implementation blueprints with strict error handling."
        )
        user_prompt = (
            f"Design and specify the implementation for: '{raw_query.strip()}'\n\n"
            f"Deliver your specification in the following structured sections:\n"
            f"1. Core Architecture & Tech Stack: Define the frontend, backend, and data storage recommendations.\n"
            f"2. Key Feature Modules & User Flow: Detail the critical features and user journeys to solve the core problem.\n"
            f"3. Production Implementation Blueprint: Provide core data models, API endpoints, or scaffold code.\n"
            f"4. Edge Cases & Offline Considerations: Address data synchronization, edge cases, and failure modes.\n"
            f"5. Testing & Verification Strategy: How to validate reliability, user engagement, and performance."
        )

    elif archetype == "research":
        system_prompt = (
            f"You are a {meta['persona']}. You provide exceptionally rigorous, "
            f"evidence-backed academic synthesis distinguishing empirical data from speculation."
        )
        user_prompt = (
            f"Conduct an academic research inquiry into: '{raw_query.strip()}'\n\n"
            f"Structure your analysis in these distinct sections:\n"
            f"1. Executive Summary & Problem Framing: Define the theoretical domain and key foundational studies.\n"
            f"2. Root Cause Analysis & Empirical Findings: Synthesize peer-reviewed findings and documented causal factors.\n"
            f"3. Identified Research Gaps & Methodological Limitations: Highlight under-explored factors and data collection biases in existing studies.\n"
            f"4. Actionable Hypotheses & Future Study Directions: Propose empirical frameworks and testable hypotheses to investigate these gaps.\n"
            f"5. Measurement & Verification Framework: Detail metrics and evaluation criteria to validate these findings."
        )

    elif archetype == "pedagogy":
        system_prompt = (
            f"You are a {meta['persona']}. You make foundational technical and scientific concepts "
            f"intuitive, vivid, and mathematically sound without unnecessary jargon."
        )
        user_prompt = (
            f"Provide a comprehensive, pedagogical breakdown of: '{raw_query.strip()}'\n\n"
            f"Structure your lesson using this progressive learning scaffold:\n"
            f"1. Intuitive Hook & Everyday Analogy: Explain the core purpose using an everyday physical system or relatable mental model.\n"
            f"2. Core Taxonomy & First Principles: Break down foundational structures, operational mechanics, and memory/flow characteristics.\n"
            f"3. Concrete Walkthrough & Trade-offs: Compare time/space complexity or performance trade-offs across common operations.\n"
            f"4. Common Misconceptions & Pitfalls: Clarify common misunderstandings and implementation edge cases.\n"
            f"5. Hands-on Verification Problem: Provide one classic problem with an optimal vs sub-optimal approach to test understanding."
        )

    elif archetype == "information":
        system_prompt = (
            f"You are a {meta['persona']}. You curate high-signal, factual intelligence briefings "
            f"cutting through hype and marketing noise to highlight verified technical and market developments."
        )
        user_prompt = (
            f"Curate a structured intelligence briefing on: '{raw_query.strip()}'\n\n"
            f"Organize your briefing with the following thematic sections:\n"
            f"1. Top Emerging Developments: Key breakthroughs, announcements, and major milestones in the past 6-12 months.\n"
            f"2. Technological Mechanics & Disruptions: What architectural or technical shifts are enabling these developments?\n"
            f"3. Industry & Societal Impact: How are enterprises, developers, and regulators responding?\n"
            f"4. Critical Limitations & Open Challenges: Technical bottlenecks, ethical concerns, or deployment risks.\n"
            f"5. Verification Sources & Key Metrics: Primary benchmarks, conferences, and repositories tracking this space."
        )

    elif archetype == "analytical":
        system_prompt = (
            f"You are a {meta['persona']}. You provide balanced, objective decision frameworks "
            f"evaluating multi-dimensional trade-offs with commercial and technical acumen."
        )
        user_prompt = (
            f"Conduct a structured strategic evaluation of: '{raw_query.strip()}'\n\n"
            f"Structure your analysis with:\n"
            f"1. Context & Key Objectives: Define the scope, assumptions, and core criteria for evaluation.\n"
            f"2. In-Depth Comparative Analysis: Compare alternatives across feasibility, risk, and impact.\n"
            f"3. Critical Trade-offs: Highlight hidden costs, operational dependencies, and failure modes.\n"
            f"4. Actionable Recommendation: Clear, prioritized next steps based on the findings."
        )

    else:
        system_prompt = (
            f"You are a {meta['persona']}. You provide exceptionally rigorous, clear, "
            f"structured, and actionable answers avoiding ungrounded generalizations."
        )
        user_prompt = (
            f"Task: '{raw_query.strip()}'\n\n"
            f"Requirements:\n"
            f"1. Objective Breakdown: Directly address the core user goal with depth and precision.\n"
            f"2. Structured Deliverable: Organize the response with clear thematic headings and bullet points.\n"
            f"3. Constraints & Edge Cases: State any necessary assumptions and address potential pitfalls.\n"
            f"4. Actionable Summary: Highlight key takeaways or immediate verification steps."
        )

    # Weave user preferences seamlessly into the prompt
    custom_directives = []
    d = str(prefs.get("depth", "2")).strip()
    if d == "1":
        custom_directives.append("- Depth Directive: Keep the explanation concise and high-signal, focusing strictly on high-level mental models and essential takeaways without unnecessary verbosity.")
    elif d == "3":
        custom_directives.append("- Depth Directive: Deliver an exhaustive, technically rigorous breakdown with thorough analysis of underlying mechanisms, edge cases, and granular nuances.")
    elif d == "2":
        custom_directives.append("- Depth Directive: Provide a balanced, practical explanation with step-by-step clarity and actionable depth.")

    elems = prefs.get("elements", ["5"])
    if isinstance(elems, str):
        elems = [elems]
    elems_set = set(str(e).strip() for e in elems)

    if "1" in elems_set or "5" in elems_set:
        custom_directives.append("- Core Definitions & Intuitive Analogies: Ground foundational concepts in relatable real-world analogies and clear definitions.")
    if "2" in elems_set or "5" in elems_set:
        custom_directives.append("- Real-World Use Cases: Include practical, concrete examples demonstrating the concepts in realistic scenarios.")
    if "3" in elems_set or "5" in elems_set:
        custom_directives.append("- Actionable Walkthrough: Detail step-by-step procedures, architectural blueprints, or implementation guidelines.")
    if "4" in elems_set or "5" in elems_set:
        custom_directives.append("- Trade-offs & Pitfalls: Explicitly analyze critical trade-offs, potential failure modes, common misconceptions, and edge cases.")

    aud = str(prefs.get("audience", "general")).strip()
    if aud and aud.lower() not in ["general", "general / domain-appropriate", ""]:
        custom_directives.append(f"- Audience Calibration: Tailor the tone, depth, and terminology specifically for: {aud}.")

    if custom_directives:
        user_prompt += "\n\nUser-Specified Refinements:\n" + "\n".join(custom_directives)

    eval_result = evaluate_prompt_universally(raw_query, system_prompt, user_prompt, archetype)

    return {
        "task_classification": meta,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "apex_score": eval_result["average"],
        "apex_breakdown": {
            "assessment": eval_result["assessment_score"],
            "performance": eval_result["performance_score"],
            "execution": eval_result["execution_score"],
            "excellence": eval_result["excellence_score"],
        },
        "evaluation": eval_result,
    }


# ---------------------------------------------------------------------------
# 1-Click Fast-Path Tools (Universal & Adaptive)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_adaptive_clarification_questions(user_query: str) -> dict:
    """Returns adaptive, low-friction clarification questions to tailor depth and component needs."""
    questions = generate_adaptive_clarification_questions(user_query)
    return {
        "status": "success",
        "user_query": user_query,
        "questions": questions,
    }


@mcp.tool()
def optimize_prompt(
    user_query: str,
    depth: str = "2",
    elements: list[str] | None = None,
    target_audience: str = "general",
) -> dict:
    """Universal 1-Click Optimizer: Adapts to ANY user prompt with optional depth & component preferences."""
    prefs = {
        "depth": depth,
        "elements": elements or ["5"],
        "audience": target_audience,
    }
    synthesis = synthesize_optimized_prompt(user_query, preferences=prefs)
    return {
        "status": "success",
        "detected_task": synthesis["task_classification"]["task_type"],
        "expert_persona": synthesis["task_classification"]["persona"],
        "final_system_prompt": synthesis["system_prompt"],
        "final_user_prompt": synthesis["user_prompt"],
        "apex_quality_score": synthesis["apex_score"],
        "apex_breakdown": synthesis["apex_breakdown"],
    }


@mcp.tool()
def optimize_explanation_prompt(concept_or_topic: str) -> dict:
    """Alias for backwards compatibility: Optimizes an explanation prompt for any concept."""
    return optimize_prompt(concept_or_topic)


# ---------------------------------------------------------------------------
# Multi-Stage Pipeline Tools (Universal)
# ---------------------------------------------------------------------------

@mcp.tool()
def start_pipeline(raw_query: str) -> dict:
    """Start a new prompt-optimization run with any user query."""
    session_id = uuid.uuid4().hex[:12]
    sess = Session(session_id=session_id, raw_query=raw_query, created_at=time.time())
    sess.data["input"] = {"raw_query": raw_query}
    sess.stage = "intent"
    sess.log("input", {"raw_query": raw_query})
    sess.save()
    return {
        "session_id": session_id,
        "stage_completed": "input",
        "next_action": "submit_intent_analysis",
    }


@mcp.tool()
def submit_intent_analysis(session_id: str, goal: str, task_type: str, keywords: list[str]) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "intent")
    sess.data["intent"] = {"goal": goal, "task_type": task_type, "keywords": keywords}
    sess.stage = "context"
    sess.log("intent", sess.data["intent"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "intent", "next_action": "submit_context_extraction"}


@mcp.tool()
def submit_context_extraction(
    session_id: str,
    audience: str = "target reader / practitioner",
    tech_stack: str = "n/a",
    constraints: list[str] | None = None,
    features: list[str] | None = None,
) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "context")
    sess.data["context"] = {
        "audience": audience,
        "tech_stack": tech_stack,
        "constraints": constraints or [],
        "features": features or [],
    }
    sess.stage = "domain"
    sess.log("context", sess.data["context"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "context", "next_action": "submit_domain_classification"}


@mcp.tool()
def submit_domain_classification(session_id: str, domain: str, confidence: float = 1.0) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "domain")
    sess.data["domain"] = {"domain": domain.strip().lower(), "confidence": confidence}
    sess.stage = "strategy_apet"
    sess.log("domain", sess.data["domain"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "domain", "next_action": "submit_strategy_apet"}


@mcp.tool()
def submit_strategy_apet(session_id: str, action: str, persona: str, execution: str, target: str) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "strategy_apet")
    sess.data["strategy_apet"] = {
        "action": action, "persona": persona, "execution": execution, "target": target
    }
    sess.stage = "structure_picco"
    sess.log("strategy_apet", sess.data["strategy_apet"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "strategy_apet", "next_action": "submit_structure_picco"}


@mcp.tool()
def submit_structure_picco(
    session_id: str,
    problem: str,
    input_: str,
    context: str,
    constraint: list[str] | None = None,
    output_format: str = "Clear structured markdown",
) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "structure_picco")
    sess.data["structure_picco"] = {
        "problem": problem,
        "input": input_,
        "context": context,
        "constraint": constraint or ["Maintain high rigor and clarity"],
        "output_format": output_format,
    }
    sess.stage = "system_prompt_metaspo"
    sess.log("structure_picco", sess.data["structure_picco"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "structure_picco", "next_action": "submit_system_prompt_metaspo"}


@mcp.tool()
def submit_system_prompt_metaspo(
    session_id: str,
    system_role: str,
    policy_rules: list[str],
    output_steering: list[str],
    safety_guardrails: list[str] | None = None,
) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "system_prompt_metaspo")
    sess.data["system_prompt_metaspo"] = {
        "system_role": system_role,
        "policy_rules": policy_rules or ["Provide factual and rigorously validated answers."],
        "output_steering": output_steering or ["Use clear headings and logical progression."],
        "safety_guardrails": safety_guardrails or [],
    }
    sess.stage = "candidate_prompt"
    sess.log("system_prompt_metaspo", sess.data["system_prompt_metaspo"])
    sess.save()
    return {"session_id": session_id, "stage_completed": "system_prompt_metaspo", "next_action": "submit_candidate_prompt"}


@mcp.tool()
def submit_candidate_prompt(session_id: str, candidate_prompt: str) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "candidate_prompt")
    cleaned = candidate_prompt.strip()
    sess.data["candidate_prompt"] = cleaned
    sess.data.setdefault("candidate_prompt_versions", []).append(cleaned)
    sess.stage = "apex_evaluation"
    sess.log("candidate_prompt", {"candidate_prompt": cleaned})
    sess.save()
    return {"session_id": session_id, "stage_completed": "candidate_prompt", "next_action": "submit_apex_evaluation"}


@mcp.tool()
def submit_apex_evaluation(
    session_id: str,
    assessment_score: float | None = None,
    performance_score: float | None = None,
    execution_score: float | None = None,
    excellence_score: float | None = None,
    flaws: list[str] | None = None,
) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "apex_evaluation")

    current_prompt = sess.data.get("candidate_prompt", "")
    domain = sess.data.get("domain", {}).get("domain", "general")
    archetype = "pedagogy" if "explain" in domain or "pedagogy" in domain else ("research" if "research" in domain else "general")
    objective = evaluate_prompt_universally(current_prompt, archetype)

    if assessment_score is not None and performance_score is not None:
        host_avg = (assessment_score + performance_score + (execution_score or 85.0) + (excellence_score or 85.0)) / 4.0
        avg = round((host_avg + objective["average"]) / 2.0, 1)
        merged_flaws = list(dict.fromkeys((flaws or []) + objective["flaws"]))
    else:
        avg = round(objective["average"], 1)
        merged_flaws = objective["flaws"]

    eval_entry = {"average": avg, "flaws": merged_flaws, "iteration": sess.opro_iterations}
    sess.data.setdefault("apex_evaluations", []).append(eval_entry)
    sess.log("apex_evaluation", eval_entry)

    if avg >= APEX_PASS_THRESHOLD or sess.opro_iterations >= MAX_OPRO_ITERATIONS:
        sess.stage = "final"
        sess.save()
        return {
            "session_id": session_id,
            "stage_completed": "apex_evaluation",
            "average_score": avg,
            "passed": avg >= APEX_PASS_THRESHOLD,
            "next_action": "finalize_prompt",
        }

    sess.stage = "opro_refinement"
    sess.save()
    return {
        "session_id": session_id,
        "stage_completed": "apex_evaluation",
        "average_score": avg,
        "passed": False,
        "flaws": merged_flaws,
        "next_action": "submit_opro_refinement",
    }


@mcp.tool()
def submit_opro_refinement(
    session_id: str,
    refined_prompt: str | None = None,
    changes_made: list[str] | None = None,
) -> dict:
    sess = _load(session_id)
    _require_stage(sess, "opro_refinement")

    last_eval = sess.data.get("apex_evaluations", [])[-1]
    flaws = last_eval.get("flaws", [])

    if refined_prompt and len(refined_prompt.strip()) >= 30:
        chosen_prompt = refined_prompt.strip()
        chosen_changes = changes_made or ["Applied user refinement."]
    else:
        # Dynamic enhancement based on flaws
        current = sess.data.get("candidate_prompt", "")
        additions = []
        for flaw in flaws:
            if "analogy" in flaw:
                additions.append("- Grounding Analogy: Anchor the core concept in a familiar physical or real-world analogy.")
            elif "misconception" in flaw:
                additions.append("- Misconception Alert: Explicitly address and refute common misunderstandings.")
            elif "gap" in flaw or "research" in flaw:
                additions.append("- Research Gaps: Identify 3-4 specific open problems and current methodological bottlenecks.")
            elif "edge case" in flaw:
                additions.append("- Edge Cases: Detail potential failure modes and boundary conditions.")
            elif "format" in flaw or "structure" in flaw:
                additions.append("- Structure: Organize the output with clear, progressive markdown headings.")

        if not additions:
            additions.append("- Quality Check: Ensure factual accuracy, clarity, and structural completeness.")

        enhanced_section = "\n\n### Mandatory Refinement Requirements:\n" + "\n".join(additions)
        chosen_prompt = current + enhanced_section
        chosen_changes = [f"Addressed: {f}" for f in flaws] or ["Enhanced structural completeness."]

    sess.opro_iterations += 1
    sess.data["candidate_prompt"] = chosen_prompt
    sess.data.setdefault("candidate_prompt_versions", []).append(chosen_prompt)
    sess.stage = "apex_evaluation"
    sess.log("opro_refinement", {"refined_prompt": chosen_prompt, "changes": chosen_changes})
    sess.save()

    return {
        "session_id": session_id,
        "stage_completed": "opro_refinement",
        "opro_iteration": sess.opro_iterations,
        "refined_prompt": chosen_prompt,
        "changes_made": chosen_changes,
        "next_action": "submit_apex_evaluation",
    }


@mcp.tool()
def finalize_prompt(session_id: str) -> dict:
    sess = _load(session_id)
    if sess.stage != "final":
        raise ValueError(f"Pipeline not ready to finalize (currently at '{sess.stage}').")

    last_eval = sess.data["apex_evaluations"][-1]
    system_role = (
        sess.data.get("system_prompt_metaspo", {}).get("system_role")
        or sess.data.get("strategy_apet", {}).get("persona")
        or "Master Domain Specialist"
    )

    return {
        "session_id": session_id,
        "final_system_prompt": system_role,
        "final_user_prompt": sess.data["candidate_prompt"],
        "final_apex_score": last_eval["average"],
        "iterations_used": sess.opro_iterations,
        "optimization_status": {
            "passed": last_eval["average"] >= APEX_PASS_THRESHOLD,
            "threshold": APEX_PASS_THRESHOLD,
        },
        "methodology_trail": sess.data,
    }


# ---------------------------------------------------------------------------
# Utility Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pipeline_status(session_id: str) -> dict:
    sess = _load(session_id)
    idx = STAGE_ORDER.index(sess.stage) if sess.stage in STAGE_ORDER else len(STAGE_ORDER) - 1
    return {
        "session_id": session_id,
        "current_stage": sess.stage,
        "progress": f"{idx + 1}/{len(STAGE_ORDER)}",
        "opro_iterations": sess.opro_iterations,
    }


@mcp.tool()
def list_sessions() -> dict:
    sessions = []
    for path in sorted(STATE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "raw_query": data.get("raw_query"),
                "stage": data.get("stage"),
            })
        except Exception:
            continue
    return {"sessions": sessions}


# ---------------------------------------------------------------------------
# Standalone CLI & Verification Suite (Allows Testing ANY Prompt)
# ---------------------------------------------------------------------------

def run_self_test():
    """Runs a complete verification of the optimizer across multiple distinct domains."""
    print("=" * 65)
    print("UNIVERSAL PROMPT OPTIMIZER VERIFICATION SUITE")
    print("=" * 65)

    test_queries = [
        "Find information about recent technology news.",
        "Research why startups fail and give me useful information.",
        "Make an app for students to study better.",
        "tell me about DSA",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n[TEST {i}] Testing query: '{q}'...")
        res = optimize_prompt(q)
        print(f"  Detected Task : {res['detected_task']}")
        print(f"  Expert Persona: {res['expert_persona']}")
        print(f"  Quality Score : {res['apex_quality_score']}/100")
        b = res.get("apex_breakdown", {})
        print(f"  APEX Breakdown: Assessment={b.get('assessment')} | Performance={b.get('performance')} | Execution={b.get('execution')} | Excellence={b.get('excellence')}")
        assert res["apex_quality_score"] >= 80.0
        print(f"  [PASS] Verified successfully.")

    print("\nALL MULTI-DOMAIN VERIFICATION TESTS PASSED.")
    print("=" * 65)


def run_interactive_cli(query: str | None = None):
    """Interactively asks adaptive, low-friction questions to tailor prompt depth and elements."""
    if not query:
        print("\n" + "=" * 65)
        print("ADAPTIVE PROMPT OPTIMIZER - INTERACTIVE MODE")
        print("=" * 65)
        query = input("Enter your raw prompt or topic: ").strip()
        if not query:
            print("No prompt provided. Exiting.")
            return

    print("\n" + "=" * 65)
    print(f"CUSTOMIZING PROMPT FOR: '{query}'")
    print("=" * 65)
    print("Answer 2-3 quick questions to tailor the prompt (or press Enter for defaults):\n")

    questions = generate_adaptive_clarification_questions(query)
    answers = {}

    # Q1: Depth
    q1 = questions[0]
    print(f"1. {q1['question']}")
    for k, v in q1["options"].items():
        print(f"   [{k}] {v}")
    ans1 = input(f"   Select (1-3) [Default: 2 - Balanced]: ").strip()
    answers["depth"] = ans1 if ans1 in q1["options"] else q1["default"]

    # Q2: Elements
    print()
    q2 = questions[1]
    print(f"2. {q2['question']}")
    for k, v in q2["options"].items():
        print(f"   [{k}] {v}")
    ans2 = input(f"   Select (1-5) [Default: 5 - All]: ").strip()
    answers["elements"] = [ans2] if ans2 in q2["options"] else [q2["default"]]

    # Q3: Target audience / custom focus
    print()
    q3 = questions[2]
    print(f"3. {q3['question']}")
    ans3 = input("   Target audience or notes [Press Enter to skip]: ").strip()
    answers["audience"] = ans3 if ans3 else q3["default"]

    print("\nSynthesizing customized prompt & calculating APEX scores...")
    res = synthesize_optimized_prompt(query, preferences=answers)

    print("\n" + "=" * 65)
    print("YOUR CUSTOMIZED OPTIMIZED PROMPT:")
    print("=" * 65)
    print(f"Detected Category: {res['task_classification']['task_type']}")
    print(f"Assigned Persona : {res['task_classification']['persona']}")
    print("\n[SYSTEM PROMPT]:")
    print(res["system_prompt"])
    print("\n[USER PROMPT TEMPLATE]:")
    print(res["user_prompt"])
    print(f"\nAPEX Quality Score: {res['apex_score']}/100")
    b = res["apex_breakdown"]
    print(f"APEX Dimensions   : Assessment={b['assessment']} | Performance={b['performance']} | Execution={b['execution']} | Excellence={b['excellence']}")
    print("=" * 65)


if __name__ == "__main__":
    # Handle command line args gracefully
    cleaned_argv = []
    for arg in sys.argv[1:]:
        if arg.startswith("--query") and len(arg) > 7 and not arg.startswith("--query="):
            cleaned_argv.append("--query")
            cleaned_argv.append(arg[7:].strip('"\''))
        else:
            cleaned_argv.append(arg)

    parser = argparse.ArgumentParser(description="Adaptive Universal Prompt Optimizer")
    parser.add_argument("--test", action="store_true", help="Run offline verification test suite")
    parser.add_argument("-i", "--interactive", action="store_true", help="Ask adaptive questions to tailor depth and components")
    parser.add_argument("--query", "-q", type=str, default=None, help="Quickly optimize any prompt in terminal")
    parser.add_argument("prompt", nargs="*", default=None, help="Positional prompt text")
    args = parser.parse_args(cleaned_argv)

    # Reconstruct prompt if passed as positional arguments
    query_text = args.query
    if not query_text and args.prompt:
        query_text = " ".join(args.prompt)

    if args.test:
        run_self_test()
    elif args.interactive:
        run_interactive_cli(query_text)
    elif query_text:
        result = optimize_prompt(user_query=query_text)
        print("\n" + "=" * 65)
        print(f"OPTIMIZED PROMPT FOR: '{query_text}'")
        print("=" * 65)
        print(f"Detected Category: {result['detected_task']}")
        print(f"Assigned Persona : {result['expert_persona']}")
        print("\n[SYSTEM PROMPT]:")
        print(result["final_system_prompt"])
        print("\n[USER PROMPT TEMPLATE]:")
        print(result["final_user_prompt"])
        print(f"\nAPEX Quality Score: {result['apex_quality_score']}/100")
        if "apex_breakdown" in result:
            b = result["apex_breakdown"]
            print(f"APEX Dimensions   : Assessment={b['assessment']} | Performance={b['performance']} | Execution={b['execution']} | Excellence={b['excellence']}")
        print("=" * 65)
        print("[Tip: Run with -i (or --interactive) to customize depth, examples, and use cases!]")
    else:
        mcp.run()
