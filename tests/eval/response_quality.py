"""Local LLM-as-judge for `custom_response_quality` (see eval_config.yaml).

Implements the rubric-based LLM-as-a-Judge methodology from "Tutorial Series:
Gemini Enterprise Agent Platform — Part 5: Observability and Evaluation".
"""

import threading

from google import genai
from google.genai import types
from pydantic import BaseModel

_local = threading.local()

EQUIB_EVAL_RUBRICS = [
    (
        "relevance",
        "The response directly and politely addresses the customer's EQUI'B Nuvemshop order, tracking, or policy query.",
    ),
    (
        "brazilian_cdc_policy_compliance",
        "The response accurately applies EQUI'B policy rules (7-day CDC refund window, 30-day complimentary size exchange, or 30-day damaged item warranty).",
    ),
    (
        "identifier_and_hitl_clarity",
        "When applicable, the response clearly states the order number, tracking or reverse-logistics code, or explains when a high-stakes refund (>= R$ 100.00) is paused for Human-in-the-Loop supervisor approval.",
    ),
]


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def _client() -> genai.Client:
    """One client per grading thread.

    The eval SDK grades cases on its own thread pool; this initialization runs once
    per thread. Avoids creating a new client for each eval case, which would re-do
    ADC and the TLS handshake every time. Each thread gets its own client, because
    google-auth freezes the SSL context after the first connection when a client
    certificate is present.
    """
    client = getattr(_local, "client", None)
    if client is None:
        # AI Studio (GEMINI_API_KEY) or Gemini platform (ADC).
        client = _local.client = genai.Client()
    return client


def evaluate(instance):
    reference = instance.get("reference")
    rubric_lines = "\n".join(
        f"- [{r_id}]: {r_desc}" for r_id, r_desc in EQUIB_EVAL_RUBRICS
    )
    rubric = (
        "Grade the agent's final response on a 1-5 scale (1 poor, 5 excellent) "
        "against the following semantic rubrics:\n"
        f"{rubric_lines}\n"
    )
    if reference:
        rubric += (
            "The response should agree semantically with the expected answer below "
            "(do not penalize natural phrasing differences or dynamic ticket IDs)."
        )
    prompt = (
        f"You are an expert QA evaluator for the EQUI'B Nuvemshop AI assistant.\n{rubric}\n\n"
        f"User Prompt: {instance.get('prompt', '')}\n"
        f"Final Response: {instance.get('response', '')}\n"
    )
    if reference:
        prompt += f"Expected Answer (ground truth): {reference}\n"
    prompt += f"Full Agent Trace: {instance.get('agent_data', '')}\n"

    response = _client().models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,  # deterministic grading
            response_mime_type="application/json",
            response_schema=_Verdict,  # guaranteed schema-valid JSON
        ),
    )
    verdict = response.parsed
    if verdict is None:  # model returned nothing usable
        return {"score": 0, "explanation": response.text or ""}
    return {"score": max(1, min(5, verdict.score)), "explanation": verdict.explanation}
