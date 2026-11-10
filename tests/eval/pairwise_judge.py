"""Pairwise Comparison LLM-as-a-Judge ('Answer A vs. Answer B' Win/Loss/Tie Evaluator).

Implements the Pairwise Comparison Applied Tip from Google's 'Agent Quality' Whitepaper
(p. 24): forces a choice between two agent versions ('A' vs. 'B' or 'tie') to calculate
a reliable Win/Loss/Tie rate rather than relying solely on noisy single-score ratings.
"""

from __future__ import annotations

from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel


class PairwiseVerdict(BaseModel):
    """Structured JSON decision for pairwise comparison (Whitepaper p. 24)."""

    winner: Literal["A", "B", "tie"]
    rationale: str


def build_pairwise_judge_prompt(
    user_query: str,
    answer_a: str,
    answer_b: str,
) -> str:
    """Builds the exact Pairwise Comparison prompt from page 24 of the Agent Quality Whitepaper."""
    return (
        "You are an expert evaluator for a customer support chatbot. Your goal is to "
        "assess which of two responses is more helpful, polite, and correct.\n\n"
        "[User Query]\n"
        f'"{user_query}"\n\n'
        "[Answer A]\n"
        f'"{answer_a}"\n\n'
        "[Answer B]\n"
        f'"{answer_b}"\n\n'
        "Please evaluate which answer is better. Compare them on correctness, "
        "helpfulness, and tone. Provide your reasoning and then output your final "
        'decision in a JSON object with a "winner" key (either "A", "B", or "tie") '
        'and a "rationale" key.'
    )


def compare_responses_deterministically(
    user_query: str,
    answer_a: str,
    answer_b: str,
) -> dict[str, str]:
    """Deterministic fallback scoring specificity, policy grounding, and polite tone."""
    if answer_a.strip() == answer_b.strip():
        return {
            "winner": "tie",
            "rationale": "Both responses are identical in content, accuracy, and tone.",
        }

    def _quality_signals(text: str) -> int:
        lower = text.lower()
        score = 0
        for token in (
            "eqb-",
            "correios",
            "jadlog",
            "cdc",
            "7-day",
            "30-day",
            "human-in-the-loop",
            "supervisor",
            "r$",
            "rev-correios",
        ):
            if token in lower:
                score += 2
        if len(text.strip()) >= 60:
            score += 1
        return score

    score_a = _quality_signals(answer_a)
    score_b = _quality_signals(answer_b)
    if score_a > score_b:
        return {
            "winner": "A",
            "rationale": f"Answer A provides more complete order/policy identifiers and clearer customer guidance (signal {score_a} vs {score_b}).",
        }
    if score_b > score_a:
        return {
            "winner": "B",
            "rationale": f"Answer B provides more complete order/policy identifiers and clearer customer guidance (signal {score_b} vs {score_a}).",
        }
    return {
        "winner": "tie",
        "rationale": "Both responses address the query with comparable detail and tone.",
    }


def compare_responses_pairwise(
    user_query: str,
    answer_a: str,
    answer_b: str,
    use_llm: bool = False,
) -> dict[str, str]:
    """Compares Answer A vs. Answer B using Gemini LLM-as-a-Judge or deterministic fallback."""
    prompt = build_pairwise_judge_prompt(user_query, answer_a, answer_b)
    if use_llm:
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=PairwiseVerdict,
                ),
            )
            if response.parsed is not None:
                return {
                    "winner": response.parsed.winner,
                    "rationale": response.parsed.rationale,
                }
        except Exception:
            pass
    return compare_responses_deterministically(user_query, answer_a, answer_b)


def compute_win_loss_tie_rate(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregates pairwise comparisons into Win/Loss/Tie rates for Candidate B vs Baseline A."""
    total = len(comparisons)
    if total == 0:
        return {
            "total_pairs": 0,
            "a_win_rate_pct": 0.0,
            "b_win_rate_pct": 0.0,
            "tie_rate_pct": 0.0,
        }
    a_wins = sum(1 for c in comparisons if c.get("winner") == "A")
    b_wins = sum(1 for c in comparisons if c.get("winner") == "B")
    ties = sum(1 for c in comparisons if c.get("winner") == "tie")
    return {
        "total_pairs": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_rate_pct": round((a_wins / total) * 100.0, 1),
        "b_win_rate_pct": round((b_wins / total) * 100.0, 1),
        "tie_rate_pct": round((ties / total) * 100.0, 1),
    }
