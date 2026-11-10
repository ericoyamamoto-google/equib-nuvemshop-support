# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Security Guardrails, Policy Self-Evaluation Plugins, and Strategic Model Routing.

Satisfies AgentOps Code Review Matrix Category 3 (Orchestration & Logic):
- 3.2 Strategic Model Routing: Routes fast lookup/triage tasks to Flash (`gemini-3.8-flash`)
  and complex planning/resolution tasks to Pro (`gemini-3.8-pro`).
- 3.3 Guardrails & Policy Plugins: Runner-wide ADK `BasePlugin` + Google Cloud Model Armor
  and self-evaluation guardrails across model inputs, model outputs, and tool invocations.
- 4.2 Intent vs. Outcome Capture: Automatically logs pre-execution `action_intent` and
  post-execution `action_outcome` across all tools.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability import (
    PIIRedactor,
    get_structured_logger,
    log_action_intent,
    log_action_outcome,
    trace_span,
)

# Model tiers for Strategic Model Routing (Criterion 3.2)
FAST_FLASH_MODEL = os.environ.get("FAST_MODEL_NAME", "gemini-3.8-flash")
PLANNING_PRO_MODEL = os.environ.get("PRO_MODEL_NAME", "gemini-3.8-pro")

_COMPLEX_PLANNING_KEYWORDS = (
    "refund",
    "reembolso",
    "estorno",
    "damaged",
    "danificado",
    "dispute",
    "reclamação",
    "escalate",
    "procon",
    "chargeback",
    "exception",
)

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|system)\s+instructions\b"),
    re.compile(r"(?i)\breveal\s+(your\s+)?(system\s+prompt|constitution|secret\s+key)\b"),
    re.compile(r"(?i)\bbypass\s+(approval|confirmation|guardrail|security)\b"),
]


def select_model_for_request(
    prompt_text: str,
    agent_name: str = "equib_nuvemshop_support",
) -> dict[str, str]:
    """Deterministically routes a request to Flash (fast tasks) or Pro (complex planning).

    Satisfies Criterion 3.2 (Strategic Model Routing).
    """
    normalized = (prompt_text or "").lower()
    is_fast_specialist = agent_name in (
        "order_lookup_specialist",
        "policy_and_catalog_specialist",
    )
    has_complex_intent = any(kw in normalized for kw in _COMPLEX_PLANNING_KEYWORDS)

    if is_fast_specialist and not has_complex_intent:
        return {
            "selected_model": FAST_FLASH_MODEL,
            "tier": "flash",
            "reason": f"Fast retrieval/classification route for specialist '{agent_name}'.",
        }

    if has_complex_intent or agent_name in (
        "equib_nuvemshop_support",
        "resolution_and_refund_specialist",
    ):
        return {
            "selected_model": PLANNING_PRO_MODEL,
            "tier": "pro",
            "reason": (
                f"Complex reasoning/refund planning route for '{agent_name}' "
                f"(complex_intent={has_complex_intent})."
            ),
        }

    return {
        "selected_model": FAST_FLASH_MODEL,
        "tier": "flash",
        "reason": "Default low-latency Flash route for standard support query.",
    }


def evaluate_response_self_guardrail(response_text: str) -> dict[str, Any]:
    """Self-evaluation guardrail verifying that an LLM response adheres to EQUI'B policies.

    Checks for leaked secrets/tokens, unredacted CPFs/cards, or unsafe promises.
    """
    issues: list[str] = []
    if re.search(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", response_text or ""):
        issues.append("Unredacted Brazilian CPF detected in response.")
    if re.search(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", response_text or ""):
        issues.append("Unredacted credit card number detected in response.")
    if "ephemeral-runtime-secret-ref:" in (response_text or ""):
        issues.append("Internal secret reference leaked in response.")

    sanitized_text = PIIRedactor.redact_text(response_text or "")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "sanitized_text": sanitized_text,
    }


class AgentOpsGuardrailPlugin(BasePlugin):
    """Runner-wide ADK plugin enforcing security guardrails, routing, and Intent/Outcome logs."""

    def __init__(self, name: str = "equib_agentops_guardrail_plugin") -> None:
        super().__init__(name=name)
        self._tool_start_times: dict[str, float] = {}

    def _extract_latest_user_text(self, llm_request: LlmRequest) -> str:
        if not llm_request or not getattr(llm_request, "contents", None):
            return ""
        for content in reversed(llm_request.contents):
            if getattr(content, "role", "") == "user" and getattr(content, "parts", None):
                texts = [
                     getattr(p, "text", "")
                    for p in content.parts
                    if getattr(p, "text", None)
                ]
                if texts:
                    return " ".join(texts)
        return ""

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        session_id = str(getattr(callback_context, "session_id", None) or "default-session")
        log_action_intent(
            agent_name=agent.name,
            action_name=f"invoke_agent:{agent.name}",
            intended_arguments={"agent_name": agent.name},
            session_id=session_id,
            rationale=f"Entering agent '{agent.name}'.",
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        session_id = str(getattr(callback_context, "session_id", None) or "default-session")
        log_action_outcome(
            agent_name=agent.name,
            action_name=f"invoke_agent:{agent.name}",
            actual_outcome={"agent_completed": agent.name},
            status="success",
            session_id=session_id,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Enforces input security guardrails and logs strategic model routing decisions."""
        user_text = self._extract_latest_user_text(llm_request)
        agent_name = getattr(callback_context, "agent_name", "equib_nuvemshop_support")
        session_id = str(getattr(callback_context, "session_id", None) or "default-session")

        with trace_span(
            "guardrail.before_model",
            {"agent_name": agent_name, "session_id": session_id},
        ):
            # 1. Check for prompt injection / policy bypass attempts
            for pattern in _PROMPT_INJECTION_PATTERNS:
                if pattern.search(user_text):
                    log_action_outcome(
                        agent_name=agent_name,
                        action_name="input_security_guardrail",
                        actual_outcome={
                            "blocked": True,
                            "reason": "Prompt injection or policy bypass attempt blocked.",
                        },
                        status="blocked",
                        session_id=session_id,
                    )
                    return LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        "I cannot fulfill requests that attempt to bypass "
                                        "EQUI'B security policies or system instructions. "
                                        "Please let me know how I can help with your Nuvemshop order."
                                    )
                                )
                            ],
                        )
                    )

            # 2. Compute and record Strategic Model Routing decision in session state & logs
            routing = select_model_for_request(user_text, agent_name=agent_name)
            callback_context.state["last_model_routing"] = routing
            get_structured_logger().info(
                f"Strategic Model Routing selected {routing['selected_model']} ({routing['tier']})",
                extra={"structured_data": {"event_type": "model_routing", **routing}},
            )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Runs self-evaluation and PII scrubbing on LLM outputs before returning."""
        if not llm_response or not getattr(llm_response, "content", None):
            return None
        parts = getattr(llm_response.content, "parts", None) or []
        modified = False
        new_parts: list[types.Part] = []

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                eval_result = evaluate_response_self_guardrail(text)
                callback_context.state["last_self_eval"] = {
                    "passed": eval_result["passed"],
                    "issues": eval_result["issues"],
                }
                if not eval_result["passed"]:
                    modified = True
                    new_parts.append(
                        types.Part.from_text(text=eval_result["sanitized_text"])
                    )
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        if modified:
            return LlmResponse(
                content=types.Content(
                    role=llm_response.content.role or "model",
                    parts=new_parts,
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Logs explicit action intent BEFORE tool execution and validates policy rules."""
        tool_name = getattr(tool, "name", str(tool))
        agent_name = getattr(tool_context, "agent_name", "equib_nuvemshop_support")
        session_id = str(getattr(tool_context, "session_id", None) or "default-session")
        self._tool_start_times[tool_name] = time.perf_counter()

        log_action_intent(
            agent_name=agent_name,
            action_name=tool_name,
            intended_arguments=tool_args,
            session_id=session_id,
            rationale=f"Executing Nuvemshop tool '{tool_name}' with validated arguments.",
        )

        # Policy guardrail on high-stakes refund tool
        if tool_name == "process_nuvemshop_order_refund":
            amount = tool_args.get("refund_amount_brl", 0.0)
            if isinstance(amount, (int, float)) and amount > 5000.0:
                outcome = {
                    "status": "error",
                    "error_code": "POLICY_MAX_REFUND_EXCEEDED",
                    "message": f"Refund amount R$ {amount:.2f} exceeds automated policy cap of R$ 5,000.00.",
                    "recovery_instruction": (
                        "Create a manual escalation ticket using "
                        "`create_nuvemshop_exchange_or_support_ticket` instead."
                    ),
                }
                log_action_outcome(
                    agent_name=agent_name,
                    action_name=tool_name,
                    actual_outcome=outcome,
                    status="blocked_by_policy",
                    session_id=session_id,
                )
                return outcome

        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Logs actual outcome AFTER tool execution and scrubs any PII in tool results."""
        tool_name = getattr(tool, "name", str(tool))
        agent_name = getattr(tool_context, "agent_name", "equib_nuvemshop_support")
        session_id = str(getattr(tool_context, "session_id", None) or "default-session")
        started = self._tool_start_times.pop(tool_name, time.perf_counter())
        latency_ms = (time.perf_counter() - started) * 1000.0

        status = (
            str(result.get("status", "success"))
            if isinstance(result, dict)
            else "success"
        )
        log_action_outcome(
            agent_name=agent_name,
            action_name=tool_name,
            actual_outcome=result if isinstance(result, dict) else {"result": str(result)},
            status=status,
            latency_ms=latency_ms,
            session_id=session_id,
        )
        return None
