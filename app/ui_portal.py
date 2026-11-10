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

"""Dual-View Web UI for EQUI'B Nuvemshop Customer Support & Order Status Agent.

Provides two dedicated web interfaces mounted on `app/fast_api_app.py`:
1. `/ui/client` — EQUI'B Client Support Portal: Customers can check order status
   (`EQB-1001`, `EQB-1002`, `EQB-1003`), track shipments, search store policies,
   open exchange tickets, request refunds, and see a live status banner when a
   high-stakes refund (>= R$ 100.00) is paused awaiting Human-in-the-Loop approval.
2. `/ui/admin` — EQUI'B Team HITL & Operations Console: Support supervisors can
   view pending HITL refund requests, click Approve or Reject to resume/complete
   the refund, view exchange/support tickets, and inspect live PII-redacted
   structured JSON Intent vs. Outcome audit logs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.observability import (
    STRUCTURED_LOG_FILE_PATH,
    TRACE_SPANS_FILE_PATH,
    PIIRedactor,
    compute_agent_metrics_summary,
    get_feedback_records,
    get_recent_audit_logs,
    get_recent_trace_spans,
    get_review_queue,
    log_action_intent,
    log_action_outcome,
    promote_feedback_to_golden_dataset,
    record_user_feedback,
    trace_span,
)
from app.tools import (
    create_nuvemshop_exchange_or_support_ticket,
    get_created_support_tickets,
    get_hitl_refund_queue,
    get_mock_orders_catalog,
    lookup_nuvemshop_order_status,
    process_nuvemshop_order_refund,
    resolve_hitl_refund_decision,
    search_equib_store_policies_and_catalog,
    track_nuvemshop_shipment_delivery,
)

ui_router = APIRouter(tags=["EQUI'B Dual-View Web UI"])


class ClientPortalActionRequest(BaseModel):
    """Request schema for actions submitted from the EQUI'B Client Support Portal."""

    action_type: Literal[
        "lookup_order",
        "search_policy",
        "create_ticket",
        "request_refund",
    ] = Field(..., description="Client portal action to execute.")
    order_number: str = Field(default="EQB-1001", description="EQUI'B order number.")
    customer_email: str = Field(
        default="marina.silva@example.com", description="Customer checkout email."
    )
    policy_category: str = Field(
        default="returns_and_refunds", description="Policy category filter."
    )
    query: str = Field(
        default="What is the EQUI'B return and exchange policy?",
        description="Policy or product question.",
    )
    issue_category: str = Field(
        default="size_exchange", description="Support ticket issue category."
    )
    customer_description: str = Field(
        default="Requesting complimentary size exchange for my order.",
        description="Customer issue description.",
    )
    refund_amount_brl: float = Field(
        default=389.90, description="Requested refund amount in BRL."
    )
    refund_reason: str = Field(
        default="Damaged velvet dressage saddle pad within 4 days of delivery.",
        description="Justification for the refund request.",
    )


class HitlDecisionRequest(BaseModel):
    """Request schema for approving or rejecting a paused HITL refund in `/ui/admin`."""

    decision: Literal["approve", "reject"] = Field(
        ..., description="Supervisor decision ('approve' or 'reject')."
    )
    reviewer_note: str = Field(
        default="Verified against EQUI'B Nuvemshop order and CDC policy.",
        description="Supervisor audit note.",
    )


class UserFeedbackRequest(BaseModel):
    """Request schema for submitting end-user feedback (thumbs up/down + issue tags)."""

    rating: Literal["thumbs_up", "thumbs_down"] = Field(
        ..., description="User satisfaction rating ('thumbs_up' or 'thumbs_down')."
    )
    issue_tags: list[str] = Field(
        default_factory=list,
        description="Optional issue tags (e.g., 'bad_plan', 'tool_misuse', 'hallucination', 'policy_error').",
    )
    comment: str = Field(default="", description="Optional free-text user feedback.")
    order_number: str = Field(default="EQB-1002", description="Order number context.")
    prompt: str = Field(default="", description="User query or action context.")
    response: str = Field(default="", description="Agent response being rated.")


class PromoteGoldenRequest(BaseModel):
    """Request schema for promoting a flagged Reviewer Queue item to the Golden Eval Set."""

    reviewer_expected_response: str | None = Field(
        default=None,
        description="Optional corrected ground-truth response curated by the human reviewer.",
    )


@ui_router.get("/ui/api/state")
async def get_portal_state() -> dict[str, Any]:
    """Returns live state for both the Client Portal and Team HITL Console."""
    return {
        "orders": get_mock_orders_catalog(),
        "hitl_refunds": get_hitl_refund_queue(),
        "support_tickets": get_created_support_tickets(),
        "audit_logs": PIIRedactor.redact_data(get_recent_audit_logs()[-25:]),
        "trace_spans": PIIRedactor.redact_data(get_recent_trace_spans()[-25:]),
        "metrics_summary": compute_agent_metrics_summary(),
        "review_queue": PIIRedactor.redact_data(get_review_queue()[-25:]),
        "feedback_records": PIIRedactor.redact_data(get_feedback_records()[-25:]),
        "structured_log_file": STRUCTURED_LOG_FILE_PATH,
        "trace_spans_file": TRACE_SPANS_FILE_PATH,
    }


@ui_router.get("/ui/api/telemetry")
async def get_telemetry_snapshot() -> dict[str, Any]:
    """Returns Pillar 1 (Logs), Pillar 2 (Traces), and Pillar 3 (Metrics) telemetry."""
    audit_logs = PIIRedactor.redact_data(get_recent_audit_logs()[-50:])
    trace_spans = PIIRedactor.redact_data(get_recent_trace_spans()[-50:])
    return {
        "status": "healthy",
        "pii_redaction_active": True,
        "structured_log_file": STRUCTURED_LOG_FILE_PATH,
        "trace_spans_file": TRACE_SPANS_FILE_PATH,
        "audit_log_count": len(audit_logs),
        "trace_span_count": len(trace_spans),
        "metrics_summary": compute_agent_metrics_summary(),
        "review_queue": PIIRedactor.redact_data(get_review_queue()[-25:]),
        "audit_logs": audit_logs,
        "trace_spans": trace_spans,
    }


@ui_router.post("/ui/api/feedback")
async def submit_user_feedback(payload: UserFeedbackRequest) -> dict[str, Any]:
    """Records user feedback and automatically routes thumbs-down traces to the Reviewer Queue."""
    entry = record_user_feedback(
        rating=payload.rating,
        issue_tags=payload.issue_tags,
        comment=payload.comment,
        order_number=payload.order_number,
        prompt=payload.prompt,
        response=payload.response,
    )
    return {
        "feedback": entry,
        "metrics_summary": compute_agent_metrics_summary(),
        "review_queue": get_review_queue(),
    }


@ui_router.post("/ui/api/review-queue/{feedback_id}/promote-golden")
async def promote_review_item_to_golden(
    feedback_id: str,
    payload: PromoteGoldenRequest | None = None,
) -> dict[str, Any]:
    """Promotes a flagged Reviewer Queue item into a Golden Set regression test case."""
    expected_override = (
        payload.reviewer_expected_response if payload is not None else None
    )
    res = promote_feedback_to_golden_dataset(
        feedback_id=feedback_id,
        reviewer_expected_response=expected_override,
    )
    return {
        "result": res,
        "review_queue": get_review_queue(),
    }


@ui_router.post("/ui/api/client/action")
async def execute_client_portal_action(
    payload: ClientPortalActionRequest,
) -> dict[str, Any]:
    """Executes a client portal action and logs structured Intent vs. Outcome + OTel span."""
    session_ctx = SimpleNamespace(state={})
    with trace_span(
        f"equib.client_portal.{payload.action_type}",
        attributes={
            "equib.portal": "client",
            "equib.action_type": payload.action_type,
            "equib.order_number": payload.order_number,
        },
    ):
        log_action_intent(
            agent_name="equib_client_portal",
            action_name=f"client_portal:{payload.action_type}",
            intended_arguments=payload.model_dump(),
            session_id="web-ui-client",
        )

        if payload.action_type == "lookup_order":
            order_res = lookup_nuvemshop_order_status(
                order_number=payload.order_number,
                customer_email=payload.customer_email,
                tool_context=session_ctx,  # type: ignore[arg-type]
            )
            tracking_res = None
            if order_res.get("status") == "success" and order_res.get("tracking_code"):
                tracking_res = track_nuvemshop_shipment_delivery(
                    tracking_code=str(order_res["tracking_code"]),
                    carrier_name=str(order_res["carrier_name"]),
                    tool_context=session_ctx,  # type: ignore[arg-type]
                )
            result = {"order": order_res, "tracking": tracking_res}
        elif payload.action_type == "search_policy":
            result = search_equib_store_policies_and_catalog(
                query=payload.query,
                category=payload.policy_category,
                tool_context=session_ctx,  # type: ignore[arg-type]
            )
        elif payload.action_type == "create_ticket":
            result = create_nuvemshop_exchange_or_support_ticket(
                order_number=payload.order_number,
                issue_category=payload.issue_category,
                customer_description=payload.customer_description,
                tool_context=session_ctx,  # type: ignore[arg-type]
            )
        else:
            result = process_nuvemshop_order_refund(
                order_number=payload.order_number,
                refund_amount_brl=payload.refund_amount_brl,
                refund_reason=payload.refund_reason,
                tool_context=session_ctx,  # type: ignore[arg-type]
            )

        log_action_outcome(
            agent_name="equib_client_portal",
            action_name=f"client_portal:{payload.action_type}",
            actual_outcome=result,
            status=str(result.get("status", "success")),
            session_id="web-ui-client",
        )
    return {
        "result": result,
        "hitl_refunds": get_hitl_refund_queue(),
        "support_tickets": get_created_support_tickets(),
    }


@ui_router.post("/ui/api/hitl/{order_number}/decision")
async def submit_hitl_refund_decision(
    order_number: str,
    payload: HitlDecisionRequest,
    x_equib_role: Annotated[str, Header()] = "supervisor",
) -> dict[str, Any]:
    """Approves or rejects a paused HITL refund from the EQUI'B Admin Console."""
    if x_equib_role.lower() not in ("supervisor", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Only EQUI'B Support Supervisors ('supervisor' or 'admin' role) may approve or reject HITL refunds.",
        )
    with trace_span(
        "equib.admin_console.resolve_hitl_refund",
        attributes={
            "equib.portal": "admin",
            "equib.order_number": order_number,
            "equib.decision": payload.decision,
            "equib.role": x_equib_role,
        },
    ):
        log_action_intent(
            agent_name="equib_admin_supervisor",
            action_name="resolve_hitl_refund_decision",
            intended_arguments={
                "order_number": order_number,
                "decision": payload.decision,
                "reviewer_note": payload.reviewer_note,
                "role": x_equib_role,
            },
            session_id="web-ui-admin",
        )
        outcome = resolve_hitl_refund_decision(
            order_number=order_number,
            decision=payload.decision,
            reviewer_note=payload.reviewer_note,
        )
        log_action_outcome(
            agent_name="equib_admin_supervisor",
            action_name="resolve_hitl_refund_decision",
            actual_outcome=outcome,
            status=str(outcome.get("status", "success")),
            session_id="web-ui-admin",
        )
    return {
        "outcome": outcome,
        "hitl_refunds": get_hitl_refund_queue(),
        "audit_logs": PIIRedactor.redact_data(get_recent_audit_logs()[-25:]),
        "trace_spans": PIIRedactor.redact_data(get_recent_trace_spans()[-25:]),
    }


_SHARED_CSS = """
:root {
  --bg-ivory: hsl(38, 38%, 97%);
  --bg-surface: hsl(38, 33%, 99%);
  --bg-warm-sand: hsl(35, 28%, 93%);
  --ink-primary: hsl(24, 28%, 14%);
  --ink-secondary: hsl(24, 15%, 36%);
  --ink-muted: hsl(24, 10%, 52%);
  --hunter-green: hsl(148, 36%, 20%);
  --hunter-green-hover: hsl(148, 38%, 15%);
  --saddle-leather: hsl(26, 46%, 28%);
  --brass-gold: hsl(36, 38%, 48%);
  --amber-bg: hsl(38, 85%, 94%);
  --amber-ink: hsl(28, 75%, 28%);
  --success-bg: hsl(145, 45%, 93%);
  --success-ink: hsl(148, 55%, 22%);
  --danger-bg: hsl(4, 65%, 94%);
  --danger-ink: hsl(4, 62%, 32%);
  --border-subtle: hsl(34, 22%, 86%);
  --shadow-soft: 0 8px 24px -6px hsla(24, 30%, 15%, 0.07);
  --radius-md: 10px;
  --radius-sm: 6px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: var(--bg-ivory);
  color: var(--ink-primary);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  line-height: 1.55;
  padding-bottom: 3rem;
}
header.topbar {
  background-color: var(--hunter-green);
  color: hsl(38, 38%, 97%);
  padding: 1.1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}
.brand-block { display: flex; align-items: baseline; gap: 0.85rem; }
.brand-mark {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}
.brand-subtitle {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  opacity: 0.85;
}
nav.view-switcher a {
  color: hsl(38, 38%, 95%);
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 500;
  padding: 0.45rem 0.9rem;
  border-radius: var(--radius-sm);
  border: 1px solid hsla(38, 38%, 95%, 0.25);
  margin-left: 0.5rem;
  transition: background 0.18s ease;
}
nav.view-switcher a:hover, nav.view-switcher a.active {
  background: hsla(38, 38%, 95%, 0.16);
}
main.container {
  max-width: 1180px;
  margin: 2rem auto;
  padding: 0 1.5rem;
}
h1.page-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 2.15rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--ink-primary);
  margin-bottom: 0.35rem;
}
p.page-lead {
  color: var(--ink-secondary);
  font-size: 0.96rem;
  margin-bottom: 1.5rem;
}
.hitl-banner {
  padding: 1rem 1.25rem;
  border-radius: var(--radius-md);
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.92rem;
}
.hitl-banner.pending {
  background-color: var(--amber-bg);
  color: var(--amber-ink);
  border: 1px solid hsl(36, 65%, 78%);
}
.hitl-banner.approved {
  background-color: var(--success-bg);
  color: var(--success-ink);
  border: 1px solid hsl(145, 42%, 76%);
}
.hitl-banner.rejected {
  background-color: var(--danger-bg);
  color: var(--danger-ink);
  border: 1px solid hsl(4, 52%, 80%);
}
.workspace-grid {
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  gap: 1.5rem;
}
@media (max-width: 900px) {
  .workspace-grid { grid-template-columns: 1fr; }
}
section.panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  box-shadow: var(--shadow-soft);
}
section.panel h2 {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.9rem;
  color: var(--hunter-green);
}
.preset-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.15rem;
}
button.preset-chip {
  background: var(--bg-warm-sand);
  color: var(--ink-primary);
  border: 1px solid var(--border-subtle);
  padding: 0.4rem 0.75rem;
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}
button.preset-chip:hover {
  background: hsl(35, 30%, 88%);
}
.field-group { margin-bottom: 0.9rem; }
.field-group label {
  display: block;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--ink-secondary);
  margin-bottom: 0.3rem;
}
.field-group input, .field-group select, .field-group textarea {
  width: 100%;
  padding: 0.6rem 0.75rem;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  background: #fff;
  color: var(--ink-primary);
  font-size: 0.9rem;
}
.btn-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-top: 1rem;
}
button.btn-primary {
  background: var(--hunter-green);
  color: #fff;
  border: none;
  padding: 0.65rem 1.1rem;
  border-radius: var(--radius-sm);
  font-size: 0.86rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.16s ease;
}
button.btn-primary:hover { background: var(--hunter-green-hover); }
button.btn-leather {
  background: var(--saddle-leather);
  color: #fff;
  border: none;
  padding: 0.65rem 1.1rem;
  border-radius: var(--radius-sm);
  font-size: 0.86rem;
  font-weight: 600;
  cursor: pointer;
}
button.btn-danger {
  background: var(--danger-ink);
  color: #fff;
  border: none;
  padding: 0.55rem 0.95rem;
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
}
pre.json-viewer {
  background: hsl(24, 18%, 12%);
  color: hsl(38, 35%, 92%);
  padding: 1rem;
  border-radius: var(--radius-sm);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  overflow-x: auto;
  max-height: 360px;
}
.queue-item {
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 1rem;
  margin-bottom: 0.85rem;
  background: var(--bg-ivory);
}
.badge {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 4px;
  font-size: 0.74rem;
  font-weight: 600;
  text-transform: uppercase;
}
.badge.pending { background: var(--amber-bg); color: var(--amber-ink); }
.badge.approved { background: var(--success-bg); color: var(--success-ink); }
.badge.rejected { background: var(--danger-bg); color: var(--danger-ink); }
"""


@ui_router.get("/ui/client", response_class=HTMLResponse)
async def render_client_support_portal() -> HTMLResponse:
    """Renders the EQUI'B Client Support & Order Status Portal (`/ui/client`)."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EQUI'B Client Support &amp; Order Status Portal — Nuvemshop</title>
  <meta name="description" content="Official EQUI'B Nuvemshop customer support portal for live order tracking, 30-day size exchanges, store policies, and refund requests." />
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <header class="topbar">
    <div class="brand-block">
      <span class="brand-mark">EQUI'B</span>
      <span class="brand-subtitle">Nuvemshop Client Care &amp; Order Concierge</span>
    </div>
    <nav class="view-switcher" aria-label="Portal View Switcher">
      <a href="/ui/client" id="nav-client-portal" class="active">Client Support Portal</a>
      <a href="/ui/admin" id="nav-admin-console">Team HITL &amp; Ops Console</a>
    </nav>
  </header>

  <main class="container">
    <h1 class="page-title">EQUI'B Client Support &amp; Order Status Portal</h1>
    <p class="page-lead">Track your Nuvemshop order in real time, consult our equestrian sizing &amp; CDC return policies, or request a complimentary size exchange or refund.</p>

    <div id="hitl-status-banner" class="hitl-banner pending" style="display: none;" role="status" aria-live="polite"></div>

    <div class="workspace-grid">
      <section class="panel" aria-labelledby="client-form-heading">
        <h2 id="client-form-heading">1. Select Demo Order or Submit Support Request</h2>
        <div class="preset-bar">
          <button type="button" class="preset-chip" id="preset-eqb-1001" onclick="selectPreset('EQB-1001', 'marina.silva@example.com', 649.90)">
            Order EQB-1001 (Shipped · Correios)
          </button>
          <button type="button" class="preset-chip" id="preset-eqb-1002" onclick="selectPreset('EQB-1002', 'carlos.mendes@example.com', 389.90)">
            Order EQB-1002 (Delivered 4d ago · Damaged Refund)
          </button>
          <button type="button" class="preset-chip" id="preset-eqb-1003" onclick="selectPreset('EQB-1003', 'fernanda.costa@example.com', 520.00)">
            Order EQB-1003 (Delivered 18d ago · 30d Size Exchange)
          </button>
        </div>

        <div class="field-group">
          <label for="client-order-number">Nuvemshop Order Number</label>
          <input id="client-order-number" type="text" value="EQB-1002" />
        </div>
        <div class="field-group">
          <label for="client-email">Checkout Email Address</label>
          <input id="client-email" type="email" value="carlos.mendes@example.com" />
        </div>
        <div class="field-group">
          <label for="client-policy-category">Policy / Product Topic</label>
          <select id="client-policy-category">
            <option value="returns_and_refunds">Returns &amp; Refunds Policy (7-Day CDC &amp; Warranty)</option>
            <option value="size_exchange">Complimentary 30-Day Size Exchange Policy</option>
            <option value="shipping_sla">Nuvemshop Shipping &amp; Carrier SLAs</option>
            <option value="product_sizing">Equestrian Apparel &amp; Tack Sizing Guide</option>
          </select>
        </div>
        <div class="field-group">
          <label for="client-refund-amount">Refund / Claim Amount (R$ BRL)</label>
          <input id="client-refund-amount" type="number" step="0.01" value="389.90" />
        </div>
        <div class="field-group">
          <label for="client-description">Issue Details / Refund Reason</label>
          <textarea id="client-description" rows="2">Damaged velvet dressage saddle pad arrived with torn stitching within 4 days of delivery.</textarea>
        </div>

        <div class="btn-row">
          <button type="button" class="btn-primary" id="btn-lookup-order" onclick="submitClientAction('lookup_order')">
            Check Order &amp; Tracking
          </button>
          <button type="button" class="btn-primary" id="btn-search-policy" onclick="submitClientAction('search_policy')">
            Consult Store Policy
          </button>
          <button type="button" class="btn-leather" id="btn-create-ticket" onclick="submitClientAction('create_ticket')">
            Open Size Exchange Ticket
          </button>
          <button type="button" class="btn-leather" id="btn-request-refund" onclick="submitClientAction('request_refund')">
            Request Refund (HITL Gate)
          </button>
        </div>
      </section>

      <section class="panel" aria-labelledby="client-response-heading">
        <h2 id="client-response-heading">2. Live Agent &amp; Nuvemshop Response</h2>
        <pre id="client-response-output" class="json-viewer">Select an action on the left to query Nuvemshop orders, tracking scans, store policies, or initiate an exchange/refund.</pre>

        <div style="margin-top: 1.15rem; padding-top: 1rem; border-top: 1px solid var(--border-subtle);">
          <h2 style="font-size: 1.05rem; margin-bottom: 0.45rem;">3. Rate This Interaction (Agent Quality Feedback Loop)</h2>
          <p style="font-size: 0.82rem; color: var(--ink-secondary); margin-bottom: 0.6rem;">
            Clicking <strong>Flag Issue (👎)</strong> captures the full execution trace into the <a href="/ui/admin">Reviewer UI Queue</a> for Golden Set curation.
          </p>
          <div class="field-group" style="margin-bottom: 0.5rem;">
            <label for="client-feedback-tag">Optional Issue Tag (for Reviewer UI)</label>
            <select id="client-feedback-tag">
              <option value="">None (General Feedback)</option>
              <option value="bad_plan">bad_plan — Inefficient or flawed reasoning plan</option>
              <option value="tool_misuse">tool_misuse — Wrong tool or parameter formatting</option>
              <option value="hallucination">hallucination — Unverified or invented claim</option>
              <option value="policy_error">policy_error — Misapplied 7-day CDC or 30-day exchange rule</option>
            </select>
          </div>
          <div class="btn-row" style="margin-top: 0.5rem;">
            <button type="button" class="btn-primary" id="btn-feedback-up" onclick="sendFeedback('thumbs_up')">
              👍 Helpful &amp; Accurate
            </button>
            <button type="button" class="btn-danger" id="btn-feedback-down" onclick="sendFeedback('thumbs_down')">
              👎 Flag Issue for Reviewer UI
            </button>
          </div>
          <p id="feedback-status-msg" style="font-size: 0.82rem; margin-top: 0.5rem; color: var(--hunter-green); font-weight: 600;"></p>
        </div>
      </section>
    </div>
  </main>

  <script>
    function selectPreset(orderNum, email, amount) {{
      document.getElementById('client-order-number').value = orderNum;
      document.getElementById('client-email').value = email;
      document.getElementById('client-refund-amount').value = amount;
      refreshHitlBanner();
    }}

    async function submitClientAction(actionType) {{
      const payload = {{
        action_type: actionType,
        order_number: document.getElementById('client-order-number').value,
        customer_email: document.getElementById('client-email').value,
        policy_category: document.getElementById('client-policy-category').value,
        query: document.getElementById('client-description').value,
        issue_category: 'size_exchange',
        customer_description: document.getElementById('client-description').value,
        refund_amount_brl: parseFloat(document.getElementById('client-refund-amount').value || '0'),
        refund_reason: document.getElementById('client-description').value,
      }};
      const res = await fetch('/ui/api/client/action', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
      }});
      const data = await res.json();
      document.getElementById('client-response-output').textContent = JSON.stringify(data.result, null, 2);
      renderBannerFromQueue(data.hitl_refunds || []);
    }}

    function renderBannerFromQueue(queue) {{
      const currentOrder = document.getElementById('client-order-number').value.trim().toUpperCase();
      const match = queue.find(item => item.order_number === currentOrder) || queue[queue.length - 1];
      const banner = document.getElementById('hitl-status-banner');
      if (!match) {{
        banner.style.display = 'none';
        return;
      }}
      banner.style.display = 'flex';
      if (match.status === 'pending_human_confirmation') {{
        banner.className = 'hitl-banner pending';
        banner.innerHTML = `<span><strong>Paused for Human-in-the-Loop Approval:</strong> Refund of R$ ${{match.refund_amount_brl.toFixed(2)}} for order <strong>${{match.order_number}}</strong> requires supervisor confirmation in the <a href="/ui/admin">Team HITL Console</a>.</span>`;
      }} else if (match.status === 'success') {{
        banner.className = 'hitl-banner approved';
        banner.innerHTML = `<span><strong>Refund Approved &amp; Processed (${{match.refund_id}}):</strong> R$ ${{match.refund_amount_brl.toFixed(2)}} refunded for order <strong>${{match.order_number}}</strong>.</span>`;
      }} else if (match.status === 'rejected_by_supervisor') {{
        banner.className = 'hitl-banner rejected';
        banner.innerHTML = `<span><strong>Refund Request Declined:</strong> ${{match.message}}</span>`;
      }}
    }}

    async function refreshHitlBanner() {{
      const res = await fetch('/ui/api/state');
      const data = await res.json();
      renderBannerFromQueue(data.hitl_refunds || []);
    }}

    async function sendFeedback(rating) {{
      const tag = document.getElementById('client-feedback-tag').value;
      const res = await fetch('/ui/api/feedback', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          rating: rating,
          issue_tags: tag ? [tag] : [],
          comment: document.getElementById('client-description').value,
          order_number: document.getElementById('client-order-number').value,
          prompt: document.getElementById('client-description').value,
          response: document.getElementById('client-response-output').textContent
        }}),
      }});
      const data = await res.json();
      const msgEl = document.getElementById('feedback-status-msg');
      if (data.feedback && data.feedback.routed_to_review_queue) {{
        msgEl.textContent = `Trace routed to Reviewer Queue (${{data.feedback.feedback_id}}) in /ui/admin for Golden Set curation.`;
      }} else {{
        msgEl.textContent = `Positive feedback recorded (${{data.feedback.feedback_id}}). Thank you!`;
      }}
    }}
    refreshHitlBanner();
    setInterval(refreshHitlBanner, 4000);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@ui_router.get("/ui/admin", response_class=HTMLResponse)
async def render_admin_hitl_console() -> HTMLResponse:
    """Renders the EQUI'B Team HITL & Operations Console (`/ui/admin`)."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EQUI'B Team HITL &amp; Operations Console — AgentOps</title>
  <meta name="description" content="Internal EQUI'B Nuvemshop supervisor console for approving Human-in-the-Loop refunds, managing exchange tickets, and inspecting PII-redacted structured JSON logs." />
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <header class="topbar">
    <div class="brand-block">
      <span class="brand-mark">EQUI'B</span>
      <span class="brand-subtitle">Team HITL Supervisor &amp; AgentOps Telemetry Console</span>
    </div>
    <nav class="view-switcher" aria-label="Portal View Switcher">
      <a href="/ui/client" id="nav-client-portal">Client Support Portal</a>
      <a href="/ui/admin" id="nav-admin-console" class="active">Team HITL &amp; Ops Console</a>
    </nav>
  </header>

  <main class="container">
    <h1 class="page-title">EQUI'B Team HITL &amp; Operations Console</h1>
    <p class="page-lead">Review and approve/reject paused high-stakes Nuvemshop refunds (&ge; R$ 100.00), monitor Pillar 3 System vs. Quality Metrics, curate the Reviewer Queue into the Golden Set, and audit PII-redacted Logs &amp; OpenTelemetry Traces.</p>

    <section class="panel" aria-labelledby="pillar3-metrics-heading" style="margin-bottom: 1.5rem;">
      <h2 id="pillar3-metrics-heading">Pillar 3 Observability: Operational Vital Signs vs. Model Quality Scorecard</h2>
      <div id="metrics-scorecard-container" style="font-size: 0.88rem; color: var(--ink-secondary);">Loading Pillar 3 System &amp; Quality metrics...</div>
    </section>

    <div class="workspace-grid">
      <section class="panel" aria-labelledby="hitl-queue-heading">
        <h2 id="hitl-queue-heading">1. Human-in-the-Loop (HITL) Refund Approval Queue</h2>
        <div id="hitl-queue-container">Loading HITL refund queue...</div>

        <h2 id="tickets-heading" style="margin-top: 1.5rem;">2. Created Nuvemshop Exchange &amp; Support Tickets</h2>
        <div id="support-tickets-container">Loading support tickets...</div>

        <h2 id="review-queue-heading" style="margin-top: 1.5rem;">3. Reviewer UI Queue (Flagged 👎 Traces &rarr; Golden Set Flywheel)</h2>
        <div id="review-queue-container">Loading Reviewer Queue...</div>
      </section>

      <section class="panel" aria-labelledby="audit-logs-heading">
        <h2 id="audit-logs-heading">4. Pillar 1: PII-Redacted Structured JSON Logs (Intent vs. Outcome)</h2>
        <pre id="audit-logs-viewer" class="json-viewer" style="max-height: 220px;">Loading structured JSON audit logs...</pre>

        <h2 id="trace-spans-heading" style="margin-top: 1.5rem;">5. Pillar 2: OpenTelemetry Distributed Trace Spans (trace_id &amp; span_id)</h2>
        <pre id="trace-spans-viewer" class="json-viewer" style="max-height: 220px;">Loading OpenTelemetry trace spans...</pre>
      </section>
    </div>
  </main>

  <script>
    async function decideHitl(orderNumber, decision) {{
      await fetch(`/ui/api/hitl/${{orderNumber}}/decision`, {{
        method: 'POST',
        headers: {{
          'Content-Type': 'application/json',
          'X-Equib-Role': 'supervisor'
        }},
        body: JSON.stringify({{
          decision: decision,
          reviewer_note: `Supervisor ${{decision}}d via EQUI'B Operations Console.`
        }}),
      }});
      await loadAdminState();
    }}

    async function promoteToGoldenSet(feedbackId) {{
      await fetch(`/ui/api/review-queue/${{feedbackId}}/promote-golden`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{}}),
      }});
      await loadAdminState();
    }}

    async function loadAdminState() {{
      const res = await fetch('/ui/api/state');
      const data = await res.json();

      const hitlContainer = document.getElementById('hitl-queue-container');
      const refunds = data.hitl_refunds || [];
      if (refunds.length === 0) {{
        hitlContainer.innerHTML = '<p style="color: var(--ink-muted); font-size: 0.9rem;">No HITL refund requests queued yet. Submit a refund (&ge; R$ 100.00) from the <a href="/ui/client">Client Portal</a> to test the approval gate.</p>';
      }} else {{
        hitlContainer.innerHTML = refunds.map(r => {{
          const badgeClass = r.status === 'pending_human_confirmation' ? 'pending' : (r.status === 'success' ? 'approved' : 'rejected');
          const actionsHtml = r.status === 'pending_human_confirmation'
            ? `<div class="btn-row">
                 <button type="button" class="btn-primary" id="btn-approve-${{r.order_number}}" onclick="decideHitl('${{r.order_number}}', 'approve')">Approve Refund (R$ ${{r.refund_amount_brl.toFixed(2)}})</button>
                 <button type="button" class="btn-danger" id="btn-reject-${{r.order_number}}" onclick="decideHitl('${{r.order_number}}', 'reject')">Reject Request</button>
               </div>`
            : `<p style="margin-top:0.5rem; font-size:0.84rem; color:var(--ink-secondary);">${{r.message}}</p>`;
          return `<div class="queue-item">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <strong>Order ${{r.order_number}} — R$ ${{r.refund_amount_brl.toFixed(2)}}</strong>
              <span class="badge ${{badgeClass}}">${{r.status}}</span>
            </div>
            <p style="font-size:0.86rem; margin-top:0.35rem;">Reason: ${{r.refund_reason}}</p>
            ${{actionsHtml}}
          </div>`;
        }}).join('');
      }}

      const ticketsContainer = document.getElementById('support-tickets-container');
      const tickets = data.support_tickets || [];
      if (tickets.length === 0) {{
        ticketsContainer.innerHTML = '<p style="color: var(--ink-muted); font-size: 0.9rem;">No exchange or support tickets created yet.</p>';
      }} else {{
        ticketsContainer.innerHTML = tickets.map(t => `<div class="queue-item">
          <strong>${{t.ticket_id}} (${{t.order_number}})</strong> — <span class="badge approved">${{t.issue_category}}</span>
          <p style="font-size:0.84rem; margin-top:0.3rem;">Reverse Logistics Code: <code>${{t.reverse_logistics_code || 'N/A'}}</code></p>
          <p style="font-size:0.84rem; color:var(--ink-secondary);">${{t.sanitized_description}}</p>
        </div>`).join('');
      }}

      const reviewContainer = document.getElementById('review-queue-container');
      const reviewItems = data.review_queue || [];
      if (reviewItems.length === 0) {{
        reviewContainer.innerHTML = '<p style="color: var(--ink-muted); font-size: 0.9rem;">No flagged interactions in Reviewer Queue. Click <strong>Flag Issue (👎)</strong> in the Client Portal to test the event-driven review pipeline.</p>';
      }} else {{
        reviewContainer.innerHTML = reviewItems.map(item => {{
          const promoted = item.review_status === 'promoted_to_golden_set';
          const btnHtml = promoted
            ? `<span class="badge approved">Promoted to Golden Eval Set</span>`
            : `<button type="button" class="btn-primary" style="margin-top:0.45rem;" onclick="promoteToGoldenSet('${{item.feedback_id}}')">Promote Trace to Golden Eval Set</button>`;
          return `<div class="queue-item">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <strong>${{item.feedback_id}} (Order ${{item.order_number}})</strong>
              <span class="badge ${{promoted ? 'approved' : 'pending'}}">${{item.review_status}}</span>
            </div>
            <p style="font-size:0.82rem; margin-top:0.25rem;">Tags: <code>${{(item.issue_tags || []).join(', ') || 'thumbs_down'}}</code> | Trace: <code>${{item.trace_id.slice(0, 16)}}...</code></p>
            <p style="font-size:0.84rem; color:var(--ink-secondary); margin-top:0.25rem;">Prompt: ${{item.prompt}}</p>
            ${{btnHtml}}
          </div>`;
        }}).join('');
      }}

      const metricsEl = document.getElementById('metrics-scorecard-container');
      if (data.metrics_summary) {{
        const sys = data.metrics_summary.system_metrics || {{}};
        const qual = data.metrics_summary.quality_metrics || {{}};
        metricsEl.innerHTML = `
          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;">
            <div class="queue-item" style="margin-bottom:0;">
              <strong>Operational Vital Signs (System Metrics)</strong>
              <p style="margin-top:0.35rem;">• Latency P50 / P99: <strong>${{sys.latency_p50_ms}} ms / ${{sys.latency_p99_ms}} ms</strong></p>
              <p>• Error Rate: <strong>${{sys.error_rate_pct}}%</strong> | Completion Rate: <strong>${{sys.task_completion_rate_pct}}%</strong></p>
              <p>• Est. Tokens / Cost per Task: <strong>${{sys.estimated_tokens_per_task}} tok ($${{sys.estimated_cost_usd_per_run}})</strong></p>
            </div>
            <div class="queue-item" style="margin-bottom:0;">
              <strong>Decision-Making Scorecard (Quality Metrics)</strong>
              <p style="margin-top:0.35rem;">• Trajectory Adherence: <strong>${{qual.trajectory_adherence_rate_pct}}%</strong></p>
              <p>• Safety Guardrail Pass Rate: <strong>${{qual.safety_guardrail_pass_rate_pct}}%</strong> | HITL Gates: <strong>${{qual.hitl_interception_count}}</strong></p>
              <p>• User CSAT: <strong>${{qual.user_csat_pct}}%</strong> (${{qual.total_feedback_count}} ratings, ${{qual.review_queue_pending_count}} pending review)</p>
            </div>
          </div>
          <p style="margin-top:0.65rem; font-weight:600; color:var(--hunter-green);">${{(qual.active_alerts || []).join(' | ')}}</p>
        `;
      }}

      document.getElementById('audit-logs-viewer').textContent = JSON.stringify(data.audit_logs || [], null, 2);
      document.getElementById('trace-spans-viewer').textContent = JSON.stringify(data.trace_spans || [], null, 2);
    }}

    loadAdminState();
    setInterval(loadAdminState, 4000);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
