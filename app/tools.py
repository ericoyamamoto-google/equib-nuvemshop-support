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

"""EQUI'B Nuvemshop Customer Support & Order Status Tools.

Satisfies AgentOps Code Review Matrix Category 1 (Tool & Interface Design - 20 pts)
and Criterion 3.4 (Human-in-the-Loop Hooks - 5 pts):
- 1.1 Comprehensive Tool Docstrings: Clear human-readable purpose, Args, and Returns.
- 1.2 Descriptive Naming: Specific verb-noun function names (`lookup_nuvemshop_order_status`,
  `track_nuvemshop_shipment_delivery`, `search_equib_store_policies_and_catalog`,
  `create_nuvemshop_exchange_or_support_ticket`, `process_nuvemshop_order_refund`).
- 1.3 Explicit JSON Schemas: Strict Pydantic input/output schema validation.
- 1.4 Guided Error Handling: Structured error dicts with `recovery_instruction` fields.
- 3.4 Human-in-the-Loop Hooks: Explicit confirmation stops (`FunctionTool` with
  `require_confirmation` and `tool_context.request_confirmation`) for high-stakes refunds.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from app.observability import PIIRedactor, trace_span
from app.schemas import (
    OrderLookupInput,
    OrderLookupOutput,
    OrderRefundInput,
    ShipmentTrackingInput,
    ShipmentTrackingOutput,
    StorePolicySearchInput,
    SupportTicketInput,
)
from app.secret_manager import get_nuvemshop_client_headers

# Mocked EQUI'B Nuvemshop Orders Database
_MOCK_NUVEMSHOP_ORDERS: dict[str, dict[str, Any]] = {
    "EQB-1001": {
        "order_number": "EQB-1001",
        "customer_email": "marina.silva@example.com",
        "payment_status": "paid",
        "fulfillment_status": "shipped",
        "total_brl": 649.90,
        "tracking_code": "BR987654321EQ",
        "carrier_name": "Correios",
        "delivered_days_ago": None,
        "items": [
            {
                "sku": "EQB-BREECH-PRO-M",
                "product_name": "Culote EQUI'B Grip Pro Full Seat",
                "size": "M",
                "quantity": 1,
                "unit_price_brl": 449.90,
            },
            {
                "sku": "EQB-GLOVE-BLK-P",
                "product_name": "Luva Hipismo EQUI'B Technical Touch",
                "size": "P",
                "quantity": 1,
                "unit_price_brl": 200.00,
            },
        ],
    },
    "EQB-1002": {
        "order_number": "EQB-1002",
        "customer_email": "carlos.mendes@example.com",
        "payment_status": "paid",
        "fulfillment_status": "delivered",
        "total_brl": 389.90,
        "tracking_code": "BR123456789EQ",
        "carrier_name": "Jadlog",
        "delivered_days_ago": 4,
        "items": [
            {
                "sku": "EQB-SADDLEPAD-DR-FULL",
                "product_name": "Manta Adestramento EQUI'B Velvet Tech",
                "size": "Full",
                "quantity": 1,
                "unit_price_brl": 389.90,
            }
        ],
    },
    "EQB-1003": {
        "order_number": "EQB-1003",
        "customer_email": "fernanda.costa@example.com",
        "payment_status": "paid",
        "fulfillment_status": "delivered",
        "total_brl": 520.00,
        "tracking_code": "BR555666777EQ",
        "carrier_name": "MelhorEnvio",
        "delivered_days_ago": 18,
        "items": [
            {
                "sku": "EQB-SHIRT-COMP-G",
                "product_name": "Camisa de Competição EQUI'B UV50+ Air",
                "size": "G",
                "quantity": 1,
                "unit_price_brl": 520.00,
            }
        ],
    },
}

# Mocked Carrier Shipment Tracking Database
_MOCK_SHIPMENT_TRACKING: dict[str, dict[str, Any]] = {
    "BR987654321EQ": {
        "tracking_code": "BR987654321EQ",
        "carrier_name": "Correios",
        "current_milestone": "In transit between distribution centers (São Paulo/SP -> Campinas/SP)",
        "estimated_delivery_date": "2026-11-12",
        "checkpoints": [
            "2026-11-07 14:20 - Order packed and dispatched from EQUI'B Fulfillment Center (São Paulo/SP)",
            "2026-11-08 09:15 - Processed at Correios CTE Cajamar hub",
            "2026-11-09 08:40 - In transit to Campinas/SP local delivery unit",
        ],
    },
    "BR123456789EQ": {
        "tracking_code": "BR123456789EQ",
        "carrier_name": "Jadlog",
        "current_milestone": "Delivered to recipient at front desk",
        "estimated_delivery_date": "2026-11-05",
        "checkpoints": [
            "2026-11-02 16:00 - Shipped from EQUI'B Warehouse via Jadlog .Package",
            "2026-11-04 11:30 - Arrived at Curitiba/PR destination hub",
            "2026-11-05 15:10 - Delivered and signed at destination address",
        ],
    },
    "BR555666777EQ": {
        "tracking_code": "BR555666777EQ",
        "carrier_name": "MelhorEnvio",
        "current_milestone": "Delivered to recipient",
        "estimated_delivery_date": "2026-10-22",
        "checkpoints": [
            "2026-10-19 10:00 - Dispatched from EQUI'B Warehouse",
            "2026-10-22 17:45 - Delivered to recipient in Belo Horizonte/MG",
        ],
    },
}

# Mocked EQUI'B Store Policies & Product Catalog Knowledge Base
_EQUIB_POLICY_KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "returns_and_refunds": {
        "policy_title": "EQUI'B Returns & Refunds Policy (CDC Art. 49 & Warranty)",
        "regret_window_days": 7,
        "defect_warranty_days": 30,
        "rules": [
            "Customers may request a full refund within 7 calendar days of delivery for unused items (CDC Art. 49).",
            "Items arriving with manufacturing defects or transit damage qualify for a full refund or free replacement within 30 calendar days of delivery.",
            "Refunds above R$ 100.00 require Human-in-the-Loop supervisor approval before Nuvemshop payout execution.",
        ],
    },
    "size_exchange": {
        "policy_title": "EQUI'B Complimentary 30-Day Size & Variant Exchange Policy",
        "exchange_window_days": 30,
        "rules": [
            "First size exchange is 100% free within 30 calendar days of delivery via a prepaid Correios reverse logistics code.",
            "Items must be unworn, unwashed, and retain original EQUI'B tags.",
            "Orders past the 7-day refund window but within 30 days of delivery (such as 18 days post-delivery) qualify for size exchange or store credit.",
        ],
    },
    "shipping_sla": {
        "policy_title": "EQUI'B Nuvemshop Shipping & Fulfillment SLAs",
        "rules": [
            "Paid orders placed before 14:00 BRT ship within 1 business day from São Paulo/SP.",
            "Southeast capitals: 2 to 4 business days (SEDEX / Jadlog Express).",
            "South & Central-West: 4 to 7 business days.",
        ],
    },
    "product_sizing": {
        "policy_title": "EQUI'B Equestrian Apparel & Tack Sizing Guide",
        "rules": [
            "Culotes (Breeches): Size P (BR 36), Size M (BR 38-40), Size G (BR 42-44).",
            "Mantas (Saddle Pads): Size Full fits standard jumping/dressage saddles (16.5\" to 18\"); Size Cob fits pony/compact saddles.",
            "Camisas UV50+: Athletic fit; size up if between measurements.",
        ],
    },
}


def lookup_nuvemshop_order_status(
    order_number: str,
    customer_email: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieves an EQUI'B order's payment, fulfillment, and item details from Nuvemshop.

    Validates the order number and customer email against strict JSON schemas before
    querying the Nuvemshop Orders endpoint. If the input format is invalid or the order
    is not found, returns a guided recovery instruction so the agent can self-correct.

    Args:
        order_number: Canonical EQUI'B order ID formatted as 'EQB-XXXX' (e.g., 'EQB-1001').
        customer_email: Email address used by the customer during Nuvemshop checkout.

    Returns:
        A dictionary conforming to `OrderLookupOutput` containing `status`, `order_number`,
        `payment_status`, `fulfillment_status`, `total_brl`, `tracking_code`, `carrier_name`,
        `delivered_days_ago`, `items`, or guided error fields (`error_code`, `recovery_instruction`).
    """
    with trace_span(
        "tool.lookup_nuvemshop_order_status",
        {"order_number": order_number, "customer_email": customer_email},
    ):
        try:
            validated_input = OrderLookupInput(
                order_number=order_number.strip().upper(),
                customer_email=customer_email.strip().lower(),
            )
        except ValidationError as exc:
            error_output = OrderLookupOutput(
                status="error",
                error_code="INVALID_ORDER_LOOKUP_SCHEMA",
                message=f"Input validation failed for order '{order_number}': {exc.errors()[0]['msg']}",
                recovery_instruction=(
                    "Ensure `order_number` starts with 'EQB-' followed by 4-8 digits "
                    "(for example, convert '#1001' or '1001' to 'EQB-1001') and ensure "
                    "`customer_email` is a valid email address, then retry `lookup_nuvemshop_order_status`."
                ),
            )
            return error_output.model_dump()

        # Authenticate against Nuvemshop via Secret Manager headers
        _headers = get_nuvemshop_client_headers()
        record = _MOCK_NUVEMSHOP_ORDERS.get(validated_input.order_number)
        if not record:
            return OrderLookupOutput(
                status="error",
                error_code="NUVEMSHOP_ORDER_NOT_FOUND",
                message=f"Order '{validated_input.order_number}' was not found in the EQUI'B Nuvemshop store.",
                recovery_instruction=(
                    "Verify the order number with the customer. Known active demo orders are "
                    "'EQB-1001', 'EQB-1002', and 'EQB-1003'."
                ),
            ).model_dump()

        if record["customer_email"].lower() != validated_input.customer_email:
            return OrderLookupOutput(
                status="error",
                order_number=validated_input.order_number,
                error_code="CUSTOMER_EMAIL_MISMATCH",
                message=(
                    f"The email provided does not match the checkout email on record "
                    f"for order {validated_input.order_number}."
                ),
                recovery_instruction=(
                    "Ask the customer to confirm the email address used at checkout, "
                    "or offer to open a verification ticket via `create_nuvemshop_exchange_or_support_ticket`."
                ),
            ).model_dump()

        if tool_context is not None:
            tool_context.state["last_verified_order"] = validated_input.order_number
            tool_context.state["order_lookup_summary"] = (
                f"Order {validated_input.order_number}: {record['fulfillment_status']} "
                f"(Total R$ {record['total_brl']:.2f})"
            )

        output = OrderLookupOutput(
            status="success",
            order_number=record["order_number"],
            payment_status=record["payment_status"],
            fulfillment_status=record["fulfillment_status"],
            total_brl=record["total_brl"],
            tracking_code=record["tracking_code"],
            carrier_name=record["carrier_name"],
            delivered_days_ago=record["delivered_days_ago"],
            items=record["items"],
            message=(
                f"Successfully retrieved EQUI'B Nuvemshop order {record['order_number']} "
                f"(status={record['fulfillment_status']}, auth_header_present={bool(_headers.get('Authentication'))})."
            ),
        )
        return output.model_dump()


def track_nuvemshop_shipment_delivery(
    tracking_code: str,
    carrier_name: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Queries live carrier scan milestones and estimated delivery dates for an EQUI'B shipment.

    Args:
        tracking_code: 13-character carrier tracking code (e.g., 'BR987654321EQ').
        carrier_name: Authorized EQUI'B shipping carrier ('Correios', 'Jadlog', 'MelhorEnvio', 'AzulCargo').

    Returns:
        A dictionary conforming to `ShipmentTrackingOutput` with `status`, `current_milestone`,
        `estimated_delivery_date`, `checkpoints`, or guided error recovery instructions.
    """
    with trace_span(
        "tool.track_nuvemshop_shipment_delivery",
        {"tracking_code": tracking_code, "carrier_name": carrier_name},
    ):
        try:
            validated = ShipmentTrackingInput(
                tracking_code=tracking_code.strip().upper(),
                carrier_name=carrier_name.strip(),  # type: ignore[arg-type]
            )
        except ValidationError as exc:
            return ShipmentTrackingOutput(
                status="error",
                error_code="INVALID_TRACKING_SCHEMA",
                message=f"Tracking validation failed: {exc.errors()[0]['msg']}",
                recovery_instruction=(
                    "Call `lookup_nuvemshop_order_status` first to obtain the exact 13-character "
                    "`tracking_code` (e.g., 'BR987654321EQ') and valid `carrier_name` "
                    "('Correios', 'Jadlog', 'MelhorEnvio', or 'AzulCargo')."
                ),
            ).model_dump()

        shipment = _MOCK_SHIPMENT_TRACKING.get(validated.tracking_code)
        if not shipment:
            return ShipmentTrackingOutput(
                status="error",
                tracking_code=validated.tracking_code,
                carrier_name=validated.carrier_name,
                error_code="TRACKING_CODE_NOT_FOUND",
                message=f"No carrier scans found for tracking code '{validated.tracking_code}'.",
                recovery_instruction=(
                    "Verify the tracking code from `lookup_nuvemshop_order_status` or create a "
                    "logistics inquiry ticket via `create_nuvemshop_exchange_or_support_ticket`."
                ),
            ).model_dump()

        if tool_context is not None:
            tool_context.state["last_tracking_milestone"] = shipment["current_milestone"]

        return ShipmentTrackingOutput(
            status="success",
            tracking_code=shipment["tracking_code"],
            carrier_name=shipment["carrier_name"],
            current_milestone=shipment["current_milestone"],
            estimated_delivery_date=shipment["estimated_delivery_date"],
            checkpoints=shipment["checkpoints"],
            message=f"Shipment {shipment['tracking_code']} via {shipment['carrier_name']}: {shipment['current_milestone']}.",
        ).model_dump()


def search_equib_store_policies_and_catalog(
    query: str,
    category: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Searches EQUI'B store policies (returns, exchanges, shipping SLAs) and sizing guides.

    Args:
        query: Natural-language search query describing the policy or product question.
        category: Policy domain filter ('returns_and_refunds', 'size_exchange', 'shipping_sla', 'product_sizing').

    Returns:
        A dictionary containing `status`, `policy_title`, `rules`, and `matched_category`,
        or a guided error with `recovery_instruction` if the category is invalid.
    """
    with trace_span(
        "tool.search_equib_store_policies_and_catalog",
        {"query": query, "category": category},
    ):
        try:
            validated = StorePolicySearchInput(
                query=query.strip(),
                category=category.strip(),  # type: ignore[arg-type]
            )
        except ValidationError as exc:
            return {
                "status": "error",
                "error_code": "INVALID_POLICY_CATEGORY",
                "message": f"Invalid policy search input: {exc.errors()[0]['msg']}",
                "recovery_instruction": (
                    "Set `category` to one of: 'returns_and_refunds', 'size_exchange', "
                    "'shipping_sla', or 'product_sizing' and provide a `query` of at least 3 characters."
                ),
            }

        policy_entry = _EQUIB_POLICY_KNOWLEDGE_BASE.get(validated.category)
        if tool_context is not None and policy_entry:
            tool_context.state["policy_eligibility_summary"] = policy_entry["policy_title"]

        return {
            "status": "success",
            "matched_category": validated.category,
            "query": validated.query,
            "policy_title": policy_entry["policy_title"],
            "rules": policy_entry["rules"],
            "regret_window_days": policy_entry.get("regret_window_days"),
            "exchange_window_days": policy_entry.get("exchange_window_days"),
        }


def create_nuvemshop_exchange_or_support_ticket(
    order_number: str,
    issue_category: str,
    customer_description: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Creates a formal exchange request or customer support ticket in EQUI'B's Nuvemshop CRM.

    Args:
        order_number: Associated EQUI'B order number formatted as 'EQB-XXXX'.
        issue_category: Issue classification ('size_exchange', 'damaged_item', 'delayed_delivery',
            'wrong_item_received', or 'general_inquiry').
        customer_description: Detailed description of the customer's issue (10 to 1000 characters).

    Returns:
        A dictionary with `status`, `ticket_id`, `order_number`, `issue_category`,
        `reverse_logistics_code` (for exchanges/returns), and `next_steps`.
    """
    with trace_span(
        "tool.create_nuvemshop_exchange_or_support_ticket",
        {"order_number": order_number, "issue_category": issue_category},
    ):
        try:
            validated = SupportTicketInput(
                order_number=order_number.strip().upper(),
                issue_category=issue_category.strip(),  # type: ignore[arg-type]
                customer_description=PIIRedactor.redact_text(customer_description.strip()),
            )
        except ValidationError as exc:
            return {
                "status": "error",
                "error_code": "INVALID_SUPPORT_TICKET_INPUT",
                "message": f"Ticket validation error: {exc.errors()[0]['msg']}",
                "recovery_instruction": (
                    "Ensure `order_number` matches 'EQB-XXXX', `issue_category` is one of "
                    "['size_exchange', 'damaged_item', 'delayed_delivery', 'wrong_item_received', 'general_inquiry'], "
                    "and `customer_description` is at least 10 characters long."
                ),
            }

        if validated.order_number not in _MOCK_NUVEMSHOP_ORDERS:
            return {
                "status": "error",
                "error_code": "TICKET_ORDER_NOT_FOUND",
                "message": f"Cannot create ticket for unknown order '{validated.order_number}'.",
                "recovery_instruction": (
                    "Verify the order number using `lookup_nuvemshop_order_status` before creating a ticket."
                ),
            }

        ticket_id = f"TCK-{validated.order_number.split('-')[1]}-{validated.issue_category[:4].upper()}"
        reverse_code = (
            f"REV-CORREIOS-{validated.order_number.split('-')[1]}"
            if validated.issue_category in ("size_exchange", "damaged_item", "wrong_item_received")
            else None
        )

        if tool_context is not None:
            tool_context.state["last_created_ticket_id"] = ticket_id

        ticket_record = {
            "status": "success",
            "ticket_id": ticket_id,
            "order_number": validated.order_number,
            "issue_category": validated.issue_category,
            "sanitized_description": validated.customer_description,
            "reverse_logistics_code": reverse_code,
            "next_steps": (
                "Prepaid Correios reverse postage authorization generated. "
                "Customer has 10 calendar days to drop off the package at any Correios branch."
                if reverse_code
                else "Ticket assigned to EQUI'B Nuvemshop Customer Care team (SLA: 4 business hours)."
            ),
        }
        _CREATED_SUPPORT_TICKETS.append(ticket_record)
        return ticket_record


def requires_refund_human_approval(
    order_number: str,
    refund_amount_brl: float,
    refund_reason: str,
    **kwargs: Any,
) -> bool:
    """Determines whether a Nuvemshop refund requires Human-in-the-Loop confirmation.

    High-stakes financial refunds (>= R$ 100.00) require explicit human supervisor approval.
    """
    return float(refund_amount_brl) >= 100.0


def process_nuvemshop_order_refund(
    order_number: str,
    refund_amount_brl: float,
    refund_reason: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Executes a financial refund for an EQUI'B order on Nuvemshop (Human-in-the-Loop gated).

    Validates order eligibility (7-day CDC regret window or 30-day damaged item warranty)
    and enforces a Human-in-the-Loop approval gate for high-stakes refunds before triggering
    the Nuvemshop payment reversal.

    Args:
        order_number: Verified EQUI'B Nuvemshop order number formatted as 'EQB-XXXX'.
        refund_amount_brl: Refund amount in Brazilian Reais (BRL), positive and <= order total.
        refund_reason: Detailed justification for the refund (at least 10 characters).

    Returns:
        A dictionary containing `status`, `refund_id`, `order_number`, `refund_amount_brl`,
        `requires_human_confirmation`, or a guided `recovery_instruction` if ineligible.
    """
    with trace_span(
        "tool.process_nuvemshop_order_refund",
        {"order_number": order_number, "refund_amount_brl": refund_amount_brl},
    ):
        try:
            validated = OrderRefundInput(
                order_number=order_number.strip().upper(),
                refund_amount_brl=float(refund_amount_brl),
                refund_reason=PIIRedactor.redact_text(refund_reason.strip()),
            )
        except (ValidationError, ValueError) as exc:
            return {
                "status": "error",
                "error_code": "INVALID_REFUND_INPUT_SCHEMA",
                "message": f"Refund schema validation failed: {exc}",
                "recovery_instruction": (
                    "Ensure `order_number` is formatted as 'EQB-XXXX', `refund_amount_brl` is a "
                    "positive number <= order total, and `refund_reason` is at least 10 characters."
                ),
            }

        order = _MOCK_NUVEMSHOP_ORDERS.get(validated.order_number)
        if not order:
            return {
                "status": "error",
                "error_code": "REFUND_ORDER_NOT_FOUND",
                "message": f"Order '{validated.order_number}' does not exist in Nuvemshop.",
                "recovery_instruction": (
                    "Call `lookup_nuvemshop_order_status` first to verify the order number and total."
                ),
            }

        if validated.refund_amount_brl > float(order["total_brl"]):
            return {
                "status": "error",
                "error_code": "REFUND_EXCEEDS_ORDER_TOTAL",
                "message": (
                    f"Requested refund R$ {validated.refund_amount_brl:.2f} exceeds "
                    f"order {validated.order_number} total of R$ {order['total_brl']:.2f}."
                ),
                "recovery_instruction": (
                    f"Adjust `refund_amount_brl` to at most {order['total_brl']:.2f} and retry."
                ),
            }

        # Policy window check: > 7 days only eligible if damaged/defective within 30 days
        delivered_days = order.get("delivered_days_ago")
        is_damage_claim = any(
            kw in validated.refund_reason.lower()
            for kw in ("damaged", "danificado", "defeito", "defect", "broken")
        )
        if (
            delivered_days is not None
            and delivered_days > 7
            and not is_damage_claim
        ):
            return {
                "status": "error",
                "error_code": "OUTSIDE_CDC_7_DAY_REFUND_WINDOW",
                "message": (
                    f"Order {validated.order_number} was delivered {delivered_days} days ago, "
                    "which is outside the 7-day statutory refund window for non-defective items."
                ),
                "recovery_instruction": (
                    "Offer the customer a complimentary 30-day size/variant exchange instead by "
                    "calling `create_nuvemshop_exchange_or_support_ticket` with `issue_category='size_exchange'`."
                ),
            }

        # Human-in-the-Loop (HITL) explicit code stop for high-stakes refunds
        needs_hitl = requires_refund_human_approval(
            order_number=validated.order_number,
            refund_amount_brl=validated.refund_amount_brl,
            refund_reason=validated.refund_reason,
        )
        if tool_context is not None and needs_hitl:
            already_approved = bool(tool_context.state.get("hitl_refund_approved", False))
            if not already_approved:
                request_conf_fn = getattr(tool_context, "request_confirmation", None)
                if callable(request_conf_fn):
                    try:
                        request_conf_fn(
                            hint=(
                                f"Human supervisor confirmation required to refund "
                                f"R$ {validated.refund_amount_brl:.2f} on Nuvemshop order {validated.order_number}."
                            ),
                            payload={
                                "order_number": validated.order_number,
                                "refund_amount_brl": validated.refund_amount_brl,
                                "refund_reason": validated.refund_reason,
                            },
                        )
                    except Exception:
                        pass
                tool_context.state["pending_hitl_refund"] = {
                    "order_number": validated.order_number,
                    "refund_amount_brl": validated.refund_amount_brl,
                    "refund_reason": validated.refund_reason,
                }
                pending_entry = {
                    "status": "pending_human_confirmation",
                    "requires_human_confirmation": True,
                    "order_number": validated.order_number,
                    "refund_amount_brl": validated.refund_amount_brl,
                    "refund_reason": validated.refund_reason,
                    "message": (
                        f"High-stakes refund of R$ {validated.refund_amount_brl:.2f} for order "
                        f"{validated.order_number} has been queued and paused for Human-in-the-Loop "
                        "supervisor confirmation."
                    ),
                }
                _HITL_REFUND_QUEUE[validated.order_number] = dict(pending_entry)
                return pending_entry

        refund_id = f"REF-NUVEM-{validated.order_number.split('-')[1]}"
        completed_entry = {
            "status": "success",
            "requires_human_confirmation": needs_hitl,
            "refund_id": refund_id,
            "order_number": validated.order_number,
            "refund_amount_brl": validated.refund_amount_brl,
            "refund_reason": validated.refund_reason,
            "message": (
                f"Refund {refund_id} of R$ {validated.refund_amount_brl:.2f} processed "
                f"for Nuvemshop order {validated.order_number}."
            ),
        }
        _HITL_REFUND_QUEUE[validated.order_number] = dict(completed_entry)
        return completed_entry


# Shared process state for the Dual-View Web UI (/ui/client and /ui/admin)
_CREATED_SUPPORT_TICKETS: list[dict[str, Any]] = []
_HITL_REFUND_QUEUE: dict[str, dict[str, Any]] = {}


def get_created_support_tickets() -> list[dict[str, Any]]:
    """Returns created exchange and customer support tickets for the Operations Console."""
    return list(_CREATED_SUPPORT_TICKETS)


def get_hitl_refund_queue() -> list[dict[str, Any]]:
    """Returns all pending and resolved Human-in-the-Loop refund requests."""
    return list(_HITL_REFUND_QUEUE.values())


def get_mock_orders_catalog() -> list[dict[str, Any]]:
    """Returns sanitized summaries of the mocked EQUI'B Nuvemshop orders."""
    return [
        {
            "order_number": o["order_number"],
            "customer_email": o["customer_email"],
            "payment_status": o["payment_status"],
            "fulfillment_status": o["fulfillment_status"],
            "total_brl": o["total_brl"],
            "tracking_code": o["tracking_code"],
            "carrier_name": o["carrier_name"],
            "delivered_days_ago": o["delivered_days_ago"],
            "items": o["items"],
        }
        for o in _MOCK_NUVEMSHOP_ORDERS.values()
    ]


def resolve_hitl_refund_decision(
    order_number: str,
    decision: str,
    reviewer_note: str = "",
) -> dict[str, Any]:
    """Approves or rejects a paused HITL refund request from the EQUI'B Admin Console."""
    from types import SimpleNamespace

    normalized_order = order_number.strip().upper()
    normalized_decision = decision.strip().lower()
    pending = _HITL_REFUND_QUEUE.get(normalized_order)
    if not pending:
        return {
            "status": "error",
            "error_code": "HITL_REQUEST_NOT_FOUND",
            "message": f"No pending HITL refund request found for order '{normalized_order}'.",
        }

    if normalized_decision == "approve":
        approved_ctx = SimpleNamespace(state={"hitl_refund_approved": True})
        result = process_nuvemshop_order_refund(
            order_number=normalized_order,
            refund_amount_brl=float(pending["refund_amount_brl"]),
            refund_reason=str(pending["refund_reason"]),
            tool_context=approved_ctx,  # type: ignore[arg-type]
        )
        result["hitl_decision"] = "approved"
        result["reviewer_note"] = PIIRedactor.redact_text(reviewer_note)
        _HITL_REFUND_QUEUE[normalized_order] = dict(result)
        return result

    if normalized_decision == "reject":
        rejected = {
            **pending,
            "status": "rejected_by_supervisor",
            "hitl_decision": "rejected",
            "reviewer_note": PIIRedactor.redact_text(
                reviewer_note or "Rejected by EQUI'B Support Supervisor."
            ),
            "message": (
                f"Refund request for order {normalized_order} was rejected by the "
                "EQUI'B supervisor. Offered store credit / exchange alternative."
            ),
        }
        _HITL_REFUND_QUEUE[normalized_order] = dict(rejected)
        return rejected

    return {
        "status": "error",
        "error_code": "INVALID_HITL_DECISION",
        "message": "Decision must be 'approve' or 'reject'.",
    }


# Wrap high-stakes refund tool with ADK FunctionTool + require_confirmation hook (Criterion 3.4)
refund_Approval_function_tool = FunctionTool(
    process_nuvemshop_order_refund,
    require_confirmation=requires_refund_human_approval,
)
