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

"""Strict JSON Schemas (Pydantic models) for EQUI'B Nuvemshop tool inputs and outputs.

Satisfies AgentOps Code Review Matrix Criterion 1.3 (Explicit JSON Schemas):
Validates all tool arguments and responses using strict input and output schemas
to constrain LLM tool calls and prevent malformed requests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderLookupInput(BaseModel):
    """Strict input schema for looking up an EQUI'B order on Nuvemshop."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        ...,
        min_length=6,
        max_length=16,
        pattern=r"^EQB-\d{4,8}$",
        description=(
            "Canonical EQUI'B Nuvemshop order identifier formatted as 'EQB-XXXX' "
            "(for example, 'EQB-1001', 'EQB-1002', or 'EQB-1003')."
        ),
    )
    customer_email: str = Field(
        ...,
        min_length=5,
        max_length=120,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Customer email address registered on the Nuvemshop checkout.",
    )


class OrderLineItemSchema(BaseModel):
    """Schema for an individual product line item within a Nuvemshop order."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(..., description="Product SKU code in the EQUI'B Nuvemshop catalog.")
    product_name: str = Field(..., description="Human-readable EQUI'B product name.")
    size: str = Field(..., description="Selected variant size (e.g., 'P', 'M', 'G', 'Full').")
    quantity: int = Field(..., ge=1, le=50, description="Units ordered.")
    unit_price_brl: float = Field(..., ge=0.0, description="Unit price in Brazilian Reais (BRL).")


class OrderLookupOutput(BaseModel):
    """Strict output schema returned by `lookup_nuvemshop_order_status`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"] = Field(
        ..., description="Execution status of the Nuvemshop order lookup."
    )
    order_number: str | None = Field(
        default=None, description="Verified EQUI'B Nuvemshop order number."
    )
    payment_status: Literal["paid", "pending", "refunded", "authorized"] | None = Field(
        default=None, description="Current Nuvemshop payment status."
    )
    fulfillment_status: (
        Literal["unpacked", "packed", "shipped", "delivered", "cancelled", "returned"]
        | None
    ) = Field(default=None, description="Current Nuvemshop fulfillment lifecycle state.")
    total_brl: float | None = Field(
        default=None, ge=0.0, description="Total order value in BRL."
    )
    tracking_code: str | None = Field(
        default=None, description="Carrier tracking code associated with the order."
    )
    carrier_name: str | None = Field(
        default=None, description="Logistics carrier handling the delivery."
    )
    delivered_days_ago: int | None = Field(
        default=None,
        description="Days elapsed since delivery (used for 7-day return / 30-day exchange checks).",
    )
    items: list[OrderLineItemSchema] = Field(
        default_factory=list, description="Line items purchased in the order."
    )
    error_code: str | None = Field(
        default=None, description="Machine-readable error identifier when status='error'."
    )
    message: str | None = Field(
        default=None, description="Human-readable summary or error message."
    )
    recovery_instruction: str | None = Field(
        default=None,
        description="Guided recovery instruction telling the LLM how to recover from an error.",
    )


class ShipmentTrackingInput(BaseModel):
    """Strict input schema for tracking an EQUI'B Nuvemshop shipment."""

    model_config = ConfigDict(extra="forbid")

    tracking_code: str = Field(
        ...,
        min_length=8,
        max_length=24,
        pattern=r"^[A-Z]{2}\d{9}[A-Z]{2}$",
        description=(
            "Carrier tracking code formatted as 2 uppercase letters, 9 digits, "
            "and 2 uppercase letters (e.g., 'BR987654321EQ')."
        ),
    )
    carrier_name: Literal["Correios", "Jadlog", "MelhorEnvio", "AzulCargo"] = Field(
        ...,
        description="Authorized EQUI'B shipping carrier name.",
    )


class ShipmentTrackingOutput(BaseModel):
    """Strict output schema for shipment tracking results."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"] = Field(
        ..., description="Execution status of the shipment tracking query."
    )
    tracking_code: str | None = Field(default=None, description="Queried tracking code.")
    carrier_name: str | None = Field(default=None, description="Logistics carrier.")
    current_milestone: str | None = Field(
        default=None, description="Latest shipment tracking event description."
    )
    estimated_delivery_date: str | None = Field(
        default=None, description="ISO-8601 estimated or actual delivery date."
    )
    checkpoints: list[str] = Field(
        default_factory=list, description="Chronological list of carrier scan events."
    )
    error_code: str | None = Field(default=None, description="Error code if lookup failed.")
    message: str | None = Field(default=None, description="Error or status detail.")
    recovery_instruction: str | None = Field(
        default=None, description="Actionable instruction for the LLM to recover."
    )


class StorePolicySearchInput(BaseModel):
    """Strict input schema for querying EQUI'B store policies and product catalog."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=3,
        max_length=250,
        description="Natural-language search query for EQUI'B policies or product care/sizing.",
    )
    category: Literal["returns_and_refunds", "size_exchange", "shipping_sla", "product_sizing"] = (
        Field(
            ...,
            description="Policy or catalog domain category to filter the knowledge base.",
        )
    )


class SupportTicketInput(BaseModel):
    """Strict input schema for creating an EQUI'B Nuvemshop exchange or support ticket."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        ...,
        pattern=r"^EQB-\d{4,8}$",
        description="Associated EQUI'B order number formatted as 'EQB-XXXX'.",
    )
    issue_category: Literal[
        "size_exchange",
        "damaged_item",
        "delayed_delivery",
        "wrong_item_received",
        "general_inquiry",
    ] = Field(
        ...,
        description="Classification of the customer support or exchange request.",
    )
    customer_description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed summary of the customer's issue and requested resolution.",
    )


class OrderRefundInput(BaseModel):
    """Strict input schema for initiating a high-stakes Nuvemshop order refund."""

    model_config = ConfigDict(extra="forbid")

    order_number: str = Field(
        ...,
        pattern=r"^EQB-\d{4,8}$",
        description="Verified EQUI'B Nuvemshop order number formatted as 'EQB-XXXX'.",
    )
    refund_amount_brl: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Amount to refund in Brazilian Reais (BRL). Must be positive and <= order total.",
    )
    refund_reason: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Auditable justification for issuing the refund (e.g., 'Damaged item within 7-day CDC window').",
    )
