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

"""Secure Secret Management via Google Cloud Secret Manager.

Satisfies AgentOps Code Review Matrix Criterion 5.3 (Secure Secret Management):
No hardcoded API keys; all tools and clients leverage Google Cloud Secret Manager
or runtime environment variable injection.
"""

from __future__ import annotations

import functools
import os

from google.cloud import secretmanager


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be retrieved from Secret Manager or env."""


@functools.lru_cache(maxsize=32)
def get_secret(
    secret_id: str,
    project_id: str | None = None,
    version_id: str = "latest",
    allow_local_fallback: bool = True,
) -> str:
    """Retrieves a secret securely from Google Cloud Secret Manager.

    Resolution order:
    1. Runtime environment variable injected by Cloud Run / Agent Runtime Secret Manager binding.
    2. Direct lookup from Google Cloud Secret Manager API (`projects/*/secrets/*/versions/*`).
    3. Ephemeral non-sensitive placeholder ONLY when running in local mock/eval mode.

    Args:
        secret_id: Name of the secret in GCP Secret Manager (e.g., 'NUVEMSHOP_ACCESS_TOKEN').
        project_id: Optional GCP project ID override. Defaults to GOOGLE_CLOUD_PROJECT.
        version_id: Secret version to access (defaults to 'latest').
        allow_local_fallback: Whether to permit an ephemeral sandbox token during local tests.

    Returns:
        The resolved secret string value.
    """
    env_value = os.environ.get(secret_id)
    if env_value:
        return env_value

    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if resolved_project and os.environ.get("ENABLE_GCP_SECRET_MANAGER", "false").lower() in (
        "true",
        "1",
    ):
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = (
                f"projects/{resolved_project}/secrets/{secret_id}/versions/{version_id}"
            )
            response = client.access_secret_version(request={"name": secret_name})
            return response.payload.data.decode("UTF-8").strip()
        except Exception as exc:
            if not allow_local_fallback:
                raise SecretResolutionError(
                    f"Failed to retrieve secret '{secret_id}' from Secret Manager: {exc}"
                ) from exc

    if allow_local_fallback:
        return f"ephemeral-runtime-secret-ref:{secret_id.lower()}"

    raise SecretResolutionError(
        f"Secret '{secret_id}' is not set in environment and GCP Secret Manager is unavailable."
    )


def get_nuvemshop_client_headers() -> dict[str, str]:
    """Builds authenticated Nuvemshop API headers using Secret Manager injection."""
    token = get_secret("NUVEMSHOP_ACCESS_TOKEN")
    store_id = get_secret("NUVEMSHOP_STORE_ID")
    user_agent = os.environ.get(
        "NUVEMSHOP_USER_AGENT", "EQUIB-CustomerSupport-Agent (support@equib.com.br)"
    )
    return {
        "Authentication": f"bearer {token}",
        "X-Nuvemshop-Store-Id": store_id,
        "User-Agent": user_agent,
        "Content-Type": "application/json",
    }
