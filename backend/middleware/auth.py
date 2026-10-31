"""
Authentication & Authorization Middleware for JEEVAN-DRISHTI
============================================================
Phase 17: FastAPI Backend & Analytical Engine Integration

Protects high-privilege reviewer actions:
- POST /api/v1/alerts/{alert_id}/review
Requires valid X-Reviewer-Key or Bearer token matching JEEVAN_REVIEWER_KEY.
"""

from fastapi import Header, HTTPException, status
from backend.config import settings


def verify_reviewer_token(
    x_reviewer_key: str = Header(None, alias="X-Reviewer-Key"),
    authorization: str = Header(None, alias="Authorization"),
) -> str:
    """
    Validate reviewer token from X-Reviewer-Key header or Bearer authorization.
    Returns the authenticated reviewer identity string or raises 401/403.
    """
    expected_key = settings.REVIEWER_API_KEY

    # Check X-Reviewer-Key
    if x_reviewer_key and x_reviewer_key == expected_key:
        return "authenticated_public_health_reviewer"

    # Check Bearer Authorization
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == expected_key:
            return "authenticated_public_health_reviewer"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Valid X-Reviewer-Key or Bearer token required for public-health review actions.",
        headers={"WWW-Authenticate": "Bearer"},
    )
