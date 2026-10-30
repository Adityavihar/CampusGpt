import os
import hmac
import logging
from functools import wraps
from flask import session, request, redirect, url_for, jsonify

logger = logging.getLogger(__name__)

# Default fallback password for development only. Documented in .env.example
DEFAULT_DEV_PASSWORD = "campusgpt-admin-2026"


def get_admin_password() -> str:
    """Retrieve configured admin password from environment."""
    return os.getenv("ADMIN_PASSWORD", DEFAULT_DEV_PASSWORD)


def get_admin_api_key() -> str:
    """Retrieve optional admin API key from environment."""
    return os.getenv("ADMIN_API_KEY", "")


def verify_admin_password(provided_password: str) -> bool:
    """Constant-time password comparison to prevent timing attacks."""
    if not provided_password:
        return False
    expected = get_admin_password().encode("utf-8")
    provided = provided_password.encode("utf-8")
    return hmac.compare_digest(expected, provided)


def is_admin_authenticated() -> bool:
    """
    Check if the current request is authenticated as an administrator
    either via active Flask session or valid admin API key header.
    """
    if session.get("is_admin") is True:
        return True

    # Check header-based authentication for API tools
    api_key_header = request.headers.get("X-Admin-Key") or request.headers.get("Authorization")
    if api_key_header:
        if api_key_header.startswith("Bearer "):
            api_key_header = api_key_header[7:].strip()
        configured_key = get_admin_api_key()
        if configured_key and hmac.compare_digest(configured_key.encode("utf-8"), api_key_header.encode("utf-8")):
            return True
        # Also allow passing the admin password directly as Bearer token for convenience
        if hmac.compare_digest(get_admin_password().encode("utf-8"), api_key_header.encode("utf-8")):
            return True

    return False


def login_admin_session():
    """Mark session as authenticated admin."""
    session["is_admin"] = True
    session.permanent = True


def logout_admin_session():
    """Clear admin authentication from session."""
    session.pop("is_admin", None)


def admin_required(f):
    """Decorator protecting HTML routes by redirecting unauthenticated users to login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin_authenticated():
            return redirect(url_for("admin_login_page", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_api_required(f):
    """Decorator protecting JSON API endpoints with 401 Unauthorized."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin_authenticated():
            return jsonify({
                "error": "Unauthorized",
                "message": "Administrator authentication is required for this operation."
            }), 401
        return f(*args, **kwargs)
    return decorated_function
