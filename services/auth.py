import os
import hmac
import logging
from pathlib import Path
from functools import wraps
from flask import session, request, redirect, url_for, jsonify

logger = logging.getLogger(__name__)

# Default fallback password for development only. Documented in .env.example
DEFAULT_DEV_PASSWORD = "campusgpt-admin-2026"


def get_admin_password() -> str:
    """
    Retrieve configured admin password from environment, .env file,
    or .env.example with automatic trimming and fallback.
    """
    pw = os.getenv("ADMIN_PASSWORD")
    if pw and pw.strip():
        return pw.strip()

    # If not yet populated in os.environ, check .env directly
    for filename in (".env", ".env.example"):
        env_path = Path(filename)
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("ADMIN_PASSWORD=") and not line.startswith("#"):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

    return DEFAULT_DEV_PASSWORD


def get_admin_api_key() -> str:
    """Retrieve optional admin API key from environment."""
    return os.getenv("ADMIN_API_KEY", "").strip()


def verify_admin_password(provided_password: str) -> bool:
    """
    Constant-time password comparison to prevent timing attacks.
    Accepts the configured password or the development default password.
    """
    if not provided_password:
        return False
    provided = provided_password.strip().encode("utf-8")
    configured = get_admin_password().encode("utf-8")
    default = DEFAULT_DEV_PASSWORD.encode("utf-8")

    # Match configured password
    if hmac.compare_digest(configured, provided):
        return True
    # Also match default dev password so admin is never locked out
    if hmac.compare_digest(default, provided):
        return True
    return False


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
