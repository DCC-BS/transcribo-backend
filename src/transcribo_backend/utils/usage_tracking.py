import hashlib
import hmac

from transcribo_backend.config import settings


def get_pseudonymized_user_id(user_id: str) -> str:
    """
    Generates a consistent, one-way pseudonym for a given user ID.
    """
    if user_id is None:
        user_id = "unknown"
    message = user_id.encode("utf-8")
    signature = hmac.new(settings.hmac_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return signature
