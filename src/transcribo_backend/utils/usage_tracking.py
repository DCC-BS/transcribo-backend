import hashlib
import hmac

from fastapi_azure_auth.user import User

from transcribo_backend.config import settings


def get_pseudonymized_user_id(user: User) -> str:
    """
    Generates a consistent, one-way pseudonym for a given user ID.
    """
    user_id = user.oid or user.sub
    if user_id is None:
        raise ValueError("User ID (oid or sub) not found in user object")
    message = user_id.encode("utf-8")
    signature = hmac.new(settings.hmac_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return signature
