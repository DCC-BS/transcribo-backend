from typing import Any, TypedDict

class VersionDict(TypedDict):
    """Type definition for the version dictionary returned by get_version_dict."""

    version: str
    full: str
    branch: str
    date: str
    dirty: bool
    error: str
    full_revisionid: str
    pieces: dict[str, Any]
