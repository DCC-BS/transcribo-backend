from pathlib import Path

from version_pioneer.versionscript import VersionDict

def get_version_dict_wo_exec(
    cwd: str | Path,
    style: str = "pep440",
    tag_prefix: str = "v",
    parentdir_prefix: str | None = None,
    versionfile_source: str | None = None,
    verbose: bool = False,
    root: str | Path | None = None,
) -> VersionDict: ...
