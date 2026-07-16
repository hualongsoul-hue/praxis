"""技能资源路径边界。"""

from pathlib import Path, PurePosixPath, PureWindowsPath

from praxis.exceptions import SkillError


def resolve_skill_path(root: str | Path, relative: str | Path) -> Path:
    """解析技能资源，并保证最终目标位于技能根目录内。"""
    base = Path(root).expanduser().resolve()
    raw = str(relative)
    if not raw or PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise SkillError("技能资源路径必须是非空相对路径", details={"path": raw})
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SkillError(
            "技能资源路径越过技能根目录",
            details={"path": raw},
        ) from exc
    return candidate


__all__ = ["resolve_skill_path"]
