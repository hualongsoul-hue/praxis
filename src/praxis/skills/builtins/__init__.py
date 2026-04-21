"""praxis.skills.builtins — 内置技能目录。

提供框架自带的技能定义，通过 BUILTIN_SKILLS_PATH 供 SkillDiscovery 扫描。
"""

from pathlib import Path

BUILTIN_SKILLS_PATH: str = str(Path(__file__).parent)
