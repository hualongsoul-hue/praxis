"""内置工具统一注册。

将所有内置工具注册到指定的 ToolRegistry 中。
"""

from praxis.persistence.store import PersistenceStore
from praxis.tools.builtins.autonomy import ask_user, submit_result, update_notes, update_plan
from praxis.tools.builtins.file_ops import edit_file, list_dir, read_file, write_file
from praxis.tools.builtins.network import web_fetch, web_search
from praxis.tools.builtins.search import code_search, find_by_name, grep_search
from praxis.tools.builtins.shell import run_command
from praxis.tools.builtins.system import get_system_info, sleep
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry


def register_builtins(
    registry: ToolRegistry,
    sandbox: ToolPolicy,
    store: PersistenceStore | None = None,
) -> None:
    """注册所有内置工具到注册表。

    Args:
        registry: 工具注册表。
        sandbox: 沙箱实例（文件/Shell 工具绑定路径限制）。
        store: 持久化存储实例（自治管理工具使用），可选。
    """
    registry.register(read_file.DEFINITION, read_file.create_handler(sandbox))
    registry.register(write_file.DEFINITION, write_file.create_handler(sandbox))
    registry.register(edit_file.DEFINITION, edit_file.create_handler(sandbox))
    registry.register(list_dir.DEFINITION, list_dir.create_handler(sandbox))

    registry.register(grep_search.DEFINITION, grep_search.create_handler(sandbox))
    registry.register(find_by_name.DEFINITION, find_by_name.create_handler(sandbox))
    registry.register(code_search.DEFINITION, code_search.create_handler(sandbox))

    registry.register(run_command.DEFINITION, run_command.create_handler(sandbox))

    registry.register(web_fetch.DEFINITION, web_fetch.create_handler(sandbox))
    registry.register(web_search.DEFINITION, web_search.create_handler(sandbox))

    registry.register(get_system_info.DEFINITION, get_system_info.handle)
    registry.register(sleep.DEFINITION, sleep.handle)

    registry.register(update_plan.DEFINITION, update_plan.create_handler(store))
    registry.register(update_notes.DEFINITION, update_notes.create_handler(store))
    registry.register(ask_user.DEFINITION, ask_user.handle)
    registry.register(submit_result.DEFINITION, submit_result.handle)
