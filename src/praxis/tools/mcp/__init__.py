"""praxis.tools.mcp — MCP 协议完整集成：传输、工具、资源、提示、客户端特性。"""

try:
    from praxis.tools.mcp.auth import MCPAuthManager, OAuthConfig, OAuthToken
    from praxis.tools.mcp.connection import MCPConnectionManager
    from praxis.tools.mcp.elicitation import ElicitationManager
    from praxis.tools.mcp.prompts import MCPPromptsBridge
    from praxis.tools.mcp.resources import MCPResourcesBridge
    from praxis.tools.mcp.roots import RootsManager
    from praxis.tools.mcp.sampling import SamplingManager
    from praxis.tools.mcp.tasks import MCPTaskManager
    from praxis.tools.mcp.tools import MCPToolsBridge
    from praxis.tools.mcp.transport import create_transport
except ModuleNotFoundError as exc:
    if exc.name == "mcp" or (exc.name is not None and exc.name.startswith("mcp.")):
        raise RuntimeError("MCP 能力不可用，请安装 praxis[mcp]") from exc
    raise

__all__ = [
    "ElicitationManager",
    "MCPAuthManager",
    "MCPConnectionManager",
    "MCPPromptsBridge",
    "MCPResourcesBridge",
    "MCPTaskManager",
    "MCPToolsBridge",
    "OAuthConfig",
    "OAuthToken",
    "RootsManager",
    "SamplingManager",
    "create_transport",
]
