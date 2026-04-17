"""MCP 授权——OAuth 2.0/PKCE 授权流程、Token 管理。"""

import time
from typing import Any

from pydantic import BaseModel, Field

from praxis.models.mcp import MCPElicitationRequest
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.elicitation import ElicitationManager

log = get_logger("tools.mcp.auth")


class OAuthToken(BaseModel):
    """OAuth Token。"""

    access_token: str
    token_type: str = "Bearer"
    refresh_token: str | None = None
    expires_at: float = 0.0
    scope: str = ""


class OAuthConfig(BaseModel):
    """OAuth 配置。"""

    authorization_url: str = ""
    token_url: str = ""
    client_id: str = ""
    redirect_uri: str = "http://localhost:8080/callback"
    scope: str = ""
    use_pkce: bool = True


class MCPAuthManager:
    """MCP 授权管理器。

    管理 OAuth 2.0/PKCE 授权流程和 Token 生命周期。
    """

    def __init__(self, elicitation_mgr: ElicitationManager) -> None:
        self.elicitation_mgr = elicitation_mgr
        self.tokens: dict[str, OAuthToken] = {}
        self.configs: dict[str, OAuthConfig] = {}

    def set_oauth_config(self, server_name: str, config: OAuthConfig) -> None:
        """设置服务器 OAuth 配置。"""
        self.configs[server_name] = config

    def get_token(self, server_name: str) -> OAuthToken | None:
        """获取有效 Token。

        Args:
            server_name: 服务器名称。

        Returns:
            有效 Token 或 None。
        """
        token = self.tokens.get(server_name)
        if token is None:
            return None
        if token.expires_at > 0 and token.expires_at < time.time():
            return None
        return token

    def store_token(self, server_name: str, token: OAuthToken) -> None:
        """存储 Token。"""
        self.tokens[server_name] = token
        log.info("Token 已存储", server=server_name)

    def revoke_token(self, server_name: str) -> None:
        """撤销 Token。"""
        self.tokens.pop(server_name, None)
        log.info("Token 已撤销", server=server_name)

    def get_auth_headers(self, server_name: str) -> dict[str, str]:
        """获取授权头。

        Args:
            server_name: 服务器名称。

        Returns:
            HTTP 授权头字典，无有效 Token 时返回空。
        """
        token = self.get_token(server_name)
        if token is None:
            return {}
        return {"Authorization": f"{token.token_type} {token.access_token}"}

    async def initiate_auth_flow(self, server_name: str) -> bool:
        """启动 OAuth 授权流程。

        通过 Elicitation URL Mode 引导用户完成授权。

        Args:
            server_name: 服务器名称。

        Returns:
            是否成功完成授权。
        """
        config = self.configs.get(server_name)
        if config is None:
            log.warning("无 OAuth 配置", server=server_name)
            return False

        # 构造授权 URL
        import secrets
        state = secrets.token_urlsafe(32)
        auth_url = (
            f"{config.authorization_url}"
            f"?client_id={config.client_id}"
            f"&redirect_uri={config.redirect_uri}"
            f"&response_type=code"
            f"&scope={config.scope}"
            f"&state={state}"
        )

        if config.use_pkce:
            import hashlib
            import base64
            code_verifier = secrets.token_urlsafe(64)
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).rstrip(b"=").decode()
            auth_url += f"&code_challenge={code_challenge}&code_challenge_method=S256"

        # 通过 Elicitation 通知用户访问 URL
        request = MCPElicitationRequest(
            server_name=server_name,
            message=f"请访问以下 URL 完成授权: {auth_url}",
            url=auth_url,
        )
        response = await self.elicitation_mgr.handle_elicitation(request)

        if not response.accepted:
            log.info("用户拒绝授权", server=server_name)
            return False

        # 从 Elicitation 响应中提取授权码
        auth_code = response.data.get("code", "")
        if not auth_code:
            log.warning("未收到授权码", server=server_name)
            return False

        log.info("OAuth 授权码已获取", server=server_name)
        return True

    def is_authenticated(self, server_name: str) -> bool:
        """检查是否已认证。"""
        return self.get_token(server_name) is not None
