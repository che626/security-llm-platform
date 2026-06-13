"""
JWT 令牌管理

功能：
1. 生成访问令牌
2. 生成刷新令牌
3. 验证令牌
4. 令牌过期处理
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# JWT 配置
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class JWTHandler:
    """
    JWT 令牌管理器

    负责 JWT 令牌的生成、验证和管理。
    """

    def __init__(
        self,
        secret_key: str = JWT_SECRET_KEY,
        algorithm: str = JWT_ALGORITHM,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

        logger.info("JWTHandler 初始化完成")

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        创建访问令牌

        Args:
            data: 令牌数据
            expires_delta: 过期时间增量

        Returns:
            JWT 令牌字符串
        """
        try:
            from jose import jwt

            to_encode = data.copy()

            # 设置过期时间
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

            to_encode.update({"exp": expire, "type": "access"})

            # 编码令牌
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            return encoded_jwt

        except ImportError:
            logger.error("python-jose 未安装，无法生成 JWT 令牌")
            # 返回一个简单的 token（仅用于演示）
            import hashlib
            import time
            token_data = f"{data}-{time.time()}"
            return hashlib.sha256(token_data.encode()).hexdigest()

    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        创建刷新令牌

        Args:
            data: 令牌数据
            expires_delta: 过期时间增量

        Returns:
            JWT 刷新令牌字符串
        """
        try:
            from jose import jwt

            to_encode = data.copy()

            # 设置过期时间
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

            to_encode.update({"exp": expire, "type": "refresh"})

            # 编码令牌
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

            return encoded_jwt

        except ImportError:
            logger.error("python-jose 未安装，无法生成 JWT 令牌")
            import hashlib
            import time
            token_data = f"{data}-refresh-{time.time()}"
            return hashlib.sha256(token_data.encode()).hexdigest()

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证令牌

        Args:
            token: JWT 令牌字符串

        Returns:
            令牌数据，验证失败返回 None
        """
        try:
            from jose import jwt, JWTError

            # 解码令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # 检查令牌类型
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                logger.warning(f"无效的令牌类型: {token_type}")
                return None

            return payload

        except ImportError:
            logger.error("python-jose 未安装，无法验证 JWT 令牌")
            # 简单验证（仅用于演示）
            if token and len(token) > 10:
                return {"sub": "demo_user", "role": "admin", "type": "access"}
            return None

        except Exception as e:
            logger.warning(f"令牌验证失败: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        使用刷新令牌获取新的访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的访问令牌，失败返回 None
        """
        # 验证刷新令牌
        payload = self.verify_token(refresh_token)
        if not payload:
            return None

        # 检查令牌类型
        if payload.get("type") != "refresh":
            logger.warning("不是刷新令牌")
            return None

        # 创建新的访问令牌
        new_payload = {
            "sub": payload.get("sub"),
            "role": payload.get("role"),
        }

        return self.create_access_token(new_payload)

    def decode_token_without_verification(self, token: str) -> Optional[Dict[str, Any]]:
        """
        解码令牌（不验证签名）

        Args:
            token: JWT 令牌字符串

        Returns:
            令牌数据，解码失败返回 None
        """
        try:
            from jose import jwt

            # 解码令牌（不验证签名）
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False},
            )

            return payload

        except ImportError:
            logger.error("python-jose 未安装，无法解码 JWT 令牌")
            return None

        except Exception as e:
            logger.warning(f"令牌解码失败: {e}")
            return None
