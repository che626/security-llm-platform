"""
密码处理模块

功能：
1. 密码哈希（bcrypt）
2. 密码验证
3. 密码强度检查
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PasswordHandler:
    """
    密码处理器

    负责密码的哈希、验证和强度检查。
    """

    def __init__(self):
        logger.info("PasswordHandler 初始化完成")

    def hash_password(self, password: str) -> str:
        """
        对密码进行哈希

        Args:
            password: 明文密码

        Returns:
            哈希后的密码
        """
        try:
            from passlib.context import CryptContext

            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.hash(password)

        except ImportError:
            logger.warning("passlib 未安装，使用简单哈希（仅用于演示）")
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码

        Args:
            plain_password: 明文密码
            hashed_password: 哈希后的密码

        Returns:
            密码是否匹配
        """
        try:
            from passlib.context import CryptContext

            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.verify(plain_password, hashed_password)

        except ImportError:
            logger.warning("passlib 未安装，使用简单验证（仅用于演示）")
            import hashlib
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

    def check_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        检查密码强度

        Args:
            password: 密码

        Returns:
            (是否通过, 失败原因)
        """
        # 长度检查
        if len(password) < 8:
            return False, "密码长度至少 8 位"

        # 大写字母检查
        if not any(c.isupper() for c in password):
            return False, "密码必须包含至少一个大写字母"

        # 小写字母检查
        if not any(c.islower() for c in password):
            return False, "密码必须包含至少一个小写字母"

        # 数字检查
        if not any(c.isdigit() for c in password):
            return False, "密码必须包含至少一个数字"

        # 特殊字符检查
        special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        if not any(c in special_chars for c in password):
            return False, "密码必须包含至少一个特殊字符"

        return True, "密码强度符合要求"

    def generate_temporary_password(self, length: int = 12) -> str:
        """
        生成临时密码

        Args:
            length: 密码长度

        Returns:
            临时密码
        """
        import random
        import string

        # 确保密码包含各种字符
        uppercase = random.choice(string.ascii_uppercase)
        lowercase = random.choice(string.ascii_lowercase)
        digit = random.choice(string.digits)
        special = random.choice("!@#$%^&*")

        # 填充剩余长度
        remaining_length = length - 4
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*"
        remaining = ''.join(random.choices(all_chars, k=remaining_length))

        # 组合并打乱
        password = uppercase + lowercase + digit + special + remaining
        password_list = list(password)
        random.shuffle(password_list)

        return ''.join(password_list)
