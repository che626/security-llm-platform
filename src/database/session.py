"""
数据库会话管理

功能：
1. 数据库连接管理
2. 会话创建和关闭
3. 数据库初始化
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

logger = logging.getLogger(__name__)

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/security_llm.db")

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """初始化数据库，创建所有表"""
    try:
        # 确保数据目录存在
        if DATABASE_URL.startswith("sqlite"):
            db_path = DATABASE_URL.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


def get_session() -> Session:
    """获取数据库会话"""
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        session.close()
        raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（上下文管理器）"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作失败: {e}")
        raise
    finally:
        session.close()


def drop_all_tables():
    """删除所有表（危险操作，仅用于测试）"""
    Base.metadata.drop_all(bind=engine)
    logger.info("所有表已删除")


def reset_db():
    """重置数据库（删除并重建所有表）"""
    drop_all_tables()
    init_db()
    logger.info("数据库已重置")
