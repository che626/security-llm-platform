"""
数据库模型定义

使用 SQLAlchemy ORM 定义系统数据模型。
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default="user")  # admin, judge, research, user
    email = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class AnalysisHistory(Base):
    """分析历史表"""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    module = Column(String(50), nullable=False)  # 日志分析器, 流量摘要解释器, etc.
    event_type = Column(String(100), nullable=False)
    risk_level = Column(String(20), nullable=False)  # 高危, 中高危, 中危, 低危
    summary = Column(Text, nullable=False)
    input_data = Column(Text, nullable=True)
    output_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "模块": self.module,
            "事件类型": self.event_type,
            "风险等级": self.risk_level,
            "摘要": self.summary,
            "时间": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
        }


class Incident(Base):
    """安全事件表"""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(String(20), unique=True, nullable=False, index=True)  # INC-0001
    user_id = Column(Integer, nullable=True)
    event_type = Column(String(100), nullable=False)
    risk_level = Column(String(20), nullable=False)
    source = Column(String(50), nullable=False)  # 来源模块
    status = Column(String(20), nullable=False, default="待处理")  # 待处理, 待确认, 处理中, 已处置, 误报
    summary = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "事件编号": self.incident_id,
            "创建时间": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
            "事件类型": self.event_type,
            "风险等级": self.risk_level,
            "来源": self.source,
            "状态": self.status,
            "摘要": self.summary,
            "处置建议": self.suggestion or "",
        }


class TrainingRun(Base):
    """训练运行记录表"""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    zero_stage = Column(Integer, nullable=False)
    offload_optimizer = Column(Boolean, default=False)
    offload_param = Column(Boolean, default=False)
    adaptive_checkpoint = Column(Boolean, default=False)
    communication_optimization = Column(Boolean, default=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    total_steps = Column(Integer, nullable=True)
    final_loss = Column(Float, nullable=True)
    peak_memory_gb = Column(Float, nullable=True)
    avg_step_time = Column(Float, nullable=True)
    throughput = Column(Float, nullable=True)
    config_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "zero_stage": self.zero_stage,
            "offload_optimizer": self.offload_optimizer,
            "offload_param": self.offload_param,
            "status": self.status,
            "total_steps": self.total_steps,
            "final_loss": self.final_loss,
            "peak_memory_gb": self.peak_memory_gb,
            "avg_step_time": self.avg_step_time,
            "throughput": self.throughput,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Benchmark(Base):
    """基准测试记录表"""
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String(50), unique=True, nullable=False, index=True)
    config_name = Column(String(100), nullable=False)
    zero_stage = Column(Integer, nullable=False)
    offload_optimizer = Column(Boolean, default=False)
    offload_param = Column(Boolean, default=False)
    adaptive_checkpoint = Column(Boolean, default=False)
    communication_optimization = Column(Boolean, default=False)
    status = Column(String(20), nullable=False, default="pending")
    total_steps = Column(Integer, nullable=True)
    avg_step_time = Column(Float, nullable=True)
    throughput = Column(Float, nullable=True)
    peak_memory_gb = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    config_json = Column(JSON, nullable=True)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "benchmark_id": self.benchmark_id,
            "config_name": self.config_name,
            "zero_stage": self.zero_stage,
            "offload_optimizer": self.offload_optimizer,
            "offload_param": self.offload_param,
            "status": self.status,
            "total_steps": self.total_steps,
            "avg_step_time": self.avg_step_time,
            "throughput": self.throughput,
            "peak_memory_gb": self.peak_memory_gb,
            "final_loss": self.final_loss,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
