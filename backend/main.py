from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.log_analyzer import analyze_log_text
from src.flow_explainer import explain_flow_summary
from src.soar_generator import generate_soar_yaml, simulate_playbook
from src.simple_rag import retrieve_knowledge
from src.security_chatbot import security_chat_response

from backend.model_client import call_ollama, load_model_config, save_model_config
from backend.security_services import extract_iocs, build_attack_chain, append_json_record, read_json_records

app = FastAPI(
    title="Security LLM Backend API",
    description="基于 DeepSpeed ZeRO 微调与 RAG 增强的智能安全分析交互系统后端服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ok_response(**payload: Any) -> Dict[str, Any]:
    return {"ok": True, **payload}


def error_response(message: str, *, code: str = "runtime_error", detail: Optional[str] = None, **payload: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if detail:
        body["error"]["detail"] = detail
    body.update(payload)
    return body


def safe_call(fn: Callable[[], Dict[str, Any]], message: str, code: str = "runtime_error") -> Dict[str, Any]:
    try:
        return ok_response(**fn())
    except Exception as exc:
        return error_response(message, code=code, detail=str(exc))


class TextRequest(BaseModel):
    text: str = Field("", description="输入文本")


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    use_model: bool = True


class SoarRequest(BaseModel):
    requirement: str


class YamlRequest(BaseModel):
    yaml_text: str


class RagRequest(BaseModel):
    query: str


class ModelConfigRequest(BaseModel):
    mode: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:3b"


@app.get("/")
def root():
    return {
        "system": "基于 DeepSpeed ZeRO 微调与 RAG 增强的智能安全分析交互系统",
        "backend": "FastAPI",
        "status": "running",
        "docs": "/docs",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "security-llm-backend",
        "mode": "local-demo",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/model/config")
def get_model_config():
    return load_model_config()


@app.post("/api/model/config")
def update_model_config(req: ModelConfigRequest):
    config = req.model_dump()
    save_model_config(config)
    return ok_response(config=config)


@app.post("/api/model/test")
def test_model(req: Optional[ModelConfigRequest] = None):
    config = req.model_dump() if req else load_model_config()
    prompt = "请用一句中文说明你在智能安全分析系统中的作用。"
    try:
        answer = call_ollama(
            prompt=prompt,
            model_name=config.get("model_name", "qwen2.5:3b"),
            base_url=config.get("ollama_url", "http://localhost:11434"),
        )
        return ok_response(answer=answer, config=config)
    except Exception as exc:
        return error_response(
            "模型连接失败，前端可以继续使用规则 + RAG 模板模式。",
            code="model_unavailable",
            detail=str(exc),
            config=config,
        )


@app.post("/api/chat")
def chat(req: ChatRequest):
    knowledge = retrieve_knowledge(req.message)
    config = load_model_config()

    system_prompt = f"""
你是一名 SOC 安全运营智能助手，服务于“基于 DeepSpeed ZeRO 微调与 RAG 增强的智能安全分析交互系统”。

请遵守：
1. 只回答防御性安全分析、日志研判、应急响应、RAG、SOAR、DeepSpeed ZeRO、LoRA 微调相关内容。
2. 不提供攻击利用、绕过检测、提权、持久化等攻击性步骤。
3. 回答要结构化，包含：问题理解、知识库参考、分析结论、建议处置。
4. 你不是普通聊天机器人，你是智能安全分析系统中的 AI 安全助手。

RAG 知识库检索结果：
{knowledge}
"""

    if req.use_model and config.get("mode", "ollama") == "ollama":
        try:
            prompt = f"{system_prompt}\n\n用户问题：{req.message}\n\n请给出中文回答："
            answer = call_ollama(
                prompt=prompt,
                model_name=config.get("model_name", "qwen2.5:3b"),
                base_url=config.get("ollama_url", "http://localhost:11434"),
            )
            append_json_record("outputs/backend_chat_history.json", {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": req.message,
                "mode": f"ollama/{config.get('model_name')}",
                "answer": answer[:1000],
            })
            return {
                "ok": True,
                "mode": f"ollama/{config.get('model_name')}",
                "knowledge": knowledge,
                "answer": answer,
            }
        except Exception as exc:
            fallback = security_chat_response(req.message, req.history)
            return {
                "ok": True,
                "mode": "fallback_rule_rag",
                "warning": {
                    "code": "model_fallback",
                    "message": "外部模型调用失败，已回退到规则 + RAG 模板模式。",
                    "detail": str(exc),
                },
                "knowledge": knowledge,
                "answer": fallback,
            }

    fallback = security_chat_response(req.message, req.history)
    return {
        "ok": True,
        "mode": "rule_rag",
        "knowledge": knowledge,
        "answer": fallback,
    }


@app.post("/api/log/analyze")
def analyze_log(req: TextRequest):
    def work():
        result = analyze_log_text(req.text)
        iocs = extract_iocs(req.text)
        chain = build_attack_chain(req.text)
        append_json_record("outputs/backend_history.json", {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module": "日志分析器",
            "summary": result[:500],
        })
        return {"report": result, "iocs": iocs, "attack_chain": chain}

    return safe_call(work, "日志分析失败，请检查输入文本格式。", "log_analysis_failed")


@app.post("/api/flow/explain")
def explain_flow(req: TextRequest):
    return safe_call(
        lambda: {"report": explain_flow_summary(req.text), "iocs": extract_iocs(req.text)},
        "流量摘要解释失败，请检查输入文本格式。",
        "flow_analysis_failed",
    )


@app.post("/api/ioc/extract")
def ioc_extract(req: TextRequest):
    return safe_call(lambda: {"iocs": extract_iocs(req.text)}, "IOC 提取失败。", "ioc_extract_failed")


@app.post("/api/attack-chain/analyze")
def attack_chain(req: TextRequest):
    return safe_call(lambda: {"attack_chain": build_attack_chain(req.text)}, "攻击链分析失败。", "attack_chain_failed")


@app.post("/api/rag/retrieve")
def rag_retrieve(req: RagRequest):
    return safe_call(lambda: {"query": req.query, "knowledge": retrieve_knowledge(req.query)}, "RAG 检索失败。", "rag_failed")


@app.post("/api/soar/generate")
def soar_generate(req: SoarRequest):
    return safe_call(lambda: {"yaml": generate_soar_yaml(req.requirement)}, "SOAR 剧本生成失败。", "soar_generate_failed")


@app.post("/api/soar/simulate")
def soar_simulate(req: YamlRequest):
    return safe_call(lambda: {"result": simulate_playbook(req.yaml_text)}, "SOAR 剧本模拟失败。", "soar_simulate_failed")


@app.get("/api/history")
def get_history():
    return ok_response(
        history=read_json_records("outputs/backend_history.json"),
        chat_history=read_json_records("outputs/backend_chat_history.json"),
    )
