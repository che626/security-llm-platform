"""
可选测试页：单独运行这个页面，可以验证前端是否能调用后端。
运行：
streamlit run frontend/backend_test_page.py
"""

import streamlit as st
import pandas as pd
from frontend_api_client import (
    backend_health,
    backend_chat,
    backend_analyze_log,
    backend_extract_iocs,
    backend_attack_chain,
)

st.set_page_config(page_title="后端 API 测试页", layout="wide")

st.title("前端调用后端 API 测试页")

if st.button("测试后端健康状态"):
    st.json(backend_health())

text = st.text_area(
    "输入日志或安全问题",
    value="""May 11 10:01:02 server sshd[1001]: Failed password for root from 8.8.8.8 port 53422 ssh2
May 11 10:02:22 web nginx: 1.2.3.4 - - "GET /login.php?id=1' OR '1'='1 HTTP/1.1" 200
May 11 10:03:11 firewall: SRC=5.5.5.5 DST=192.168.1.10 DPT=22 ACTION=DENY""",
    height=220,
)

col1, col2, col3, col4 = st.columns(4)

if col1.button("后端日志分析"):
    data = backend_analyze_log(text)
    st.markdown(data.get("report", ""))

if col2.button("后端 IOC 提取"):
    data = backend_extract_iocs(text)
    st.dataframe(pd.DataFrame(data.get("iocs", [])), use_container_width=True)

if col3.button("后端攻击链"):
    data = backend_attack_chain(text)
    st.dataframe(pd.DataFrame(data.get("attack_chain", [])), use_container_width=True)

if col4.button("后端 AI 问答"):
    data = backend_chat(text)
    st.write("回答模式：", data.get("mode"))
    st.markdown(data.get("answer", ""))
