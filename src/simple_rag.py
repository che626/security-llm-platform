def retrieve_knowledge(query: str, kb_path="data/knowledge_base.md"):

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = f.read()

    sections = kb.split("## ")

    hits = []

    for sec in sections:

        if not sec.strip():
            continue

        if any(word in sec for word in query.split()):
            hits.append(sec.strip())

    return "\n\n".join(hits[:3]) or "未检索到相关知识。"