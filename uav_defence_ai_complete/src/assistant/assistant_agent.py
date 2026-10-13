from __future__ import annotations

import os
from typing import Any

from .llm_client import OllamaClient
from .tools import AssistantTools
from ..knowledge.retriever import SimpleRetriever


class PerceptionAssistant:
    """Evidence-grounded operator assistant.

    It reads structured perception state and project documents. It does not
    make engagement or control decisions.
    """

    def __init__(self, tools: AssistantTools, retriever: SimpleRetriever | None = None, provider: str = "rule_based", url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.tools = tools
        self.retriever = retriever
        self.provider = provider
        self.llm = OllamaClient(url, model)

    def _rule_answer(self, question: str) -> str:
        q = question.lower()
        if "summary" in q or "current scene" in q:
            s = self.tools.get_scene_summary()
            classes = ", ".join(f"{k}: {v}" for k, v in s["classes"].items()) or "no class records"
            return f"Current scene state: {s['active_track_count']} active track(s). Class counts: {classes}. Recent recorded events: {len(s['recent_events'])}."
        if "low confidence" in q:
            tracks = [t for t in self.tools.get_active_tracks() if float(t["confidence"]) < 0.60]
            return "Low-confidence tracks: " + (", ".join(str(t["track_id"]) for t in tracks) if tracks else "none recorded") + "."
        if "event" in q or "recent" in q:
            events = self.tools.get_recent_events(10)
            return "Recent events: " + ("; ".join(f"Track {e['track_id']}: {e['event']} ({e['confidence']:.2f})" for e in events) if events else "none recorded") + "."
        return "I can summarize the current scene, inspect tracks, list recent events, identify low-confidence records, or search approved project documents."

    def answer(self, question: str) -> dict[str, Any]:
        context = self.tools.get_scene_summary()
        docs = self.retriever.search(question, top_k=3) if self.retriever else []
        sources = [d.source for d in docs]
        prompt = (
            "You are a read-only perception assistant. Use only the supplied structured state and document excerpts. "
            "Do not invent facts. Do not make operational engagement decisions. Clearly distinguish missing evidence.\n\n"
            f"QUESTION:\n{question}\n\nSTATE:\n{context}\n\nDOCUMENT SOURCES:\n{sources}\n"
        )
        if self.provider == "ollama" and self.llm.available():
            try:
                answer = self.llm.generate(prompt)
            except Exception:
                answer = self._rule_answer(question)
        else:
            answer = self._rule_answer(question)
        return {"answer": answer, "sources": sources, "state": context}
