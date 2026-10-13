from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from src.assistant.assistant_agent import PerceptionAssistant
from src.assistant.tools import AssistantTools
from src.knowledge.retriever import SimpleRetriever
from src.state.event_store import EventStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--db", default="outputs/state/events.db")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--provider", default="rule_based", choices=["rule_based", "ollama"])
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="llama3.2:3b")
    args = ap.parse_args()
    store = EventStore(args.db)
    retriever = SimpleRetriever.from_folder(args.docs)
    assistant = PerceptionAssistant(AssistantTools(store), retriever, args.provider, args.ollama_url, args.ollama_model)
    result = assistant.answer(args.question)
    print(result["answer"])
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source}")
    store.close()


if __name__ == "__main__":
    main()
