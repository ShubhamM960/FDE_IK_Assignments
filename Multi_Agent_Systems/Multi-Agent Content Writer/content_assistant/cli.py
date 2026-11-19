"""Command-line chat:  python -m content_assistant.cli [--thread my-thread]"""

import argparse
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from content_assistant.graph import build_graph, make_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Content Assistant (CLI)")
    parser.add_argument("--thread", default=str(uuid.uuid4())[:8], help="conversation thread id")
    args = parser.parse_args()

    graph = build_graph()
    config = make_config(args.thread)
    print(f"Thread: {args.thread}  (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        final = None
        for update in graph.stream({"user_input": user_input}, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "router":
                    print(f"  [router] -> {data['route']}")
                    continue
                for msg in (data or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [{node}] calls {tc['name']}({tc['args']})")
                    elif isinstance(msg, ToolMessage):
                        print(f"  [{node}] {msg.name} returned {len(str(msg.content))} chars")
                    elif isinstance(msg, AIMessage):
                        final = msg
        if final is not None:
            print(f"\nAssistant ({final.name}):\n{final.content}\n")


if __name__ == "__main__":
    main()
