from __future__ import annotations

import argparse
import json

from support_agent.agent import SupportAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer support SOP-grounded agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask one support question")
    ask_parser.add_argument("message", help="Customer message")
    ask_parser.add_argument("--json", action="store_true", help="Print full JSON response")

    subparsers.add_parser("chat", help="Start an interactive chat loop")
    args = parser.parse_args()

    agent = SupportAgent()
    if args.command == "ask":
        response = agent.ask(args.message)
        if args.json:
            print(response.model_dump_json(indent=2))
        else:
            print(response.answer)
            print("\nRoute:")
            print(json.dumps(response.route.model_dump(), indent=2))
        return

    print("Customer Support Agent. Type 'exit' to quit.")
    while True:
        message = input("\nCustomer> ").strip()
        if message.casefold() in {"exit", "quit"}:
            break
        if not message:
            continue
        response = agent.ask(message)
        print(f"\nAgent> {response.answer}")
        print(f"\nRoute> {response.route.model_dump_json()}")


if __name__ == "__main__":
    main()
