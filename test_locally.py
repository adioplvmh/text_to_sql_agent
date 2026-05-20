"""Quick local test — run a single NL query through the agent using ADK Runner."""
from __future__ import annotations

import asyncio
import sys
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from text_to_sql_adk.agent import create_text_to_sql_agent

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _fmt_event(event) -> None:
    author = getattr(event, "author", "") or ""
    if event.is_final_response():
        if event.content and event.content.parts:
            print(f"\n{BOLD}🤖 Answer:{RESET}")
            print(event.content.parts[0].text)
        return
    if event.content:
        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args_str = str(dict(fc.args) if fc.args else {})[:300]
                print(f"{YELLOW}[{author}] → {fc.name}({args_str}){RESET}")
            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                resp = str(fr.response)[:400] if fr.response else ""
                print(f"{GREEN}[TOOL:{fr.name}]{RESET} {resp}")


async def main(query: str):
    session_id = str(uuid.uuid4())
    agent = create_text_to_sql_agent()
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test_local", session_service=svc)
    await svc.create_session(
        app_name="test_local", user_id="user", session_id=session_id
    )

    print(f"\n{BOLD}📝 Query:{RESET} {query}")
    print("─" * 60)

    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=query)]
    )
    async for event in runner.run_async(
        user_id="user", session_id=session_id, new_message=content
    ):
        _fmt_event(event)

    print("\n" + "─" * 60)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Show me all Black Dress items"
    asyncio.run(main(q))
