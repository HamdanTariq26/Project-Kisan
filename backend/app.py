import asyncio
import json
import threading
from pathlib import Path
from threading import Thread
from typing import Optional

import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pyngrok import ngrok

from .config import get_ngrok_token
from .generation import (
    generate_direct_answer_stream,
    generate_final_answer_stream,
)
from .router import execute_native_tool, generate_tool_call


app = FastAPI(
    title="Project Kisan API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


def sse_event(event_type, data):
    payload = {
        "type": event_type,
        **data,
    }

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def extract_sources(tool_result):
    if not tool_result:
        return []

    sources = []

    for item in tool_result.get("results", []):
        if "source" in item:
            sources.append({
                "type": "internal",
                "title": Path(
                    item.get("source", "Unknown")
                ).name,
                "page": item.get(
                    "page",
                    "N/A",
                ),
            })

        elif "url" in item:
            sources.append({
                "type": "web",
                "title": item.get(
                    "title",
                    "Web source",
                ),
                "url": item.get(
                    "url",
                    "",
                ),
            })

    return sources


def run_agent_stream(query: str):
    yield sse_event(
        "status",
        {
            "message": "Understanding your question..."
        },
    )

    tool_call = generate_tool_call(query)

    if not tool_call:
        yield sse_event(
            "error",
            {
                "message": "Unable to determine how to answer."
            },
        )

        yield sse_event("done", {})
        return

    tool_name = tool_call.get("name")

    if tool_name == "direct":
        yield sse_event(
            "status",
            {
                "message": "Preparing answer..."
            },
        )

        for text in generate_direct_answer_stream(query):
            yield sse_event(
                "token",
                {
                    "content": text
                },
            )

        yield sse_event("done", {})
        return

    if tool_name == "rag_search":
        yield sse_event(
            "status",
            {
                "message": "Searching Project Kisan knowledge base..."
            },
        )

    elif tool_name == "web_search_tool":
        yield sse_event(
            "status",
            {
                "message": "Searching the web..."
            },
        )

    else:
        yield sse_event(
            "error",
            {
                "message": "Unknown tool selected."
            },
        )

        yield sse_event("done", {})
        return

    tool_result = execute_native_tool(tool_call)

    if tool_result.get("status") in [
        "web_search_failed",
        "tool_execution_failed",
        "unknown_tool",
        "invalid_query",
    ]:
        yield sse_event(
            "status",
            {
                "message": "The requested source was unavailable. Preparing an answer from available knowledge..."
            },
        )

    yield sse_event(
        "status",
        {
            "message": "Preparing your answer..."
        },
    )

    for text in generate_final_answer_stream(
        query=query,
        tool_result=tool_result,
    ):
        yield sse_event(
            "token",
            {
                "content": text
            },
        )

    sources = extract_sources(tool_result)

    if sources:
        yield sse_event(
            "sources",
            {
                "sources": sources
            },
        )

    yield sse_event("done", {})


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def run_in_thread():
        try:
            for event in run_agent_stream(request.message):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        except Exception as e:
            err_event = sse_event("error", {"message": str(e)})
            asyncio.run_coroutine_threadsafe(queue.put(err_event), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def event_generator():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    tool_call = generate_tool_call(request.message)

    if not tool_call:
        return {
            "answer": "I was unable to determine how to answer.",
            "tool": None,
            "sources": [],
        }

    if tool_call.get("name") == "direct":
        answer = ""

        for text in generate_direct_answer_stream(
            request.message
        ):
            answer += text

        return {
            "answer": answer,
            "tool": tool_call,
            "sources": [],
        }

    tool_result = execute_native_tool(tool_call)

    answer = ""

    for text in generate_final_answer_stream(
        query=request.message,
        tool_result=tool_result,
    ):
        answer += text

    return {
        "answer": answer,
        "tool": tool_call,
        "sources": extract_sources(tool_result),
    }


def run_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


def start_server():
    server_thread = threading.Thread(
        target=run_server,
        daemon=True,
    )
    server_thread.start()
    return server_thread


def start_ngrok():
    ngrok_token = get_ngrok_token()
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(8000)

    print("PROJECT KISAN API:")
    print(public_url)
    return public_url


if __name__ == "__main__":
    nest_asyncio.apply()
    start_server()
    start_ngrok()
