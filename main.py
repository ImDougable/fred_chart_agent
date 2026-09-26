import asyncio
import os
from contextlib import asynccontextmanager

import markdown
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock, ToolResultBlock, UserMessage
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from agent import build_client
from tools.python_runner import reset_namespace

templates = Jinja2Templates(directory="templates")

client: ClaudeSDKClient | None = None
history: list[dict] = []
history_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, history_lock
    history_lock = asyncio.Lock()
    client = build_client()
    await client.connect()
    try:
        yield
    finally:
        await client.disconnect()


app = FastAPI(lifespan=lifespan)


def render_markdown(text: str) -> str:
    return markdown.markdown(text, extensions=["extra", "sane_lists"])


def _extract_images(block_content) -> list[str]:
    """Extract base64 PNG data from a ToolResultBlock's content list.

    The CLI normalizes MCP image content into the Anthropic-API nested
    `source` shape (confirmed empirically), but we also accept a flat
    `data`/`mimeType` shape defensively in case that ever changes.
    """
    images = []
    if isinstance(block_content, list):
        for item in block_content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            if "source" in item and isinstance(item["source"], dict):
                data = item["source"].get("data")
            else:
                data = item.get("data")
            if data:
                images.append(data)
    return images


async def run_turn(user_text: str) -> dict:
    await client.query(user_text)
    reply_text_parts = []
    images_b64 = []
    error = None

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    reply_text_parts.append(block.text)
        elif isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    images_b64.extend(_extract_images(block.content))
        elif isinstance(msg, ResultMessage):
            if msg.is_error:
                error = msg.result or "The agent run ended in an error."

    return {
        "text": "\n\n".join(reply_text_parts).strip(),
        "images": images_b64,
        "error": error,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"history": history, "error": None})


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    error = None
    async with history_lock:
        history.append({"role": "user", "content": message})
        try:
            turn = await run_turn(message)
            history.append(
                {
                    "role": "assistant",
                    "content": turn["text"],
                    "html": render_markdown(turn["text"]) if turn["text"] else "",
                    "images": turn["images"],
                }
            )
            error = turn["error"]
        except Exception as exc:
            error = f"Sorry, the agent ran into a problem: {exc}"

    return templates.TemplateResponse(request, "chat.html", {"history": history, "error": error})


@app.post("/reset")
async def reset():
    global client
    async with history_lock:
        history.clear()
        reset_namespace()
        await client.disconnect()
        client = build_client()
        await client.connect()
    return RedirectResponse(url="/", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8010)))
