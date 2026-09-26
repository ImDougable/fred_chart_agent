"""Persistent, in-process Python execution tool exposed to the agent via MCP."""
import asyncio
import base64
import contextlib
import io
import traceback

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt

from claude_agent_sdk import create_sdk_mcp_server, tool

# Persistent notebook-like namespace. Lives for the life of the process
# (module-level, so it survives across every tool call / every turn of a
# ClaudeSDKClient session). Single global namespace -- fine for a single-user
# demo, but note it is shared across any concurrent conversations.
_namespace: dict = {}


def _seed_namespace() -> None:
    import io as _io

    import numpy as np
    import pandas as pd

    _namespace.clear()
    _namespace.update(pd=pd, np=np, plt=plt, io=_io)


_seed_namespace()


def reset_namespace() -> None:
    """Reseed the persistent namespace, e.g. when the chat session is reset."""
    _seed_namespace()


def _run_sync(code: str) -> dict:
    """Runs synchronously; called via asyncio.to_thread from the async tool handler."""
    stdout_buf = io.StringIO()
    error_text = None

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, _namespace)
    except Exception:
        error_text = traceback.format_exc()

    stdout_text = stdout_buf.getvalue()

    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)
        images.append(base64.b64encode(buf.read()).decode("ascii"))
        plt.close(fig)  # never re-return the same figure on a later call

    return {"stdout": stdout_text, "error": error_text, "images": images}


@tool(
    "run_python",
    (
        "Execute Python code in a persistent, stateful namespace (like a notebook "
        "cell). Variables, imports, and DataFrames persist across calls within this "
        "conversation, so fetch data once and reuse it across multiple calls instead "
        "of re-fetching. pandas (pd), numpy (np), matplotlib.pyplot (plt), and io are "
        "pre-imported. To produce a chart, create a matplotlib figure (e.g. "
        "plt.plot(...); plt.title(...)) and leave it open -- do NOT call "
        "plt.savefig() to a path and do NOT call plt.show(); any figure left open at "
        "the end of the call is automatically captured and returned as an inline "
        "image. Use print() for any text output you want to see."
    ),
    {"code": str},
)
async def run_python(args: dict) -> dict:
    code = args["code"]
    result = await asyncio.to_thread(_run_sync, code)

    content = []
    if result["stdout"]:
        content.append({"type": "text", "text": result["stdout"]})
    if result["error"]:
        content.append({"type": "text", "text": f"Traceback:\n{result['error']}"})
    if not content:
        content.append({"type": "text", "text": "(no output)"})
    for img_b64 in result["images"]:
        content.append({"type": "image", "data": img_b64, "mimeType": "image/png"})

    return {"content": content, "is_error": bool(result["error"])}


python_server = create_sdk_mcp_server("python_runner", version="1.0.0", tools=[run_python])
