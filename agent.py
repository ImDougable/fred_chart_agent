import os
import pathlib

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from tools.python_runner import python_server

ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / "env.txt"


def load_env_file(path: pathlib.Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


_env = load_env_file(ENV_PATH)
ANTHROPIC_API_KEY = _env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

FRED_MCP_URL = "https://fred.kerryback.com/mcp"

SYSTEM_PROMPT = """You are a FRED economic and market data analyst assistant.

Your ONLY source of data is the FRED MCP tools (mcp__fred__*). You must never
fabricate, estimate, or recall economic figures from your own knowledge -- every
number you state must come from a fred_get_observations, fred_series_info,
fred_search_series, fred_list_catalog, or fred_get_vintage call made during this
conversation. If FRED does not have the data, or a tool call fails, say so
plainly rather than guessing.

For any computation, transformation, or chart, use the run_python tool
(mcp__python_runner__run_python). Its Python namespace is persistent across the
whole conversation (like a Jupyter kernel) -- fetch a series once and reuse the
resulting DataFrame/variables for follow-up questions and additional charts
rather than re-fetching. Parse the CSV text returned by fred_get_observations
yourself (e.g. via pandas.read_csv(io.StringIO(...))).

To produce a chart, build it with matplotlib inside run_python and leave the
figure open (do not call plt.savefig(path) or plt.show()) -- any open figure is
automatically captured and returned as an inline image after each run_python
call. Never write files to disk; everything must flow back as tool output.

Cite the FRED series ID(s) and date range you used when you present a number or
a chart.
"""

ALLOWED_TOOLS = [
    "mcp__fred__fred_list_catalog",
    "mcp__fred__fred_search_series",
    "mcp__fred__fred_series_info",
    "mcp__fred__fred_get_observations",
    "mcp__fred__fred_get_vintage",
    "mcp__python_runner__run_python",
]


def build_client() -> ClaudeSDKClient:
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={
            "fred": {"type": "http", "url": FRED_MCP_URL},
            "python_runner": python_server,
        },
        allowed_tools=ALLOWED_TOOLS,
        tools=[],  # disable all built-in tools (Bash/Read/Write/WebSearch/WebFetch/...)
        strict_mcp_config=True,  # ignore any ambient .mcp.json / user MCP config
        permission_mode="bypassPermissions",  # no interactive terminal to approve calls
        env={"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY} if ANTHROPIC_API_KEY else {},
        model="claude-sonnet-5",
    )
    return ClaudeSDKClient(options=options)
