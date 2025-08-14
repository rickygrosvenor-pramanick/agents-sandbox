"""
MCP server using the `mcp` Python library only (no FastAPI, no uvicorn).
Exposes a single tool: `business_analyst_story_generator` via MCP stdio transport.

Run with:
  PYTHONUNBUFFERED=1 python -u mcp_rag_server.py

Notes:
- Designed for MCP-compatible clients over stdio (not HTTP).
- Fast initialize: all RAG/LLM/Jira imports happen inside the tool.
- Optional Jira creation supported via environment:
    JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
  Pass create_jira=true to have the tool create a Story in Jira.
- Tool parameters:
    prompt: str (required)
    create_jira: bool = False
    project_key: str | None = None
    labels: list[str] | None = None
"""

from mcp.server.fastmcp import FastMCP
import sys
import time
import logging
from typing import Optional

# Log to stderr ONLY — stdout must remain clean for JSON protocol
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Keep stdout line-buffered just in case
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

app = FastMCP(name="agent-001-rag-user-story-generator")

@app.tool("business_analyst_story_generator")
def business_analyst_story_generator(
    prompt: str,
    create_jira: bool = False,
    project_key: Optional[str] = None,
    labels: Optional[list[str]] = None,
) -> str:
    """Generate a detailed BA user story from a high-level prompt using RAG.

    Steps:
      1) Retrieve context from vector store
      2) Build enhanced prompt
      3) Generate final story with LLM
      4) (Optional) Create a Jira Story and append the issue key
    """
    try:
        logging.info("Tool invoked: business_analyst_story_generator")
        start_ts = time.time()

        # Heavy stuff only when the tool is invoked
        from retriever import query_vector_store
        from agent import generate_enhanced_prompt
        from utils.llm import get_response, llm

        # 1) Retrieve context
        retrieved_results = query_vector_store(query=prompt, n_results=3)
        retrieved_docs = []
        if retrieved_results and retrieved_results.get("documents"):
            retrieved_docs = retrieved_results["documents"][0]
        logging.info(f"Retrieved {len(retrieved_docs)} docs")

        # 2) Build enhanced prompt
        enhanced_prompt = generate_enhanced_prompt(prompt, retrieved_docs)
        logging.info(f"Enhanced prompt length: {len(enhanced_prompt)}")

        # 3) Generate with LLM
        final_response = get_response(user_prompt=enhanced_prompt, llm=llm)
        logging.info(f"LLM completed in {time.time()-start_ts:.2f}s")

        # 4) Optional Jira creation
        if create_jira:
            try:
                logging.info("Creating Jira Story...")
                # Import Jira client lazily
                from utils.jira_client import create_story

                # Construct a concise summary (Jira limit ~255 chars)
                summary = (final_response.splitlines()[0] or prompt).strip()
                if not summary:
                    summary = prompt.strip()
                summary = summary[:250]

                issue = create_story(
                    summary=summary,
                    description=final_response,
                    project_key=project_key,
                    labels=labels or [],
                )
                if isinstance(issue, dict) and issue.get("key"):
                    final_response += f"\n\nJira issue created: {issue['key']}"
                    logging.info(f"Jira created: {issue['key']}")
                else:
                    final_response += "\n\nJira creation returned unexpected response."
                    logging.warning("Jira creation response had no key")
            except Exception as je:
                logging.exception("Jira creation failed")
                final_response += f"\n\nJira creation failed: {je}"

        return final_response

    except Exception as e:
        logging.exception("Error in BA story generation")
        return f"Error during BA story generation: {e}"


def main():
    logging.info("Starting MCP loop")
    app.run("stdio")

if __name__ == "__main__":
    main()
