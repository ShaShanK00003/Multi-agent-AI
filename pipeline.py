"""Command-line entry point for the shared ResearchMind research flow."""

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from research_pipeline import run_research


def run_research_pipeline(topic: str, model_name: str | None = None) -> dict:
    """Backward-compatible CLI wrapper with legacy result keys."""
    result = run_research(topic, model_name)
    return {
        "search_results": result["search_results"],
        "scraped_content": result["scraped_content"],
        "report": result["report"],
        "feedback": result["feedback"],
        "sources": result["sources"],
    }


if __name__ == "__main__":
    run_research_pipeline(input("\nEnter a research topic: "))
