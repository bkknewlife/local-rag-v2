"""Assemble the self-correcting RAG LangGraph."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from rag_eval.graph.state import GraphState
from rag_eval.graph import nodes, edges


def build_rag_graph() -> StateGraph:
    """Build and compile the full agentic RAG evaluation graph.

    Graph topology::

        retrieve --(web_search on?)--> web_search -> grade_documents
                 \\--(off)-----------> grade_documents
        grade_documents --(relevant)--> generate
                        \\--(empty)----> rewrite -> retrieve
        generate -> check_hallucination --(grounded)--> check_usefulness
                                        \\--(not)-----> generate
        check_usefulness --(useful)----> END
                         \\--(not)------> rewrite -> retrieve
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("check_hallucination", nodes.check_hallucination)
    graph.add_node("check_usefulness", nodes.check_usefulness)
    graph.add_node("rewrite", nodes.rewrite_query)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        edges.route_after_retrieve,
        {"web_search": "web_search", "grade_documents": "grade_documents"},
    )
    graph.add_edge("web_search", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        edges.route_after_grading,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        edges.route_after_hallucination,
        {"check_useful": "check_usefulness", "regenerate": "generate"},
    )

    graph.add_conditional_edges(
        "check_usefulness",
        edges.route_after_usefulness,
        {"end": END, "rewrite": "rewrite"},
    )

    return graph.compile()
