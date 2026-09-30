from typing import List, Dict, Any

class PromptTemplateBuilder:
    """OOP Builder Class for formatting technical RAG Prompts and Guardrails."""
    
    MANUAL_QA_SYSTEM_PROMPT = """You are the Smart Asset Platform Technical AI Assistant. Your job is to assist industrial technicians with machine maintenance, troubleshooting, and operating procedures based strictly on the official technical manual catalog context provided below.

RULES:
1. Base your answer strictly on the provided Context Chunks. Do not hallucinate or use outside assumptions.
2. If the context does not contain enough information to answer the user's question, clearly state: "I cannot find sufficient information in the provided manual context to answer this question."
3. Format technical procedures as clear, numbered step-by-step instructions.
4. Include explicit citations to the manual page number and section title whenever referencing technical data (e.g., [Manual: {{catalog_name}}, Page: {{page_number}}]).
5. Highlight safety warnings (WARNING/CAUTION) if relevant to the maintenance step.

CONTEXT CHUNKS:
{context_str}

USER QUESTION:
{query}

TECHNICAL ANSWER (with citations):
"""

    OUT_OF_SCOPE_DECLINE_MESSAGE = (
        "I am designed specifically for industrial equipment maintenance and manual Q&A. "
        "Your question appears to be outside the scope of asset technical documentation."
    )

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self.MANUAL_QA_SYSTEM_PROMPT

    def build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved vector store chunks into a structured RAG prompt."""
        if not context_chunks:
            context_str = "No relevant context chunks found."
        else:
            formatted_chunks = []
            for i, chunk in enumerate(context_chunks, start=1):
                meta = chunk.get("metadata", {})
                catalog = meta.get("catalog_name", "Unknown Manual")
                page = meta.get("page_number", 1)
                section = meta.get("section_title", "General")
                diagram = "[Visual Diagram Present]" if meta.get("diagram_present") else ""
                
                chunk_header = f"--- Chunk {i} [{catalog} | Page {page} | Section: {section} {diagram}] ---"
                chunk_text = chunk.get("text_content", "").strip()
                formatted_chunks.append(f"{chunk_header}\n{chunk_text}")
                
            context_str = "\n\n".join(formatted_chunks)
            
        return self.system_prompt.format(context_str=context_str, query=query)

    def get_decline_message(self) -> str:
        return self.OUT_OF_SCOPE_DECLINE_MESSAGE

# Global Constants & Helper Functions for Backward Compatibility
OUT_OF_SCOPE_DECLINE_MESSAGE = PromptTemplateBuilder.OUT_OF_SCOPE_DECLINE_MESSAGE

def format_manual_qa_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    builder = PromptTemplateBuilder()
    return builder.build_rag_prompt(query, context_chunks)
