from typing import Any
import groq
from groq import AsyncGroq

from app.config import settings
from app.core.logging import logger

FINANCIAL_SYSTEM_PROMPT = """You are a Senior FinTech Financial Analyst.
Your role is to answer questions regarding company filings (10-K, 10-Q, earnings reports, balance sheets) with 100% audit precision.

STRICT OPERATING RULES:
1. ONLY answer using facts, figures, and tables explicitly stated in the provided context.
2. CITATIONS: You MUST cite every financial metric, balance, and statement with its source page, e.g., [Page 45] or [Source 1, Page 12].
3. ZERO HALLUCINATION: If the requested figure or answer is NOT directly stated in the context, explicitly respond: "The provided financial document does not contain this information." Do NOT calculate, guess, extrapolate, or use outside knowledge.
4. NUMERIC ACCURACY: Preserve exact currency signs, negative amounts in parentheses (e.g. $(1,250)M), units (in millions, thousands), and footnote qualifications.
5. CONCISENESS: Deliver concise, professional executive summaries followed by exact figures and citations.
"""


class GroqLLMService:
    """
    High-performance Groq LLM inference service optimized for financial RAG reasoning.
    Runs fast inference (e.g. llama-3.3-70b-versatile or llama-3.1-8b-instant) with zero temperature.
    """

    def __init__(
        self,
        api_key: str = settings.GROQ_API_KEY,
        default_model: str = settings.GROQ_MODEL,
        temperature: float = settings.GROQ_TEMPERATURE,
        max_tokens: int = settings.GROQ_MAX_TOKENS,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._async_client: AsyncGroq | None = None

    def _get_client(self) -> AsyncGroq:
        if self._async_client is None:
            self._async_client = AsyncGroq(api_key=self.api_key)
        return self._async_client

    async def generate_financial_answer(
        self,
        query: str,
        context: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Executes grounded financial Q&A inference against Groq.
        """
        target_model = model or self.default_model
        target_temp = temperature if temperature is not None else self.temperature
        target_tokens = max_tokens or self.max_tokens

        # Check if API key is provided
        if not self.api_key or self.api_key.strip() == "":
            logger.warning("GROQ_API_KEY is not set. Returning preview mode response.")
            return {
                "answer": (
                    "[DRY-RUN / PREVIEW MODE: Please configure GROQ_API_KEY in your .env or environment].\n\n"
                    f"Reranked financial context retrieved successfully ({len(context.split())} words). "
                    "Set GROQ_API_KEY to receive real-time answers from Llama-3.3-70B on Groq."
                ),
                "model": f"{target_model} (offline-preview)",
                "usage": {"prompt_tokens": len(context) // 4, "completion_tokens": 0}
            }

        client = self._get_client()

        user_content = (
            f"FINANCIAL CONTEXT EXCERPTS:\n"
            f"============================\n"
            f"{context}\n"
            f"============================\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Provide an audit-ready financial answer with page citations based strictly on the excerpts above."
        )

        try:
            chat_completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=target_model,
                temperature=target_temp,
                max_tokens=target_tokens,
            )

            response_text = chat_completion.choices[0].message.content or ""
            usage_info = {
                "prompt_tokens": chat_completion.usage.prompt_tokens if chat_completion.usage else len(context) // 4,
                "completion_tokens": chat_completion.usage.completion_tokens if chat_completion.usage else len(response_text) // 4,
            }

            return {
                "answer": response_text,
                "model": target_model,
                "usage": usage_info
            }

        except Exception as e:
            logger.error(f"Groq API completion error: {str(e)}", exc_info=True)
            raise e


llm_service = GroqLLMService()

