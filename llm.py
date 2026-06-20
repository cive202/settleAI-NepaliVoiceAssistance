"""
llm.py — LLM client with multi-turn conversation history.

Responsibility:
  - Hold the OpenAI-compatible client
  - Maintain conversation history across turns
  - Return Nepali text responses
"""

from openai import OpenAI

from config import (
    API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)


class LLM:
    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if not api_key:
            raise EnvironmentError("API_KEY not found — check your .env_local file.")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

        self._history = [{"role": "system", "content": system_prompt}]

    def get_response(self, user_text: str) -> str:
        """
        Sends user_text to the LLM with full conversation history.
        Appends both the user message and assistant reply to history.

        Returns:
            Assistant reply as a string.
        """
        self._history.append({"role": "user", "content": user_text})

        print("Getting response...")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            temperature=LLM_TEMPERATURE,
            top_p=1,
            max_tokens=LLM_MAX_TOKENS,
            stream=False,
        )

        assistant_text = response.choices[0].message.content.strip()

        # Print reasoning if the model exposes it (e.g. reasoning models)
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            print(f"\n Reasoning:\n{reasoning}\n")

        self._history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset_history(self) -> None:
        """Clears conversation history but keeps the system prompt."""
        self._history = [self._history[0]]
