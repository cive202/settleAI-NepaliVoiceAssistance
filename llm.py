"""
llm.py — LLM client with multi-turn conversation history.

Responsibility:
  - Hold the OpenAI-compatible client
  - Maintain conversation history across turns
  - Return Nepali text responses
"""

from openai import OpenAI

from config import (
    GROQ_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_HISTORY_TURNS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    SYSTEM_PROMPT,
)
from perf import timed


class LLM:
    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found — check your .env_local file.")

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=LLM_TIMEOUT_S)
        self._model = model

        self._history = [{"role": "system", "content": system_prompt}]

    def _build_messages(self, user_text: str, context: str | None, clarify: bool) -> list[dict]:
        """Append user_text to history and return the message list to send to the API.

        If `context` is given (even empty), the model is instructed to answer
        using only that context and to say so if the answer isn't in it —
        the retrieved context itself is never stored in history, only the
        plain user_text, so history stays clean across turns.

        If `clarify` is True, the model is asked to politely request that the
        user repeat/rephrase instead of attempting to answer — used when the
        question itself couldn't be confidently understood (e.g. badly garbled
        ASR output), as opposed to a clear question that's simply unanswerable
        from the knowledge base.
        """
        self._history.append({"role": "user", "content": user_text})

        if clarify:
            augmented = (
                "प्रयोगकर्ताको प्रश्न अस्पष्ट वा अपूर्ण देखिन्छ (सायद आवाज पहिचानमा त्रुटि "
                "भएको हुन सक्छ)। प्रश्नको अनुमान नगरी वा जवाफ दिने प्रयास नगरी, विनम्रतापूर्वक "
                "नेपालीमा एक छोटो वाक्यमा प्रयोगकर्तालाई प्रश्न फेरि भन्न वा स्पष्ट पार्न "
                f"अनुरोध गर्नुहोस्।\n\nप्रयोगकर्ताका शब्दहरू: {user_text}"
            )
            return self._history[:-1] + [{"role": "user", "content": augmented}]
        elif context is not None:
            augmented = (
                "प्रश्नमा हिज्जे त्रुटि वा अस्पष्टता भए पनि (जस्तै आवाजबाट लेखिएको हुन सक्छ), "
                "त्यसको सम्भावित आशय बुझ्ने प्रयास गर्नुहोस्। त्यसपछि तलको प्रसंग (context) "
                "मात्र प्रयोग गरेर जवाफ दिनुहोस् — प्रसंग बाहिरको कुनै नयाँ जानकारी नथप्नुहोस्। "
                "प्रसंगमा साँच्चै सान्दर्भिक जानकारी नभएमा मात्र, विनम्रतापूर्वक भन्नुहोस् कि "
                "तपाईंसँग त्यो जानकारी छैन।\n\n"
                f"प्रसंग:\n{context}\n\nप्रश्न: {user_text}"
            )
            return self._history[:-1] + [{"role": "user", "content": augmented}]
        else:
            return self._history

    def get_response(self, user_text: str, context: str | None = None, clarify: bool = False) -> str:
        """
        Sends user_text to the LLM with full conversation history.
        Appends both the user message and assistant reply to history.

        Streams the reply from the API internally (lower time-to-first-token)
        and returns it assembled as a single string once complete.

        Returns:
            Assistant reply as a string.
        """
        return "".join(self.get_response_stream(user_text, context, clarify)).strip()

    def get_response_stream(self, user_text: str, context: str | None = None, clarify: bool = False):
        """Like get_response, but yields text deltas as they arrive from the API.

        History is updated with the full assembled reply once the stream is
        exhausted — a caller that doesn't fully consume the generator leaves
        history unmodified for that turn.
        """
        api_messages = self._build_messages(user_text, context, clarify)
        assistant_parts: list[str] = []

        with timed("llm.chat_completion_stream"):
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                temperature=LLM_TEMPERATURE,
                top_p=1,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    assistant_parts.append(delta)
                    yield delta

        assistant_text = "".join(assistant_parts).strip()
        self._history.append({"role": "assistant", "content": assistant_text})
        self._trim_history()

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Record a user/assistant turn without calling the API.

        Used when a reply was served from the answer cache instead of freshly
        generated — history still needs the turn so later follow-ups have the
        right conversational context.
        """
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep the system prompt plus the last LLM_MAX_HISTORY_TURNS user/assistant pairs.

        Unbounded history means prompt size — and therefore request latency —
        grows every turn for the life of a session.
        """
        turns = self._history[1:]
        max_messages = LLM_MAX_HISTORY_TURNS * 2
        if len(turns) > max_messages:
            self._history = [self._history[0]] + turns[-max_messages:]

    def reset_history(self) -> None:
        """Clears conversation history but keeps the system prompt."""
        self._history = [self._history[0]]

    def last_user_message(self) -> str | None:
        """Most recent user turn already in history, or None if there isn't one."""
        for message in reversed(self._history):
            if message["role"] == "user":
                return message["content"]
        return None
