"""
slack_bot/listener.py — Live-ingests messages from named Slack channels into
the RAG store over a Socket Mode connection.

Requires no public URL (unlike the Events API): the bot opens an outbound
websocket to Slack and receives events over it. Each qualifying message
becomes its own retrievable page via RAGService.add_texts(), keyed by
channel+timestamp, so the voice agent can answer questions grounded in
recent Slack traffic without a manual /api/rag/ingest call.

Requires SLACK_BOT_TOKEN (xoxb-..., scopes: channels:history, channels:read)
and SLACK_APP_TOKEN (xapp-..., scope: connections:write) — see config.py.
"""

import asyncio
import logging
from collections.abc import Callable

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from config import SLACK_APP_TOKEN, SLACK_BOT_TOKEN, SLACK_CHANNELS
from rag import RAGService

log = logging.getLogger(__name__)

# Message subtypes that aren't a person saying something new — edits,
# deletes, and channel-membership churn shouldn't become knowledge-base pages.
_IGNORED_SUBTYPES = {
    "message_changed",
    "message_deleted",
    "message_replied",
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "bot_message",
}


class SlackListener:
    """Owns the Socket Mode connection and the channel allow-list."""

    def __init__(self, rag: RAGService, on_ingest: Callable[[], None] | None = None):
        self._rag = rag
        self._on_ingest = on_ingest
        self._app = AsyncApp(token=SLACK_BOT_TOKEN)
        self._handler = AsyncSocketModeHandler(self._app, SLACK_APP_TOKEN)
        self._channel_ids: set[str] = set()
        self._user_names: dict[str, str] = {}
        self._app.event("message")(self._on_message)

    async def start(self) -> None:
        self._channel_ids = await self._resolve_channel_ids(SLACK_CHANNELS)
        if not self._channel_ids:
            log.warning("Slack listener: no configured channel resolved to an ID; not starting.")
            return
        log.info("Slack listener: watching channels %s", self._channel_ids)
        await self._handler.connect_async()

    async def stop(self) -> None:
        await self._handler.disconnect_async()

    async def _resolve_channel_ids(self, wanted: list[str]) -> set[str]:
        """Match configured names (e.g. "#support") or raw IDs against the
        channels the bot can see, via conversations.list. Already-an-ID
        entries are kept as-is without a lookup.

        Only looks up public channels — listing private ones requires the
        groups:read scope, which isn't part of the default setup (see
        config.py's SLACK_BOT_TOKEN comment). Configure private channels by
        their ID directly in SLACK_CHANNELS if needed, and add groups:history
        too, so the bot can actually read messages there.
        """
        by_id = {w for w in wanted if w.startswith("C")}
        by_name = {w.lstrip("#") for w in wanted if not w.startswith("C")}
        resolved = set(by_id)
        if not by_name:
            return resolved

        cursor = None
        while True:
            resp = await self._app.client.conversations_list(
                types="public_channel", cursor=cursor, limit=200
            )
            for channel in resp["channels"]:
                if channel["name"] in by_name:
                    resolved.add(channel["id"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return resolved

    async def _resolve_user_name(self, user_id: str) -> str:
        if user_id not in self._user_names:
            try:
                resp = await self._app.client.users_info(user=user_id)
                profile = resp["user"]
                self._user_names[user_id] = profile.get("real_name") or profile.get("name") or user_id
            except Exception:
                self._user_names[user_id] = user_id
        return self._user_names[user_id]

    async def _on_message(self, event: dict) -> None:
        channel = event.get("channel")
        if channel not in self._channel_ids:
            return
        if event.get("subtype") in _IGNORED_SUBTYPES or event.get("bot_id"):
            return
        text = (event.get("text") or "").strip()
        if not text:
            return

        user = event.get("user")
        author = await self._resolve_user_name(user) if user else "someone"
        ts = event.get("ts", "")
        source = f"slack:{channel}:{ts}"
        page_content = f"{author} said in Slack: {text}"

        try:
            # RAGService.add_texts is sync (blocking Ollama embedding call) —
            # run it off the event loop so it doesn't stall other Slack events.
            await asyncio.to_thread(self._rag.add_texts, [(page_content, source)])
        except Exception:
            log.exception("Slack listener: failed to ingest message %s", source)
            return

        if self._on_ingest:
            self._on_ingest()
