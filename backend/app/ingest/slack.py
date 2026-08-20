import datetime as dt
from dataclasses import dataclass
from typing import Any, Protocol

from app.timeutil import from_slack_ts, to_slack_ts

PAGE_SIZE = 200
SKIP_SUBTYPES = frozenset(
    {
        "channel_join",
        "channel_leave",
        "channel_topic",
        "channel_purpose",
        "channel_name",
        "message_deleted",
        "thread_broadcast",
    }
)


@dataclass(frozen=True)
class SlackMessageData:
    channel_id: str
    slack_ts: str
    author: str | None
    text: str
    permalink: str | None
    posted_at: dt.datetime


class SlackWebClient(Protocol):
    async def conversations_history(self, **kwargs: Any) -> Any: ...

    async def chat_getPermalink(self, **kwargs: Any) -> Any: ...

    async def users_info(self, **kwargs: Any) -> Any: ...


class SlackReader:
    def __init__(self, client: SlackWebClient, *, resolve_permalinks: bool = True) -> None:
        self._client = client
        self._resolve_permalinks = resolve_permalinks
        self._user_names: dict[str, str] = {}

    async def fetch_range(
        self, channel_id: str, start: dt.datetime, end: dt.datetime
    ) -> list[SlackMessageData]:
        collected: list[SlackMessageData] = []
        cursor: str | None = None

        while True:
            params = {
                "channel": channel_id,
                "oldest": to_slack_ts(start),
                "latest": to_slack_ts(end),
                "inclusive": True,
                "limit": PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            response = await self._client.conversations_history(**params)

            for raw in response.get("messages", []):
                message = await self._build(channel_id, raw, start, end)
                if message is not None:
                    collected.append(message)

            cursor = response.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break

        return sorted(collected, key=lambda message: message.slack_ts)

    async def _build(
        self, channel_id: str, raw: dict, start: dt.datetime, end: dt.datetime
    ) -> SlackMessageData | None:
        if raw.get("subtype") in SKIP_SUBTYPES:
            return None
        text = (raw.get("text") or "").strip()
        slack_ts = raw.get("ts")
        if not text or not slack_ts:
            return None

        posted_at = from_slack_ts(slack_ts)
        if not start <= posted_at < end:
            return None

        return SlackMessageData(
            channel_id=channel_id,
            slack_ts=slack_ts,
            author=await self._author(raw),
            text=text,
            permalink=await self._permalink(channel_id, slack_ts),
            posted_at=posted_at,
        )

    async def _author(self, raw: dict) -> str | None:
        if username := raw.get("username"):
            return username
        if name := (raw.get("bot_profile") or {}).get("name"):
            return name

        user_id = raw.get("user")
        if not user_id:
            return None
        if user_id not in self._user_names:
            try:
                response = await self._client.users_info(user=user_id)
                profile = response.get("user", {})
                self._user_names[user_id] = (
                    profile.get("real_name") or profile.get("name") or user_id
                )
            except Exception:
                self._user_names[user_id] = user_id
        return self._user_names[user_id]

    async def _permalink(self, channel_id: str, slack_ts: str) -> str | None:
        if not self._resolve_permalinks:
            return None
        try:
            response = await self._client.chat_getPermalink(channel=channel_id, message_ts=slack_ts)
            return response.get("permalink")
        except Exception:
            return None


def create_reader(token: str) -> SlackReader:
    from slack_sdk.web.async_client import AsyncWebClient

    return SlackReader(AsyncWebClient(token=token))
