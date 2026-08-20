import datetime as dt

from app.ingest.slack import PAGE_SIZE, SlackReader, create_reader
from app.timeutil import UTC, to_slack_ts

START = dt.datetime(2026, 8, 18, 16, 0, tzinfo=UTC)
END = dt.datetime(2026, 8, 19, 16, 0, tzinfo=UTC)
CHANNEL = "C0TEST"


def ts_at(hour: int, minute: int = 0) -> str:
    return to_slack_ts(dt.datetime(2026, 8, 19, hour, minute, tzinfo=UTC))


class FakeWebClient:
    def __init__(self, pages=None, users=None, permalink="https://slack/p1"):
        self.pages = pages or [{"messages": []}]
        self.users = users or {}
        self.permalink = permalink
        self.history_calls: list[dict] = []
        self.permalink_calls = 0
        self.users_info_calls: list[str] = []

    async def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return self.pages[len(self.history_calls) - 1]

    async def chat_getPermalink(self, **kwargs):
        self.permalink_calls += 1
        return {"permalink": self.permalink}

    async def users_info(self, user):
        self.users_info_calls.append(user)
        if user not in self.users:
            raise RuntimeError("user_not_found")
        return {"user": self.users[user]}


def message(ts: str, **overrides) -> dict:
    return {"ts": ts, "text": "美股收紅", "user": "U1"} | overrides


async def test_fetch_passes_the_utc_window():
    client = FakeWebClient([{"messages": []}])
    await SlackReader(client).fetch_range(CHANNEL, START, END)

    call = client.history_calls[0]
    assert call["channel"] == CHANNEL
    assert call["oldest"] == to_slack_ts(START)
    assert call["latest"] == to_slack_ts(END)
    assert call["limit"] == PAGE_SIZE


async def test_messages_are_returned_in_timestamp_order():
    pages = [{"messages": [message(ts_at(9)), message(ts_at(2))]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert [m.slack_ts for m in result] == [ts_at(2), ts_at(9)]


async def test_pagination_follows_the_cursor():
    pages = [
        {
            "messages": [message(ts_at(2))],
            "response_metadata": {"next_cursor": "abc"},
        },
        {"messages": [message(ts_at(9))], "response_metadata": {"next_cursor": ""}},
    ]
    client = FakeWebClient(pages)
    result = await SlackReader(client).fetch_range(CHANNEL, START, END)

    assert len(result) == 2
    assert client.history_calls[1]["cursor"] == "abc"


async def test_join_and_leave_messages_are_skipped():
    pages = [
        {
            "messages": [
                message(ts_at(2), subtype="channel_join"),
                message(ts_at(3)),
            ]
        }
    ]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert [m.slack_ts for m in result] == [ts_at(3)]


async def test_empty_text_is_skipped():
    pages = [{"messages": [message(ts_at(2), text="   "), message(ts_at(3))]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert len(result) == 1


async def test_message_without_ts_is_skipped():
    pages = [{"messages": [{"text": "美股收紅"}]}]
    assert await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END) == []


async def test_messages_outside_the_window_are_filtered_out():
    outside = to_slack_ts(END)
    pages = [{"messages": [message(outside), message(ts_at(3))]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert [m.slack_ts for m in result] == [ts_at(3)]


async def test_username_wins_over_user_lookup():
    pages = [{"messages": [message(ts_at(2), username="finance-bot")]}]
    client = FakeWebClient(pages)
    result = await SlackReader(client).fetch_range(CHANNEL, START, END)
    assert result[0].author == "finance-bot"
    assert client.users_info_calls == []


async def test_bot_profile_name_is_used_when_present():
    pages = [{"messages": [message(ts_at(2), bot_profile={"name": "digest-bot"})]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert result[0].author == "digest-bot"


async def test_user_names_are_resolved_once_and_cached():
    pages = [{"messages": [message(ts_at(2)), message(ts_at(3))]}]
    client = FakeWebClient(pages, users={"U1": {"real_name": "Marcus"}})
    result = await SlackReader(client).fetch_range(CHANNEL, START, END)

    assert [m.author for m in result] == ["Marcus", "Marcus"]
    assert client.users_info_calls == ["U1"]


async def test_user_lookup_failure_falls_back_to_the_id():
    pages = [{"messages": [message(ts_at(2))]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert result[0].author == "U1"


async def test_message_without_user_has_no_author():
    pages = [{"messages": [{"ts": ts_at(2), "text": "美股收紅"}]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert result[0].author is None


async def test_permalink_is_resolved():
    pages = [{"messages": [message(ts_at(2))]}]
    result = await SlackReader(FakeWebClient(pages)).fetch_range(CHANNEL, START, END)
    assert result[0].permalink == "https://slack/p1"


async def test_permalink_resolution_can_be_disabled():
    pages = [{"messages": [message(ts_at(2))]}]
    client = FakeWebClient(pages)
    reader = SlackReader(client, resolve_permalinks=False)
    result = await reader.fetch_range(CHANNEL, START, END)

    assert result[0].permalink is None
    assert client.permalink_calls == 0


async def test_permalink_failure_degrades_to_none():
    class Failing(FakeWebClient):
        async def chat_getPermalink(self, **kwargs):
            raise RuntimeError("missing scope")

    pages = [{"messages": [message(ts_at(2))]}]
    result = await SlackReader(Failing(pages)).fetch_range(CHANNEL, START, END)
    assert result[0].permalink is None


def test_create_reader_builds_a_slack_backed_reader():
    reader = create_reader("xoxb-test")
    assert isinstance(reader, SlackReader)
    assert reader._client.token == "xoxb-test"
