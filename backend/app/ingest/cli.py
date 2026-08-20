import argparse
import asyncio
import datetime as dt

from app.config import get_settings
from app.db import create_engine, create_session_factory, session_scope
from app.ingest.pipeline import days_to_ingest, ingest_days
from app.ingest.slack import create_reader
from app.providers import create_embedder, create_generator
from app.providers.ollama import create_client


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingest", description="Ingest Slack finance messages")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", type=dt.date.fromisoformat, help="single local date, YYYY-MM-DD")
    group.add_argument("--backfill", type=int, help="number of days back from yesterday")
    parser.add_argument("--channel", help="override SLACK_CHANNEL_ID")
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    channel_id = args.channel or settings.slack_channel_id
    if not channel_id:
        print("no channel configured; set SLACK_CHANNEL_ID or pass --channel")
        return 2

    today = dt.datetime.now(settings.tzinfo).date()
    days = days_to_ingest(today, day=args.date, backfill=args.backfill)

    engine = create_engine(settings)
    ollama = create_client(settings)
    try:
        reader = create_reader(settings.slack_bot_token.get_secret_value())
        embedder = create_embedder(settings, ollama)
        generator = create_generator(settings, ollama)

        async with session_scope(create_session_factory(engine)) as session:
            results = await ingest_days(
                session,
                reader,
                embedder,
                generator,
                channel_id=channel_id,
                days=days,
                tz=settings.tzinfo,
            )
    finally:
        await ollama.aclose()
        await engine.dispose()

    for result in results:
        if result.is_empty:
            print(f"{result.day}: no messages")
        else:
            topics = ", ".join(result.digest_topics) or "-"
            print(
                f"{result.day}: {result.messages} messages, "
                f"{result.chunks} chunks, topics: {topics}"
            )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
