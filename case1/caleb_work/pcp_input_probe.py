from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIB = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

from utcxchangelib import XChangeClient
from utcxchangelib import service_pb2


KEYWORD_PATTERNS = (
    re.compile(r"(^|[^a-z0-9])(risk[_ -]?free)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(rate)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(rf)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(yield)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(expiry)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(expiration)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(expire)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(maturity)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(time[_ -]?to[_ -]?expiry)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(time[_ -]?to[_ -]?maturity)($|[^a-z0-9])", re.IGNORECASE),
    re.compile(r"(^|[^a-z0-9])(tenor)($|[^a-z0-9])", re.IGNORECASE),
)


@dataclass(frozen=True)
class MatchRow:
    source: str
    path: str
    detail: str


def matches_keyword(text: str) -> bool:
    return any(pattern.search(text) for pattern in KEYWORD_PATTERNS)


def find_schema_matches() -> list[MatchRow]:
    rows: list[MatchRow] = []
    seen_messages: set[str] = set()

    def walk_descriptor(descriptor, prefix: str) -> None:
        if descriptor.full_name in seen_messages:
            return
        seen_messages.add(descriptor.full_name)
        for field in descriptor.fields:
            field_path = f"{prefix}.{field.name}" if prefix else field.name
            if matches_keyword(field.name):
                rows.append(
                    MatchRow(
                        source="schema",
                        path=field_path,
                        detail=f"type={field.type} label={field.label}",
                    )
                )
            if field.message_type is not None:
                walk_descriptor(field.message_type, field_path)

    walk_descriptor(service_pb2.ExchangeMessageToClient.DESCRIPTOR, "ExchangeMessageToClient")
    walk_descriptor(service_pb2.NewsEvent.DESCRIPTOR, "NewsEvent")
    return rows


def find_code_matches() -> list[MatchRow]:
    rows: list[MatchRow] = []
    bot_path = REPO_ROOT / "case1" / "caleb_work" / "codex_att1.py"
    lines = bot_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if matches_keyword(line):
            rows.append(
                MatchRow(
                    source="code",
                    path=f"{bot_path}:{index}",
                    detail=line.strip(),
                )
            )
    return rows


def flatten_message(message: Message) -> dict[str, Any]:
    return MessageToDict(message, preserving_proto_field_name=True, always_print_fields_with_no_presence=False)


def walk_dict(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if matches_keyword(str(key)):
                rows.append((path, json.dumps(value, sort_keys=True)))
            rows.extend(walk_dict(value, path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            rows.extend(walk_dict(value, f"{prefix}[{index}]"))
    elif isinstance(obj, str):
        if matches_keyword(obj):
            rows.append((prefix, obj))
    return rows


class PCPInputProbe(XChangeClient):
    def __init__(self, host: str, username: str, password: str, output_path: Path, runtime_sec: float):
        super().__init__(host, username, password, silent=False)
        self.output_path = output_path
        self.runtime_sec = runtime_sec
        self.started_at = time.monotonic()
        self.hit_count = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.output_path.open("a", encoding="utf-8", buffering=1)

    def log(self, kind: str, **payload) -> None:
        row = {
            "ts_wall": time.time(),
            "ts_mono": time.monotonic(),
            "kind": kind,
            "payload": payload,
        }
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.handle.flush()

    async def process_message(self, msg) -> None:
        msg_type = msg.WhichOneof("body") if hasattr(msg, "WhichOneof") else None
        if msg_type is not None:
            raw = flatten_message(msg)
            rows = walk_dict(raw)
            if rows:
                self.hit_count += len(rows)
                self.log(
                    "live_keyword_match",
                    msg_type=msg_type,
                    index=getattr(msg, "index", None),
                    matches=[{"path": path, "value": value} for path, value in rows],
                )
                print(f"[MATCH] type={msg_type} index={getattr(msg, 'index', None)} matches={len(rows)}")
        await super().process_message(msg)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: str | None = None) -> None:
        return

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        return

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        return

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        return

    async def bot_handle_book_update(self, symbol: str) -> None:
        return

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        return

    async def bot_handle_news(self, news_release: dict):
        if any(matches_keyword(str(part)) for part in news_release.values()):
            self.log("live_news_keyword_match", news_release=news_release)
            print(f"[MATCH] news={news_release}")

    async def stop_after_timeout(self) -> None:
        await asyncio.sleep(self.runtime_sec)
        self.log("summary", runtime_sec=self.runtime_sec, hit_count=self.hit_count)
        print(f"[DONE] runtime_sec={self.runtime_sec} hit_count={self.hit_count} output={self.output_path}")
        raise SystemExit(0)

    async def start(self) -> None:
        asyncio.create_task(self.stop_after_timeout())
        await self.connect()


def default_output_path() -> Path:
    timestamp = time.strftime("pcp_probe_%Y%m%d_%H%M%S.jsonl")
    return REPO_ROOT / "case1" / "caleb_work" / "run_logs" / timestamp


def print_rows(title: str, rows: list[MatchRow]) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("none")
        return
    for row in rows:
        print(f"{row.source}: {row.path} -> {row.detail}")


async def run_live_probe(args) -> None:
    output_path = Path(args.output).expanduser().resolve()
    probe = PCPInputProbe(args.host, args.username, args.password, output_path, args.seconds)
    print(f"[LIVE] host={args.host} runtime_sec={args.seconds} output={output_path}")
    await probe.start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether PCP inputs such as risk-free rate or expiry are present in the client schema or live stream."
    )
    parser.add_argument("--live", action="store_true", help="Connect to the exchange and scan live messages for rate/expiry-related fields.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Runtime for --live mode.")
    parser.add_argument("--host", default="34.197.188.76:3333")
    parser.add_argument("--username", default="uiuc")
    parser.add_argument("--password", default="mesa-lynx-octopus")
    parser.add_argument("--output", default=str(default_output_path()))
    args = parser.parse_args()

    schema_rows = find_schema_matches()
    code_rows = find_code_matches()
    print_rows("Schema Matches", schema_rows)
    print_rows("Current Bot Matches", code_rows)

    if not args.live:
        return
    asyncio.run(run_live_probe(args))


if __name__ == "__main__":
    main()
