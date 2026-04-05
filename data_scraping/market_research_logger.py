from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import grpc

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
LOCAL_UTCXCHANGE_PATH = REPO_ROOT / "utc_exchange_codebase" / "utcxchangelib-main"
if str(LOCAL_UTCXCHANGE_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_UTCXCHANGE_PATH))

from config import ResearchLoggerConfig, load_config
from csv_writer import AppendSafeCsvWriter


LOGGER = logging.getLogger("market-research-logger")

try:
    from utcxchangelib import XChangeClient
    import utcxchangelib.service_pb2 as utc_bot_pb2
except ModuleNotFoundError as exc:
    XChangeClient = None  # type: ignore[assignment]
    utc_bot_pb2 = None  # type: ignore[assignment]
    UTCXCHANGE_IMPORT_ERROR = exc
else:
    UTCXCHANGE_IMPORT_ERROR = None

BaseXChangeClient = XChangeClient if XChangeClient is not None else object

BOOK_SNAPSHOT_FIELDNAMES = [
    "message_index",
    "symbol",
    "bids_json",
    "asks_json",
]

BOOK_UPDATE_FIELDNAMES = [
    "message_index",
    "symbol",
    "side",
    "px",
    "dq",
]

TRADE_FIELDNAMES = [
    "message_index",
    "symbol",
    "price",
    "qty",
]

NEWS_FIELDNAMES = [
    "message_index",
    "tick",
    "tick_ms",
    "kind",
    "symbol",
    "message_type",
    "structured_subtype",
    "earnings_asset",
    "earnings_value",
    "petition_asset",
    "petition_new_signatures",
    "petition_cumulative",
    "cpi_forecast",
    "cpi_actual",
    "raw_content",
    "normalized_content",
]

STREAM_FLUSH_EVERY = 200
SNAPSHOT_FLUSH_EVERY = 25
NEWS_FLUSH_EVERY = 10


def utc_now() -> tuple[str, int]:
    wall_time_ns = datetime.now(tz=timezone.utc).timestamp()
    wall_ns = int(wall_time_ns * 1_000_000_000)
    return datetime.fromtimestamp(wall_time_ns, tz=timezone.utc).isoformat(), wall_ns


def serialize_levels(levels: list[tuple[int, int]]) -> str:
    return json.dumps(
        [{"px": int(px), "qty": int(qty)} for px, qty in levels],
        separators=(",", ":"),
    )


class MarketResearchLogger(BaseXChangeClient):  # type: ignore[misc, valid-type]
    def __init__(self, config: ResearchLoggerConfig, run_dir: Path):
        super().__init__(config.host, config.username, config.password, symbols=config.monitored_symbols)
        self.config = config
        self.monitored_symbols = list(dict.fromkeys(config.monitored_symbols))
        self.run_dir = run_dir
        self.session_id = run_dir.name
        self.run_id = uuid.uuid4().hex
        self.shutdown_event = asyncio.Event()
        self.current_message_index: int | None = None
        self.latest_message_index: int | None = None
        self.message_counts: Counter[str] = Counter()

        self.book_snapshot_writers_by_symbol: dict[str, AppendSafeCsvWriter] = {
            symbol: AppendSafeCsvWriter(
                run_dir / f"raw_book_snapshots_{symbol}.csv",
                BOOK_SNAPSHOT_FIELDNAMES,
                flush_every=SNAPSHOT_FLUSH_EVERY,
            )
            for symbol in self.monitored_symbols
        }
        self.book_update_writers_by_symbol: dict[str, AppendSafeCsvWriter] = {
            symbol: AppendSafeCsvWriter(
                run_dir / f"raw_book_updates_{symbol}.csv",
                BOOK_UPDATE_FIELDNAMES,
                flush_every=STREAM_FLUSH_EVERY,
            )
            for symbol in self.monitored_symbols
        }
        self.trade_writers_by_symbol: dict[str, AppendSafeCsvWriter] = {
            symbol: AppendSafeCsvWriter(
                run_dir / f"raw_trade_events_{symbol}.csv",
                TRADE_FIELDNAMES,
                flush_every=STREAM_FLUSH_EVERY,
            )
            for symbol in self.monitored_symbols
        }
        self.news_writer = AppendSafeCsvWriter(
            run_dir / "raw_news_events.csv",
            NEWS_FIELDNAMES,
            flush_every=NEWS_FLUSH_EVERY,
        )

    async def start(self) -> None:
        self.write_session_metadata(status="starting")
        self.install_signal_handlers()
        try:
            await self.connect()
        except EOFError:
            LOGGER.info("Exchange stream ended for %s", self.run_dir)
        finally:
            self.write_session_metadata(status="stopped")
            self.close_writers()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signame in ("SIGINT", "SIGTERM"):
            if hasattr(signal, signame):
                try:
                    loop.add_signal_handler(getattr(signal, signame), self.shutdown_event.set)
                except NotImplementedError:
                    pass

    def close_writers(self) -> None:
        for writer in self.book_snapshot_writers_by_symbol.values():
            writer.close()
        for writer in self.book_update_writers_by_symbol.values():
            writer.close()
        for writer in self.trade_writers_by_symbol.values():
            writer.close()
        self.news_writer.close()

    def write_session_metadata(self, *, status: str) -> None:
        written_at_iso, written_at_ns = utc_now()
        payload = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": status,
            "written_at_iso": written_at_iso,
            "written_at_ns": written_at_ns,
            "latest_message_index": self.latest_message_index,
            "message_counts": dict(sorted(self.message_counts.items())),
            "monitored_symbols": self.monitored_symbols,
            "positions": dict(self.positions),
            "open_orders_count": len(self.open_orders),
            "notes": {
                "no_trading": "This listener never calls place_order, cancel_order, or place_swap_order.",
                "raw_capture_only": "Live capture stores compact raw exchange message fields only. Any book reconstruction, mid/spread computation, and plotting happen offline.",
                "tick_limitations": "Exchange tick is only present on news callbacks. Book/trade messages do not carry tick and are stored using exchange message_index instead.",
            },
            "config": self.config.to_metadata(),
        }
        with (self.run_dir / "session_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    async def process_message(self, msg) -> None:
        if msg == grpc.aio.EOF:
            raise EOFError("End of GRPC stream")

        self.current_message_index = int(msg.index)
        self.latest_message_index = self.current_message_index
        msg_type = msg.WhichOneof("body")
        if msg_type:
            self.message_counts[msg_type] += 1
        await super().process_message(msg)

    def _message_index(self) -> int | None:
        return self.current_message_index

    def _should_capture_symbol(self, symbol: str) -> bool:
        return symbol in self.book_update_writers_by_symbol

    def _write_book_snapshot(self, msg) -> None:
        if not self._should_capture_symbol(msg.symbol):
            return
        row = {
            "message_index": self._message_index(),
            "symbol": msg.symbol,
            "bids_json": serialize_levels([(bid.px, bid.qty) for bid in msg.bids]),
            "asks_json": serialize_levels([(ask.px, ask.qty) for ask in msg.asks]),
        }
        self.book_snapshot_writers_by_symbol[msg.symbol].write_row(row)

    def _write_book_update(self, msg) -> None:
        if not self._should_capture_symbol(msg.symbol):
            return
        side = "BUY" if msg.side == utc_bot_pb2.BookUpdate.Side.BUY else "SELL"
        row = {
            "message_index": self._message_index(),
            "symbol": msg.symbol,
            "side": side,
            "px": int(msg.px),
            "dq": int(msg.dq),
        }
        self.book_update_writers_by_symbol[msg.symbol].write_row(row)

    def _write_trade(self, msg) -> None:
        if not self._should_capture_symbol(msg.symbol):
            return
        row = {
            "message_index": self._message_index(),
            "symbol": msg.symbol,
            "price": int(msg.px),
            "qty": int(msg.qty),
        }
        self.trade_writers_by_symbol[msg.symbol].write_row(row)

    def _write_news(self, news_msg) -> None:
        news_type = "structured" if news_msg.HasField("structured") else "unstructured"
        symbol = news_msg.symbol if news_msg.HasField("symbol") else None
        tick = int(news_msg.tick)
        tick_ms = tick * 200

        message_type: str | None = None
        structured_subtype: str | None = None
        earnings_asset: str | None = None
        earnings_value: float | None = None
        petition_asset: str | None = None
        petition_new_signatures: int | None = None
        petition_cumulative: int | None = None
        cpi_forecast: float | None = None
        cpi_actual: float | None = None
        raw_content: str | None = None
        normalized_payload: dict[str, Any]

        if news_type == "structured":
            structured_subtype = news_msg.structured.WhichOneof("subtype")
            normalized_payload = {"structured_subtype": structured_subtype}
            if structured_subtype == "earnings":
                earnings_asset = news_msg.structured.earnings.asset
                earnings_value = float(news_msg.structured.earnings.value)
                normalized_payload["asset"] = earnings_asset
                normalized_payload["value"] = earnings_value
            elif structured_subtype == "petition":
                petition_asset = news_msg.structured.petition.asset
                petition_new_signatures = int(news_msg.structured.petition.new_signatures)
                petition_cumulative = int(news_msg.structured.petition.cumulative)
                normalized_payload["asset"] = petition_asset
                normalized_payload["new_signatures"] = petition_new_signatures
                normalized_payload["cumulative"] = petition_cumulative
            elif structured_subtype == "cpi_print":
                cpi_forecast = float(news_msg.structured.cpi_print.forecast)
                cpi_actual = float(news_msg.structured.cpi_print.actual)
                normalized_payload["forecast"] = cpi_forecast
                normalized_payload["actual"] = cpi_actual
        else:
            message_type = news_msg.unstructured.message_type
            raw_content = news_msg.unstructured.content
            normalized_payload = {
                "content": raw_content,
                "type": message_type,
            }

        row = {
            "message_index": self._message_index(),
            "tick": tick,
            "tick_ms": tick_ms,
            "kind": news_type,
            "symbol": symbol,
            "message_type": message_type,
            "structured_subtype": structured_subtype,
            "earnings_asset": earnings_asset,
            "earnings_value": earnings_value,
            "petition_asset": petition_asset,
            "petition_new_signatures": petition_new_signatures,
            "petition_cumulative": petition_cumulative,
            "cpi_forecast": cpi_forecast,
            "cpi_actual": cpi_actual,
            "raw_content": raw_content,
            "normalized_content": json.dumps(normalized_payload, sort_keys=True, separators=(",", ":")),
        }
        self.news_writer.write_row(row)

    async def handle_book_snapshot(self, msg) -> None:
        self._write_book_snapshot(msg)
        await super().handle_book_snapshot(msg)

    async def handle_book_update(self, msg) -> None:
        self._write_book_update(msg)
        await super().handle_book_update(msg)

    async def handle_trade_msg(self, msg):
        self._write_trade(msg)
        await super().handle_trade_msg(msg)

    async def handle_news_message(self, news_msg):
        self._write_news(news_msg)
        await super().handle_news_message(news_msg)

    async def bot_handle_book_update(self, symbol: str) -> None:
        return

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        return

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        LOGGER.warning("Received order fill in passive logger. order_id=%s qty=%s price=%s", order_id, qty, price)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        LOGGER.warning("Unexpected order rejection in passive logger. order_id=%s reason=%s", order_id, reason)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        LOGGER.warning(
            "Unexpected cancel response in passive logger. order_id=%s success=%s error=%s",
            order_id,
            success,
            error,
        )

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        LOGGER.warning("Unexpected swap response in passive logger. swap=%s qty=%s success=%s", swap, qty, success)

    async def bot_handle_news(self, news_release: dict):
        return

    async def bot_handle_market_resolved(self, market_id: str, winning_symbol: str, tick: int):
        LOGGER.info("Market resolved market_id=%s winning_symbol=%s tick=%s", market_id, winning_symbol, tick)

    async def bot_handle_settlement_payout(self, user: str, market_id: str, amount: int, tick: int):
        LOGGER.info("Settlement payout user=%s market_id=%s amount=%s tick=%s", user, market_id, amount, tick)


def make_run_dir(config: ResearchLoggerConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    label_suffix = f"_{config.run_label}" if config.run_label else ""
    run_dir = config.log_root / f"market_research_{timestamp}{label_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def async_main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config()
    if UTCXCHANGE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "utcxchangelib dependencies are not available. Install grpcio/protobuf and the client library "
            "before running the live logger."
        ) from UTCXCHANGE_IMPORT_ERROR

    run_dir = make_run_dir(config)
    LOGGER.info("Writing raw market research logs to %s", run_dir)
    client = MarketResearchLogger(config, run_dir)
    await client.start()


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "This logger reads settings from data_scraping/local_config.json. "
            "Edit that file, then run `python3 data_scraping/market_research_logger.py` with no extra arguments."
        )
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
