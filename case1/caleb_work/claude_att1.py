"""

Here's the bot. A few critical things before you can run it:
You MUST update before competition day:

Option symbol names. The OPTION_STRIKES list at the top uses placeholder strings like "B95C". You need to replace these with whatever the actual exchange symbols are. You'll discover these during practice rounds or from the exchange's symbol universe. This is the single most important thing — if the symbols are wrong, the entire fair value computation fails.
Strike prices. The strikes (95, 100, 105) are placeholders. The case says "3 strikes around spot" — these shift dynamically. You may need to detect them from the symbol universe at runtime rather than hardcoding.
Credentials and host. HOST, USERNAME, PASSWORD at the top.
Risk limits. Update the MAX_* variables when final limits are published on Ed.



B Stock Trading Bot
====================
Trades stock B using the option chain as a fair-value oracle.

Strategy layers (in priority order):
  1. Conversions / Reversals  — near-riskless PCP arb (options + stock)
  2. Box Spread arb           — pure options arb across strike pairs
  3. Avellaneda-Stoikov MM    — inventory-aware market making on B stock
  4. Convergence trading      — aggressive trades when stock deviates from Ŝ

Key assumptions from case packet:
  - r = 0  (risk-free rate is zero)
  - European options, single expiry, 3 strikes around spot
  - Cash-settled options
  - FCFS matching engine — speed matters

IMPORTANT: Option symbol names below are PLACEHOLDERS.
Replace them with the actual exchange symbols before running.
"""

from typing import Optional, Dict, List, Tuple
import asyncio
import math
import time
from collections import deque

from utcxchangelib import XChangeClient, Side


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — UPDATE THESE BEFORE RUNNING
# ═══════════════════════════════════════════════════════════════════

# ---- Exchange connection ----
HOST = "localhost:3333"
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"

# ---- Option symbol mapping ----
# The case provides 3 strikes around spot. Fill in the actual
# exchange symbol strings once you know them.
# Format: (strike_price, call_symbol, put_symbol)
OPTION_STRIKES: List[Tuple[float, str, str]] = [
    # (strike, call_symbol, put_symbol)
    (95,  "B95C",  "B95P"),    # PLACEHOLDER — replace with real symbols
    (100, "B100C", "B100P"),   # PLACEHOLDER
    (105, "B105C", "B105P"),   # PLACEHOLDER
]

STOCK_SYMBOL = "B"

# ---- Risk limits (parameterized — update on competition day) ----
MAX_ORDER_SIZE = 40
MAX_OPEN_ORDER_SIZE = 100
MAX_OUTSTANDING_VOLUME = 200
MAX_ABSOLUTE_POSITION = 100

# ---- Avellaneda-Stoikov parameters ----
GAMMA = 0.1               # Risk aversion (tune: higher = tighter inventory, lower profit)
SIGMA_DEFAULT = 2.0        # Default volatility if not enough data
SIGMA_WINDOW = 60          # Rolling window size for realized vol estimation
MM_QUOTE_SIZE = 5          # Size of each market-making quote
MM_REFRESH_INTERVAL = 0.5  # Seconds between quote refreshes

# ---- Convergence trading ----
CONVERGENCE_THRESHOLD = 1.5  # Min deviation (Ŝ vs stock mid) to trigger aggressive trade
CONVERGENCE_SIZE = 10        # Size of convergence trades

# ---- Arb parameters ----
ARB_MIN_PROFIT = 0.10       # Minimum profit per unit to execute an arb
ARB_SIZE = 5                # Size of arb trades

# ---- Round timing ----
SECONDS_PER_DAY = 90
DAYS_PER_ROUND = 10
TICKS_PER_SECOND = 5
TOTAL_SECONDS_PER_ROUND = SECONDS_PER_DAY * DAYS_PER_ROUND


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def best_bid(order_book) -> Optional[float]:
    """Return the highest bid price, or None if book is empty."""
    if order_book.bids:
        return max(order_book.bids.keys())
    return None


def best_ask(order_book) -> Optional[float]:
    """Return the lowest ask price, or None if book is empty."""
    if order_book.asks:
        return min(order_book.asks.keys())
    return None


def mid_price(order_book) -> Optional[float]:
    """Return the mid-price, or None if either side is empty."""
    bb = best_bid(order_book)
    ba = best_ask(order_book)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return None


def spread(order_book) -> Optional[float]:
    """Return bid-ask spread, or None."""
    bb = best_bid(order_book)
    ba = best_ask(order_book)
    if bb is not None and ba is not None:
        return ba - bb
    return None


# ═══════════════════════════════════════════════════════════════════
# MAIN BOT
# ═══════════════════════════════════════════════════════════════════

class BStockBot(XChangeClient):
    """
    Trades stock B using its option chain as a continuously-updating
    fair value oracle, combined with Avellaneda-Stoikov inventory-aware
    market making.
    """

    def __init__(self, host, username, password):
        # Collect all symbols we need to track
        all_symbols = [STOCK_SYMBOL]
        for strike, call_sym, put_sym in OPTION_STRIKES:
            all_symbols.extend([call_sym, put_sym])

        super().__init__(host, username, password, silent=True, symbols=all_symbols)

        # ---- Strategy state ----
        self.fair_value: Optional[float] = None       # Option-implied fair value Ŝ
        self.sigma: float = SIGMA_DEFAULT              # Current volatility estimate
        self.price_history: deque = deque(maxlen=SIGMA_WINDOW)  # Recent B mid-prices
        self.round_start_time: float = time.time()

        # ---- Order tracking for market making ----
        self.mm_bid_order_id: Optional[str] = None
        self.mm_ask_order_id: Optional[str] = None

        # ---- Arb tracking ----
        self.pending_arb_legs: Dict[str, dict] = {}    # order_id -> arb context

        # ---- Performance tracking ----
        self.total_arb_profit: float = 0.0
        self.total_mm_fills: int = 0
        self.total_convergence_trades: int = 0

    # ───────────────────────────────────────────────────────────
    # FAIR VALUE COMPUTATION
    # ───────────────────────────────────────────────────────────

    def compute_option_implied_fair_value(self) -> Optional[float]:
        """
        Extract B's fair value from put-call parity across all 3 strikes.

        With r = 0:  C - P = S - K  →  Ŝ = C - P + K

        Returns the median of the 3 implied spots (robust to one
        strike being transiently mispriced).
        """
        implied_spots = []

        for strike, call_sym, put_sym in OPTION_STRIKES:
            call_book = self.order_books.get(call_sym)
            put_book = self.order_books.get(put_sym)

            if call_book is None or put_book is None:
                continue

            call_mid = mid_price(call_book)
            put_mid = mid_price(put_book)

            if call_mid is None or put_mid is None:
                continue

            # PCP with r=0:  Ŝ = C - P + K
            implied_spot = call_mid - put_mid + strike
            implied_spots.append(implied_spot)

        if not implied_spots:
            return None

        # Return median for robustness
        implied_spots.sort()
        n = len(implied_spots)
        if n % 2 == 1:
            return implied_spots[n // 2]
        else:
            return (implied_spots[n // 2 - 1] + implied_spots[n // 2]) / 2.0

    def update_sigma(self):
        """
        Update volatility estimate from recent tick-to-tick returns.
        Uses realized vol from B's stock price history.
        """
        if len(self.price_history) < 5:
            self.sigma = SIGMA_DEFAULT
            return

        prices = list(self.price_history)
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

        if len(returns) < 3:
            self.sigma = SIGMA_DEFAULT
            return

        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.001

        # Scale to per-second volatility (each tick is 0.2 seconds)
        # then to the price level
        current_price = prices[-1] if prices[-1] > 0 else 100
        self.sigma = max(std_r * current_price * math.sqrt(TICKS_PER_SECOND), 0.5)

    # ───────────────────────────────────────────────────────────
    # TIME MANAGEMENT
    # ───────────────────────────────────────────────────────────

    def time_remaining(self) -> float:
        """Estimate seconds remaining in current round."""
        elapsed = time.time() - self.round_start_time
        remaining = TOTAL_SECONDS_PER_ROUND - elapsed
        return max(remaining, 1.0)  # Floor at 1 to avoid division by zero

    def effective_gamma(self) -> float:
        """
        Increase risk aversion as round approaches end.
        Encourages inventory flattening near settlement.
        """
        remaining = self.time_remaining()
        # In the last 90 seconds (last day), ramp up gamma 3x
        if remaining < SECONDS_PER_DAY:
            return GAMMA * 3.0
        # In the last 2 days, ramp up 1.5x
        elif remaining < SECONDS_PER_DAY * 2:
            return GAMMA * 1.5
        return GAMMA

    # ───────────────────────────────────────────────────────────
    # POSITION & RISK CHECKS
    # ───────────────────────────────────────────────────────────

    def current_b_position(self) -> int:
        """Current inventory in B stock."""
        return self.positions.get(STOCK_SYMBOL, 0)

    def can_place_order(self, symbol: str, qty: int, side) -> bool:
        """
        Pre-flight check against risk limits.
        Returns True if the order is likely to be accepted.
        """
        if qty > MAX_ORDER_SIZE:
            return False

        # Check position limit
        current_pos = self.positions.get(symbol, 0)
        if side == Side.BUY or side == "buy":
            new_pos = current_pos + qty
        else:
            new_pos = current_pos - qty

        if abs(new_pos) > MAX_ABSOLUTE_POSITION:
            return False

        # Check outstanding volume
        outstanding = sum(
            order[1]  # remaining qty
            for order in self.open_orders.values()
        )
        if outstanding + qty > MAX_OUTSTANDING_VOLUME:
            return False

        return True

    # ───────────────────────────────────────────────────────────
    # STRATEGY 1: CONVERSIONS & REVERSALS (PCP ARBITRAGE)
    # ───────────────────────────────────────────────────────────

    async def check_pcp_arb(self):
        """
        Check put-call parity at each strike and execute
        conversions/reversals when profitable.

        With r = 0:
          Conversion:  C - P > S - K  → buy stock, buy put, sell call
          Reversal:    C - P < S - K  → sell stock, sell put, buy call
        """
        stock_book = self.order_books.get(STOCK_SYMBOL)
        if stock_book is None:
            return

        stock_bb = best_bid(stock_book)
        stock_ba = best_ask(stock_book)
        if stock_bb is None or stock_ba is None:
            return

        for strike, call_sym, put_sym in OPTION_STRIKES:
            call_book = self.order_books.get(call_sym)
            put_book = self.order_books.get(put_sym)
            if call_book is None or put_book is None:
                continue

            call_bb = best_bid(call_book)
            call_ba = best_ask(call_book)
            put_bb = best_bid(put_book)
            put_ba = best_ask(put_book)

            if any(x is None for x in [call_bb, call_ba, put_bb, put_ba]):
                continue

            # ---- Check for CONVERSION opportunity ----
            # Synthetic is rich: (sell call bid) - (buy put ask) > (buy stock ask) - K
            # Profit = (call_bid - put_ask) - (stock_ask - strike)
            conversion_profit = (call_bb - put_ba) - (stock_ba - strike)
            if conversion_profit > ARB_MIN_PROFIT:
                size = min(ARB_SIZE, MAX_ORDER_SIZE)
                if (self.can_place_order(STOCK_SYMBOL, size, Side.BUY) and
                    self.can_place_order(call_sym, size, Side.SELL) and
                    self.can_place_order(put_sym, size, Side.BUY)):

                    print(f"[ARB] CONVERSION at K={strike}, profit={conversion_profit:.2f}/unit")

                    # Execute all three legs as aggressively as possible
                    await self.place_order(STOCK_SYMBOL, size, Side.BUY, int(stock_ba))
                    await self.place_order(call_sym, size, Side.SELL, int(call_bb))
                    await self.place_order(put_sym, size, Side.BUY, int(put_ba))

                    self.total_arb_profit += conversion_profit * size

            # ---- Check for REVERSAL opportunity ----
            # Synthetic is cheap: (stock_bid - strike) > (call_ask - put_bid)
            # Profit = (stock_bid - strike) - (call_ask - put_bid)
            reversal_profit = (stock_bb - strike) - (call_ba - put_bb)
            if reversal_profit > ARB_MIN_PROFIT:
                size = min(ARB_SIZE, MAX_ORDER_SIZE)
                if (self.can_place_order(STOCK_SYMBOL, size, Side.SELL) and
                    self.can_place_order(call_sym, size, Side.BUY) and
                    self.can_place_order(put_sym, size, Side.SELL)):

                    print(f"[ARB] REVERSAL at K={strike}, profit={reversal_profit:.2f}/unit")

                    await self.place_order(STOCK_SYMBOL, size, Side.SELL, int(stock_bb))
                    await self.place_order(call_sym, size, Side.BUY, int(call_ba))
                    await self.place_order(put_sym, size, Side.SELL, int(put_bb))

                    self.total_arb_profit += reversal_profit * size

    # ───────────────────────────────────────────────────────────
    # STRATEGY 2: BOX SPREAD ARBITRAGE
    # ───────────────────────────────────────────────────────────

    async def check_box_arb(self):
        """
        Check box spreads across all strike pairs.

        A box at (K1, K2) with r=0 should cost exactly K2 - K1.

        Box = bull call spread + bear put spread
            = (buy C_K1, sell C_K2) + (buy P_K2, sell P_K1)

        If box price < K2 - K1 → buy the box (it's cheap)
        If box price > K2 - K1 → sell the box (it's rich)
        """
        for i in range(len(OPTION_STRIKES)):
            for j in range(i + 1, len(OPTION_STRIKES)):
                k1, c1_sym, p1_sym = OPTION_STRIKES[i]
                k2, c2_sym, p2_sym = OPTION_STRIKES[j]

                c1_book = self.order_books.get(c1_sym)
                c2_book = self.order_books.get(c2_sym)
                p1_book = self.order_books.get(p1_sym)
                p2_book = self.order_books.get(p2_sym)

                if any(b is None for b in [c1_book, c2_book, p1_book, p2_book]):
                    continue

                c1_bb, c1_ba = best_bid(c1_book), best_ask(c1_book)
                c2_bb, c2_ba = best_bid(c2_book), best_ask(c2_book)
                p1_bb, p1_ba = best_bid(p1_book), best_ask(p1_book)
                p2_bb, p2_ba = best_bid(p2_book), best_ask(p2_book)

                if any(x is None for x in [c1_bb, c1_ba, c2_bb, c2_ba,
                                           p1_bb, p1_ba, p2_bb, p2_ba]):
                    continue

                theoretical_value = k2 - k1  # r=0 so no discounting

                # Cost to BUY box: buy C_K1 at ask, sell C_K2 at bid,
                #                  buy P_K2 at ask, sell P_K1 at bid
                buy_box_cost = c1_ba - c2_bb + p2_ba - p1_bb
                buy_box_profit = theoretical_value - buy_box_cost

                if buy_box_profit > ARB_MIN_PROFIT:
                    size = min(ARB_SIZE, MAX_ORDER_SIZE)
                    print(f"[BOX] BUY box ({k1},{k2}), profit={buy_box_profit:.2f}/unit")

                    await self.place_order(c1_sym, size, Side.BUY, int(c1_ba))
                    await self.place_order(c2_sym, size, Side.SELL, int(c2_bb))
                    await self.place_order(p2_sym, size, Side.BUY, int(p2_ba))
                    await self.place_order(p1_sym, size, Side.SELL, int(p1_bb))

                    self.total_arb_profit += buy_box_profit * size

                # Cost to SELL box: sell C_K1 at bid, buy C_K2 at ask,
                #                   sell P_K2 at bid, buy P_K1 at ask
                sell_box_revenue = c1_bb - c2_ba + p1_ba  # wait, let me redo
                # Revenue from selling box = sell C_K1 bid + buy C_K2 ask
                #   + sell P_K2 bid + buy P_K1 ask
                # Actually: sell box means take opposite of buy box
                sell_box_revenue = c1_bb - c2_ba + p1_bb - p2_ba
                # We receive sell_box_revenue now, pay back theoretical_value at expiry
                # Wait — with cash settlement, selling the box means:
                # sell C_K1 (receive c1_bb), buy C_K2 (pay c2_ba),
                # sell P_K2 (receive p2_bb), buy P_K1 (pay p1_ba)
                sell_box_net = c1_bb - c2_ba + p2_bb - p1_ba
                sell_box_profit = sell_box_net - theoretical_value

                if sell_box_profit > ARB_MIN_PROFIT:
                    size = min(ARB_SIZE, MAX_ORDER_SIZE)
                    print(f"[BOX] SELL box ({k1},{k2}), profit={sell_box_profit:.2f}/unit")

                    await self.place_order(c1_sym, size, Side.SELL, int(c1_bb))
                    await self.place_order(c2_sym, size, Side.BUY, int(c2_ba))
                    await self.place_order(p2_sym, size, Side.SELL, int(p2_bb))
                    await self.place_order(p1_sym, size, Side.BUY, int(p1_ba))

                    self.total_arb_profit += sell_box_profit * size

    # ───────────────────────────────────────────────────────────
    # STRATEGY 3: AVELLANEDA-STOIKOV MARKET MAKING ON B STOCK
    # ───────────────────────────────────────────────────────────

    def compute_as_quotes(self) -> Optional[Tuple[float, float]]:
        """
        Compute optimal bid and ask prices using Avellaneda-Stoikov.

        Reservation price:  r = Ŝ - q * γ * σ² * (T - t)
        Optimal spread:     δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)

        Returns (bid_price, ask_price) or None if fair value unavailable.
        """
        if self.fair_value is None:
            return None

        q = self.current_b_position()
        gamma = self.effective_gamma()
        sigma = self.sigma
        T_minus_t = self.time_remaining()

        # Reservation price: option-implied fair value adjusted for inventory
        reservation = self.fair_value - q * gamma * (sigma ** 2) * T_minus_t

        # Spread calculation
        # k controls fill probability decay — calibrate empirically
        # Using k = 1.5 as in the paper's simulation
        k = 1.5
        optimal_spread = gamma * (sigma ** 2) * T_minus_t + (2.0 / gamma) * math.log(1 + gamma / k)

        # Floor the spread to avoid quoting inside the minimum tick
        optimal_spread = max(optimal_spread, 0.5)

        # Cap the spread to avoid absurdly wide quotes that never fill
        optimal_spread = min(optimal_spread, 10.0)

        half_spread = optimal_spread / 2.0

        bid_px = reservation - half_spread
        ask_px = reservation + half_spread

        return (bid_px, ask_px)

    async def cancel_mm_orders(self):
        """Cancel existing market-making orders before refreshing."""
        if self.mm_bid_order_id and self.mm_bid_order_id in self.open_orders:
            await self.cancel_order(self.mm_bid_order_id)
        if self.mm_ask_order_id and self.mm_ask_order_id in self.open_orders:
            await self.cancel_order(self.mm_ask_order_id)

    async def refresh_mm_quotes(self):
        """
        Cancel old quotes and place new bid/ask for B stock
        using the Avellaneda-Stoikov framework.
        """
        quotes = self.compute_as_quotes()
        if quotes is None:
            return

        bid_px, ask_px = quotes

        # Round to integers (exchange likely uses integer prices)
        bid_px = int(math.floor(bid_px))
        ask_px = int(math.ceil(ask_px))

        # Ensure bid < ask
        if bid_px >= ask_px:
            ask_px = bid_px + 1

        # Cancel old orders
        await self.cancel_mm_orders()

        # Small delay to let cancels process
        await asyncio.sleep(0.05)

        # Place new orders if risk limits allow
        if self.can_place_order(STOCK_SYMBOL, MM_QUOTE_SIZE, Side.BUY):
            self.mm_bid_order_id = await self.place_order(
                STOCK_SYMBOL, MM_QUOTE_SIZE, Side.BUY, bid_px
            )

        if self.can_place_order(STOCK_SYMBOL, MM_QUOTE_SIZE, Side.SELL):
            self.mm_ask_order_id = await self.place_order(
                STOCK_SYMBOL, MM_QUOTE_SIZE, Side.SELL, ask_px
            )

    # ───────────────────────────────────────────────────────────
    # STRATEGY 4: CONVERGENCE TRADING
    # ───────────────────────────────────────────────────────────

    async def check_convergence(self):
        """
        If B's stock mid-price deviates significantly from the
        option-implied fair value, trade aggressively toward convergence.
        """
        if self.fair_value is None:
            return

        stock_book = self.order_books.get(STOCK_SYMBOL)
        if stock_book is None:
            return

        stock_mid = mid_price(stock_book)
        if stock_mid is None:
            return

        deviation = self.fair_value - stock_mid

        if abs(deviation) < CONVERGENCE_THRESHOLD:
            return

        # Stock is cheap relative to options → buy stock
        if deviation > CONVERGENCE_THRESHOLD:
            stock_ba = best_ask(stock_book)
            if stock_ba is not None:
                size = min(CONVERGENCE_SIZE, MAX_ORDER_SIZE)
                if self.can_place_order(STOCK_SYMBOL, size, Side.BUY):
                    print(f"[CONV] BUY B: stock_mid={stock_mid:.1f}, "
                          f"fair={self.fair_value:.1f}, dev={deviation:.2f}")
                    await self.place_order(STOCK_SYMBOL, size, Side.BUY, int(stock_ba))
                    self.total_convergence_trades += 1

        # Stock is expensive relative to options → sell stock
        elif deviation < -CONVERGENCE_THRESHOLD:
            stock_bb = best_bid(stock_book)
            if stock_bb is not None:
                size = min(CONVERGENCE_SIZE, MAX_ORDER_SIZE)
                if self.can_place_order(STOCK_SYMBOL, size, Side.SELL):
                    print(f"[CONV] SELL B: stock_mid={stock_mid:.1f}, "
                          f"fair={self.fair_value:.1f}, dev={deviation:.2f}")
                    await self.place_order(STOCK_SYMBOL, size, Side.SELL, int(stock_bb))
                    self.total_convergence_trades += 1

    # ───────────────────────────────────────────────────────────
    # CALLBACK HANDLERS
    # ───────────────────────────────────────────────────────────

    async def bot_handle_book_update(self, symbol: str):
        """
        Called whenever any order book updates. This is where we
        recompute fair value and check for arb on every tick.
        """
        # Track B stock mid-price for sigma estimation
        if symbol == STOCK_SYMBOL:
            stock_mid = mid_price(self.order_books.get(STOCK_SYMBOL))
            if stock_mid is not None:
                self.price_history.append(stock_mid)

        # Recompute fair value from options on any relevant book change
        is_option_update = any(
            symbol in (c, p) for _, c, p in OPTION_STRIKES
        )

        if symbol == STOCK_SYMBOL or is_option_update:
            new_fv = self.compute_option_implied_fair_value()
            if new_fv is not None:
                self.fair_value = new_fv

            # Update sigma periodically
            self.update_sigma()

            # Check arb opportunities (highest priority — run on every update)
            await self.check_pcp_arb()
            await self.check_box_arb()

            # Check convergence trades
            await self.check_convergence()

    async def bot_handle_trade_msg(self, symbol: str, price: float, qty: int):
        """
        React to trade prints — look for smart money signals.
        Large trades far from fair value may indicate informed flow.
        """
        if symbol != STOCK_SYMBOL or self.fair_value is None:
            return

        deviation = price - self.fair_value

        # If a large trade prints significantly away from our fair value,
        # shade our fair value slightly in that direction
        if qty >= 20 and abs(deviation) > 1.0:
            # Nudge fair value 10% of the way toward the trade price
            adjustment = deviation * 0.10
            self.fair_value += adjustment
            print(f"[SMART] Large trade at {price} (qty={qty}), "
                  f"nudged fair value by {adjustment:.2f} to {self.fair_value:.2f}")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: float):
        """Track fills for logging and inventory awareness."""
        # Check if this was a market-making fill
        if order_id == self.mm_bid_order_id or order_id == self.mm_ask_order_id:
            self.total_mm_fills += 1
            side = "BUY" if order_id == self.mm_bid_order_id else "SELL"
            print(f"[MM] Fill: {side} {qty}@{price}, "
                  f"position now {self.current_b_position()}")

    async def bot_handle_order_rejected(self, order_id: str, reason: str):
        """Handle rejections — likely a risk limit breach."""
        print(f"[REJECTED] Order {order_id}: {reason}")

        # Clear our tracked MM order IDs if they were rejected
        if order_id == self.mm_bid_order_id:
            self.mm_bid_order_id = None
        elif order_id == self.mm_ask_order_id:
            self.mm_ask_order_id = None

    async def bot_handle_cancel_response(self, order_id: str,
                                          success: bool,
                                          error: Optional[str]):
        """Track cancel results."""
        if not success and error:
            # Order may have already been filled — that's fine
            pass

        # Clear tracked IDs on successful cancel
        if success:
            if order_id == self.mm_bid_order_id:
                self.mm_bid_order_id = None
            elif order_id == self.mm_ask_order_id:
                self.mm_ask_order_id = None

    # ───────────────────────────────────────────────────────────
    # MAIN TRADING LOOP
    # ───────────────────────────────────────────────────────────

    async def trading_loop(self):
        """
        Main loop: periodically refresh market-making quotes.
        Arb and convergence are handled reactively in book updates.
        """
        # Wait for initial connection and book data
        await asyncio.sleep(2)

        self.round_start_time = time.time()
        print("[INIT] B Stock Bot started")
        print(f"[INIT] Strikes: {[(s, c, p) for s, c, p in OPTION_STRIKES]}")
        print(f"[INIT] Gamma={GAMMA}, Sigma_default={SIGMA_DEFAULT}")

        iteration = 0
        while True:
            try:
                # Refresh market-making quotes
                await self.refresh_mm_quotes()

                # Periodic status log
                iteration += 1
                if iteration % 20 == 0:
                    pos = self.current_b_position()
                    fv = f"{self.fair_value:.2f}" if self.fair_value else "N/A"
                    stock_mid_val = mid_price(self.order_books.get(STOCK_SYMBOL))
                    sm = f"{stock_mid_val:.2f}" if stock_mid_val else "N/A"
                    print(f"[STATUS] pos={pos}, fair_value={fv}, "
                          f"stock_mid={sm}, sigma={self.sigma:.3f}, "
                          f"arb_pnl={self.total_arb_profit:.1f}, "
                          f"mm_fills={self.total_mm_fills}, "
                          f"conv_trades={self.total_convergence_trades}")

                await asyncio.sleep(MM_REFRESH_INTERVAL)

            except Exception as e:
                print(f"[ERROR] Trading loop: {e}")
                await asyncio.sleep(1)

    # ───────────────────────────────────────────────────────────
    # ENTRY POINT
    # ───────────────────────────────────────────────────────────

    async def start(self):
        """Launch the trading loop and connect to the exchange."""
        asyncio.create_task(self.trading_loop())
        await self.connect()


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = BStockBot(HOST, USERNAME, PASSWORD)
    asyncio.run(bot.start())