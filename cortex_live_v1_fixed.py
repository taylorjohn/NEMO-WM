import time
import struct
import socket
import numpy as np
import pynwb
import os
import cortex_math  # Phase 12: Compiled Rust Math Engine
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

# --- SURGICAL CONFIGURATION ---
load_dotenv()
ALPACA_KEY = os.getenv("ALPACA_PAPER_KEY")
ALPACA_SECRET = os.getenv("ALPACA_PAPER_SECRET")
ALPACA_URL = 'https://paper-api.alpaca.markets'

TARGET_IP = "192.168.1.150"  # Your Mac Studio HUD IP
PORT = 5005
SYMBOL = "SPY"
Z_THRESHOLD = 2.92  
MAX_SPREAD = 0.15 
TOXIC_THRESHOLD = 10.0 # Phase 13: Order Book Imbalance Ratio

class CortexGoldenMaster:
    def __init__(self):
        self.alpaca = REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rho_history = [] 
        self.history_limit = 1000
        
        # --- STATE MANAGEMENT ---
        self.engine_state = 1 
        self.is_toxic_state = False 
        self.cached_qty = 0.0
        self.active_order_id = None
        self.order_submit_time = 0.0
        
        # --- ACCOUNT & RISK ---
        self.mirror_wallet = 500.00
        self.mirror_dtbp = 500.00 
        self.daytrade_count = 0
        self.dynamic_z_threshold = Z_THRESHOLD
        
        # --- TELEMETRY & CLOCK ---
        self.last_rtt = 0.0
        self.last_account_sync = 0.0
        self.last_quote_sync = 0.0
        self.last_clock_sync = 0.0   # required by check_market_hours()
        self.market_open = False
        self.seconds_to_close = 999999.0

        # Load Allen Institute Neural Session
        nwb_path = r'C:\Users\MeteorAI\Desktop\cortex-12v15\allen_cache\session_715093703\session_715093703.nwb'
        self.io = pynwb.NWBHDF5IO(nwb_path, 'r')
        self.units = self.io.read().units.to_dataframe()
        self.start_time = time.time()
        
        # Phase 12 Warm-up
        print("⚡ Priming Rust Engine and Phase 13 Toxicity Filters...")
        for _ in range(5): cortex_math.calculate_z_score([1.0, 2.0], 1.5)

    def get_rho(self):
        elapsed = (time.time() - self.start_time) % 100
        total = 0
        for i in range(10):
            t = self.units.spike_times.iloc[i]
            total += len(t[(t > elapsed) & (t < elapsed + 0.033)])
        return float(total)

    def audit_liquidity_toxicity(self, bid_sz, ask_sz):
        """Phase 13: Detect informed institutional flow via OBI"""
        if bid_sz == 0: return True
        ratio = ask_sz / bid_sz
        return ratio >= TOXIC_THRESHOLD

    def manage_fills(self):
        """Phase 3: Partial Fill & TTL Management"""
        if not self.active_order_id: return 
        try:
            order = self.alpaca.get_order(self.active_order_id)
            if order.status in ['filled', 'canceled', 'expired', 'rejected']:
                if order.status == 'filled': self.last_account_sync = 0.0 
                self.active_order_id = None
                return
            if time.perf_counter() - self.order_submit_time > 3.0: 
                self.alpaca.cancel_order(self.active_order_id)
                self.active_order_id = None
        except: pass

    def live_execute(self, z_score):
        """The Main Surgical Loop"""
        # Phase 6: Adaptive Polling
        poll_interval = 0.1 if z_score > 2.5 else 0.5
        if time.perf_counter() - self.last_quote_sync < poll_interval: return
        
        try:
            t_start = time.perf_counter()
            quote = self.alpaca.get_latest_quote(SYMBOL)
            self.last_quote_sync = time.perf_counter() 
            self.last_rtt = (time.perf_counter() - t_start) * 1000.0 
            
            # Phase 2 & 4: Latency Watchdog
            if self.last_rtt > 45.0 or (time.time() - quote.t.timestamp()) > 1.5: return

            bid, ask = float(quote.bp), float(quote.ap)
            bid_sz, ask_sz = float(quote.bs), float(getattr(quote, 'as'))
            
            # Phase 9 & 13: Liquidity Defenses
            if (ask - bid) > MAX_SPREAD: return
            self.is_toxic_state = self.audit_liquidity_toxicity(bid_sz, ask_sz)
            
            # Phase 1: VWMP Calculation
            micro_price = (ask_sz * bid + bid_sz * ask) / (bid_sz + ask_sz) if (bid_sz + ask_sz) > 0 else (bid + ask) / 2.0
            
            # Entry/Exit Pricing Logic
            urgency = min(1.0, max(0.0, (z_score - Z_THRESHOLD) / 1.0))
            limit_price = round(micro_price + ((ask - micro_price) * urgency), 2)
            is_eod_panic = self.seconds_to_close <= 300.0

            # --- EXIT LOGIC ---
            if (z_score > Z_THRESHOLD or is_eod_panic or self.is_toxic_state) and self.cached_qty > 0:
                # If toxic or panic, sweep the bid. Otherwise, use VWMP.
                exit_p = round(bid - 0.01, 2) if (self.is_toxic_state or z_score > 4.0 or is_eod_panic) else limit_price
                order = self.alpaca.submit_order(symbol=SYMBOL, qty=self.cached_qty, side='sell', type='limit', limit_price=exit_p, time_in_force='day')
                self.active_order_id, self.order_submit_time = order.id, time.perf_counter()
                print(f"🚨 EXIT TRIGGERED: {z_score:.2f}σ | Price: ${exit_p}")

            # --- ENTRY LOGIC ---
            elif z_score <= Z_THRESHOLD and self.cached_qty == 0 and not is_eod_panic and not self.is_toxic_state:
                if self.dynamic_z_threshold != -999.0 and z_score <= (2.92 - (self.dynamic_z_threshold - 2.92)):
                    qty = round(min(self.mirror_wallet * 0.95, self.mirror_dtbp) / limit_price, 4)
                    if qty > 0.0001:
                        order = self.alpaca.submit_order(symbol=SYMBOL, qty=qty, side='buy', type='limit', limit_price=limit_price, time_in_force='day')
                        self.active_order_id, self.order_submit_time = order.id, time.perf_counter()
                        print(f"🎯 ENTRY DETECTED: {z_score:.2f}σ | Qty: {qty}")

        except Exception as e: print(f"⚠️ EXECUTION ERROR: {e}")

    def run(self):
        print(f"🚀 CORTEX-16 GOLDEN MASTER ONLINE")
        while True:
            try:
                # Core Neural Pulse
                rho = self.get_rho()
                self.rho_history.append(rho)
                if len(self.rho_history) > self.history_limit: self.rho_history.pop(0)
                
                # Phase 12: Rust-Powered Math
                z = cortex_math.calculate_z_score(self.rho_history, rho)
                
                # State Syncs
                self.check_market_hours()
                if self.market_open:
                    self.update_account_state()
                    self.manage_fills()
                    if not self.active_order_id: self.live_execute(z)
                
                # Telemetry Dispatch
                self.update_engine_state()
                pkt = struct.pack('<dffffi', time.time(), float(self.mirror_wallet), float(z), float(self.last_rtt), float(self.dynamic_z_threshold), int(self.engine_state))
                try: self.sock.sendto(pkt, (TARGET_IP, PORT))
                except: pass
                
                time.sleep(0.05) 
            except Exception as e: 
                print(f"🔌 CRITICAL RECOVERY: {e}")
                time.sleep(2)

    def update_engine_state(self):
        if not self.market_open: self.engine_state = 3 
        elif self.seconds_to_close <= 300.0: self.engine_state = 4 
        elif self.is_toxic_state: self.engine_state = 5 # New Toxic State
        elif self.dynamic_z_threshold == -999.0: self.engine_state = 0 
        elif self.active_order_id: self.engine_state = 2 
        else: self.engine_state = 1 

    def check_market_hours(self):
        if time.perf_counter() - self.last_clock_sync < 60.0: return
        try:
            c = self.alpaca.get_clock()
            self.market_open = c.is_open
            self.seconds_to_close = (c.next_close - c.timestamp).total_seconds() if self.market_open else 999999.0
            self.last_clock_sync = time.perf_counter()
        except: pass

    def update_account_state(self):
        if time.perf_counter() - self.last_account_sync < 5.0: return
        try:
            acc = self.alpaca.get_account()
            self.mirror_wallet = 500.00 * (float(acc.equity) / 100000.0)
            self.mirror_dtbp = self.mirror_wallet * (float(acc.daytrading_buying_power) / float(acc.equity))
            self.daytrade_count = int(acc.daytrade_count)
            self.dynamic_z_threshold = Z_THRESHOLD if (3 - self.daytrade_count) >= 2 else (Z_THRESHOLD * 1.25 if (3 - self.daytrade_count) == 1 else -999.0)
            try: self.cached_qty = float(self.alpaca.get_position(SYMBOL).qty)
            except: self.cached_qty = 0.0
            self.last_account_sync = time.perf_counter()
        except: pass

if __name__ == "__main__":
    from cortex_brain_inject import BrainInjector
    BrainInjector(CortexGoldenMaster()).run()
