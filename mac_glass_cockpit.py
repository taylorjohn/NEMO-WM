"""
mac_glass_cockpit.py — CORTEX-16 Glass Cockpit Bridge
Run this on the NUC. Open the browser on Mac at:
    http://192.168.1.195:9090/glass_cockpit.html
"""
import asyncio, socket, json, threading, time
import http.server, socketserver, os, webbrowser
from pathlib import Path

UDP_PORT  = 5005
WS_PORT   = 8765
HTTP_PORT = 9090
COCKPIT_HTML = Path(__file__).parent / "glass_cockpit.html"

_clients = set()
_clients_lock = threading.Lock()
_loop = None

def broadcast_sync(data: dict):
    global _loop
    if _loop is None: return
    asyncio.run_coroutine_threadsafe(_broadcast(json.dumps(data)), _loop)

async def _broadcast(msg: str):
    with _clients_lock:
        dead = set()
        for ws in list(_clients):
            try: await ws.send(msg)
            except Exception: dead.add(ws)
        for ws in dead: _clients.discard(ws)

async def ws_handler(websocket, path=None):
    with _clients_lock: _clients.add(websocket)
    print(f"  Browser connected ({len(_clients)} total)")
    try:
        async for msg in websocket:
            try:
                d = json.loads(msg)
                if d.get("cmd") == "KILL": print("  KILL command received")
            except Exception: pass
    except Exception: pass
    finally:
        with _clients_lock: _clients.discard(websocket)

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"  UDP listening on port {UDP_PORT}")
    buf = b""
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            buf += data
            while buf:
                start = buf.find(b"{")
                if start == -1: buf = b""; break
                buf = buf[start:]
                try:
                    packet, idx = json.JSONDecoder().raw_decode(buf.decode("utf-8", errors="ignore"))
                    buf = buf[idx:].lstrip()
                    broadcast_sync(packet)
                except json.JSONDecodeError as e:
                    if e.pos < len(buf) - 1:
                        nxt = buf.find(b"{", 1)
                        buf = b"" if nxt == -1 else buf[nxt:]
                    break
        except socket.timeout: buf = b""; continue
        except Exception as e: buf = b""; time.sleep(0.1)

class CockpitHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(COCKPIT_HTML.parent), **kw)
    def log_message(self, *a): pass

def start_http_server():
    try:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", HTTP_PORT), CockpitHandler) as s:
            print(f"  HTTP serving on port {HTTP_PORT}")
            s.serve_forever()
    except OSError as e:
        print(f"  HTTP port {HTTP_PORT} busy: {e}")

async def main():
    global _loop
    _loop = asyncio.get_event_loop()
    threading.Thread(target=udp_listener, daemon=True).start()
    threading.Thread(target=start_http_server, daemon=True).start()
    import websockets
    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        import socket as _s
        local_ip = _s.gethostbyname(_s.gethostname())
        print(f"\n  Cockpit live!")
        print(f"  Open on Mac: http://{local_ip}:{HTTP_PORT}/glass_cockpit.html")
        print(f"  Or try:      http://192.168.1.195:{HTTP_PORT}/glass_cockpit.html")
        print("  Press Ctrl+C to stop\n")
        await asyncio.Future()

if __name__ == "__main__":
    try: import websockets
    except ImportError: os.system("pip install websockets"); import websockets
    print(f"""
  CORTEX-16 | GLASS COCKPIT BRIDGE
  UDP:{UDP_PORT} (from run_trading.py)
  WS:{WS_PORT}   (to browser)
  HTTP:{HTTP_PORT} (serves glass_cockpit.html)
""")
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nCockpit stopped.")
