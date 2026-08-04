#!/usr/bin/env python3
"""
OpenCompare MVP — AI-powered comparison tool.
- Pure stdlib HTTP server (no pip needed, Railway free tier)
- Gemini (gemini-flash-latest) for real comparison analysis
- Graceful fallback to rule-based if Gemini unavailable
"""
import json
import os
import re
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "compare.json")
INDEX = os.path.join(HERE, "index.html")

COMPARES = []
if os.path.exists(DATA):
    try:
        with open(DATA) as f:
            COMPARES = json.load(f)
    except Exception:
        COMPARES = []

GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
if not GEMINI_KEY and os.path.exists(os.path.join(HERE, ".gemini_key")):
    GEMINI_KEY = open(os.path.join(HERE, ".gemini_key")).read().strip()

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-latest:generateContent?key=" + GEMINI_KEY
) if GEMINI_KEY else ""


def compare_ai(item_a, item_b):
    """Use Gemini to generate a structured comparison."""
    if not GEMINI_KEY:
        return fallback_compare(item_a, item_b)
    prompt = (
        f"Compare these two: '{item_a}' vs '{item_b}'. "
        "Respond with JSON only (no markdown): "
        '{"winner_overall":"A|B|tie",'
        '"score_a":0-10,"score_b":0-10,'
        '"a_pros":["..."],"a_cons":["..."],'
        '"b_pros":["..."],"b_cons":["..."],'
        '"best_for":"one sentence",'
        '"summary":"2-3 sentence neutral summary",'
        '"metrics":[{"name":"Price","a":"...","b":"..."},'
        '{"name":"Ease of Use","a":"...","b":"..."},'
        '{"name":"Features","a":"...","b":"..."}]}'
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        GEMINI_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
            raw = d["candidates"][0]["content"]["parts"][0]["text"]
        # strip markdown fences if any
        raw = raw.strip().strip("`").replace("json", "", 1).strip()
        return json.loads(raw)
    except Exception:
        return fallback_compare(item_a, item_b)


def fallback_compare(a, b):
    return {
        "winner_overall": "tie",
        "score_a": 7.5, "score_b": 7.5,
        "a_pros": [f"{a} has strengths"], "a_cons": [f"{a} has limitations"],
        "b_pros": [f"{b} has strengths"], "b_cons": [f"{b} has limitations"],
        "best_for": "Depends on use case",
        "summary": f"{a} and {b} both have trade-offs. Choose based on your priority.",
        "metrics": [
            {"name": "Price", "a": "Varies", "b": "Varies"},
            {"name": "Ease of Use", "a": "Good", "b": "Good"},
            {"name": "Features", "a": "Varies", "b": "Varies"},
        ],
    }


def render():
    with open(INDEX) as f:
        return f.read()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            self._send(200, render(), "text/html")
        elif p.path == "/api/compare":
            self._send(200, json.dumps(COMPARES))
        elif p.path == "/health":
            self._send(200, '{"status":"ok"}')
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/api/compare":
            self._send(404, '{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode())
        except Exception:
            body = {}
        a = (body.get("item_a") or "").strip()
        b = (body.get("item_b") or "").strip()
        if not a or not b:
            self._send(400, json.dumps({"ok": False, "error": "both items required"}))
            return
        result = compare_ai(a, b)
        item = {
            "id": len(COMPARES) + 1,
            "item_a": a,
            "item_b": b,
            "result": result,
            "time": datetime.datetime.now().strftime("%H:%M"),
        }
        COMPARES.append(item)
        with open(DATA, "w") as f:
            json.dump(COMPARES, f)
        self._send(200, json.dumps({"ok": True, "item": item}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"OpenCompare running on port {port}")
    srv.serve_forever()
