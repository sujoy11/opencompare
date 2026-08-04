#!/usr/bin/env python3
"""
OpenCompare — AI comparison engine (MVP, professional build).
Pure stdlib HTTP server · Gemini for analysis · graceful fallback.
"""
import json
import os
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
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

KEY_FILE = os.path.join(HERE, ".hf_key")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LAST_ERROR = ""
if not HF_TOKEN and os.path.exists(KEY_FILE):
    HF_TOKEN = open(KEY_FILE).read().strip()

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_URL = f"https://router.huggingface.co/v1/chat/completions" if HF_TOKEN else ""

CATEGORIES = [
    "AI Tools", "Software", "Hosting", "Smartphones", "Laptops",
    "VPN", "Antivirus", "Cloud Providers", "Web Browsers",
    "Databases", "Payment Gateways", "Website Builders",
]

PROMPT = (
    "You are OpenCompare, a neutral, fact-based comparison engine. "
    "Compare the two items thoroughly and fairly. "
    "Return ONLY valid JSON (no markdown, no code fences): "
    '{"winner":"A|B|Tie",'
    '"scores":{"a":<0-10>,"b":<0-10>},'
    '"a_pros":[3 short strings],"a_cons":[3 short strings],'
    '"b_pros":[3 short strings],"b_cons":[3 short strings],'
    '"metrics":['
    '{"label":"Price","a":"...","b":"..."},'
    '{"label":"Key Features","a":"...","b":"..."},'
    '{"label":"Performance","a":"...","b":"..."},'
    '{"label":"Ease of Use","a":"...","b":"..."},'
    '{"label":"Supported Platforms","a":"...","b":"..."},'
    '{"label":"Integrations","a":"...","b":"..."},'
    '{"label":"Security","a":"...","b":"..."}],'
    '"best_for":"one short phrase",'
    '"summary":"2-3 sentence neutral summary",'
    '"recommendation":"3-4 sentence detailed buyer advice: clearly state which to pick for whom and why (mention specific strengths, budget, and use-case)",'
    '"alternatives":["alt1","alt2"]}'
)


def _call_hf(prompt):
    """Call HuggingFace router (OpenAI-compatible) and return model text."""
    body = json.dumps({
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": "You are OpenCompare, a neutral comparison engine. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 350,
    }).encode()
    req = urllib.request.Request(
        HF_URL, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
        return d["choices"][0]["message"]["content"]


def _parse_json(raw):
    raw = raw.strip()
    # strip markdown fences
    if "```" in raw:
        # grab content between first ``` and closing ```
        start = raw.find("```")
        end = raw.find("```", start + 3)
        if end != -1:
            raw = raw[start + 3:end]
        if raw.startswith("json"):
            raw = raw[4:]
    # extract first {...} block
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    return json.loads(raw.strip())


CACHE = {}  # key: "a|b" -> result dict (in-memory cache)


def compare(a, b):
    if not HF_TOKEN:
        return _fallback(a, b)
    key = f"{a.lower()}|{b.lower()}"
    # cache hit -> no API call (saves quota)
    if key in CACHE:
        return CACHE[key]
    try:
        result = _parse_json(_call_hf(f"Compare '{a}' vs '{b}'.\n{PROMPT}"))
        if "scores" not in result or "winner" not in result:
            raise ValueError("incomplete")
        CACHE[key] = result
        return result
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
        # 429 = rate limited -> signal so UI can say "try later"
        if "429" in str(e) or "too many" in str(e).lower():
            return {"_rate_limited": True, "a": a, "b": b}
        return _fallback(a, b)


def _fallback(a, b):
    return {
        "winner": "Tie",
        "scores": {"a": 7.5, "b": 7.5},
        "a_pros": [f"{a} has notable strengths"],
        "a_cons": [f"{a} has some limitations"],
        "b_pros": [f"{b} has notable strengths"],
        "b_cons": [f"{b} has some limitations"],
        "metrics": [
            {"label": "Price", "a": "Varies", "b": "Varies"},
            {"label": "Key Features", "a": "Varies", "b": "Varies"},
            {"label": "Performance", "a": "Good", "b": "Good"},
            {"label": "Ease of Use", "a": "Good", "b": "Good"},
            {"label": "Supported Platforms", "a": "Multiple", "b": "Multiple"},
            {"label": "Integrations", "a": "Varies", "b": "Varies"},
            {"label": "Security", "a": "Standard", "b": "Standard"},
        ],
        "best_for": "Depends on use case",
        "summary": f"{a} and {b} each have trade-offs; pick based on your priority.",
        "recommendation": "Try both where possible before committing.",
        "alternatives": ["Other options exist in this category"],
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
        elif p.path == "/api/categories":
            self._send(200, json.dumps(CATEGORIES))
        elif p.path == "/api/debug":
            net_ok = "unknown"
            try:
                # test HF endpoint reachability
                test_url = "https://api-inference.huggingface.co/models/" + HF_MODEL
                urllib.request.urlopen(test_url, timeout=8)
                net_ok = "reachable"
            except urllib.error.HTTPError as he:
                if he.code in (400, 401, 403, 404):
                    net_ok = "reachable (api responded with %d)" % he.code
                else:
                    net_ok = f"FAIL HTTP {he.code}"
            except Exception as ne:
                net_ok = f"FAIL: {type(ne).__name__}: {str(ne)[:100]}"
            self._send(200, json.dumps({
                "hf_token_present": bool(HF_TOKEN),
                "hf_token_len": len(HF_TOKEN),
                "hf_model": HF_MODEL,
                "port": os.environ.get("PORT", "unset"),
                "network_to_hf": net_ok,
                "last_error": LAST_ERROR,
            }))
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
        result = compare(a, b)
        # rate-limited: don't store fake data, return clear signal
        if isinstance(result, dict) and result.get("_rate_limited"):
            self._send(429, json.dumps({
                "ok": False,
                "rate_limited": True,
                "error": "AI is rate-limited right now. Please try again in a minute — or check back shortly.",
            }))
            return
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
