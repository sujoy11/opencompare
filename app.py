#!/usr/bin/env python3
"""
OpenCompare — AI comparison engine (MVP, professional build).
Pure stdlib HTTP server · Gemini for analysis · graceful fallback.
"""
import re
import json
import os
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = HERE
DATA = os.path.join(HERE, "compare.json")
INDEX = os.path.join(HERE, "index.html")

COMPARES = []
if os.path.exists(DATA):
    try:
        with open(DATA) as f:
            COMPARES = json.load(f)
    except Exception:
        COMPARES = []

KEY_FILE = os.path.join(HERE, ".mistral_key")
MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "")
LAST_ERROR = ""
if not MISTRAL_KEY and os.path.exists(KEY_FILE):
    MISTRAL_KEY = open(KEY_FILE).read().strip()

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions" if MISTRAL_KEY else ""
MISTRAL_MODEL = "mistral-small-latest"

def slugify(text):
    """Turn 'ChatGPT' and 'Claude' into 'chatgpt-vs-claude'."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "item"

def make_slug(a, b):
    return f"{slugify(a)}-vs-{slugify(b)}"

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
    '"scores":{"a":7.5,"b":7.5},'
    '"a_pros":[3 detailed points, each a full sentence with specifics],"a_cons":[3 detailed points, each a full sentence],'
    '"b_pros":[3 detailed points, each a full sentence with specifics],"b_cons":[3 detailed points, each a full sentence],'
    '"metrics":['
    '{"label":"Price","a":"detailed value e.g. $999 starting, varies by config","b":"detailed value"},'
    '{"label":"Key Features","a":"detailed description","b":"detailed description"},'
    '{"label":"Performance","a":"detailed assessment","b":"detailed assessment"},'
    '{"label":"Ease of Use","a":"detailed assessment","b":"detailed assessment"},'
    '{"label":"Supported Platforms","a":"detailed answer","b":"detailed answer"},'
    '{"label":"Integrations","a":"detailed answer","b":"detailed answer"},'
    '{"label":"Security","a":"detailed answer","b":"detailed answer"}],'
    '"best_for":"one detailed phrase explaining ideal user",'
    '"summary":"3-4 sentence detailed neutral summary covering key differences",'
    '"recommendation":"3-4 sentence detailed buyer advice: clearly state which to pick for whom and why (mention specific strengths, budget, and use-case)",'
    '"alternatives":["alt1","alt2"]}'
    '\n\nEXAMPLE of exact format you must follow:\n'
    '{"winner":"Tie","scores":{"a":7.5,"b":7.5},"a_pros":["p1","p2","p3"],"a_cons":["c1","c2","c3"],"b_pros":["p1","p2","p3"],"b_cons":["c1","c2","c3"],"metrics":[{"label":"Price","a":"$999","b":"$899"},{"label":"Key Features","a":"x","b":"y"},{"label":"Performance","a":"x","b":"y"},{"label":"Ease of Use","a":"x","b":"y"},{"label":"Supported Platforms","a":"x","b":"y"},{"label":"Integrations","a":"x","b":"y"},{"label":"Security","a":"x","b":"y"}],"best_for":"phrase","summary":"2-3 sentences","recommendation":"3-4 sentences","alternatives":["alt1","alt2"]}'
)


def _call_mistral(prompt):
    """Call Mistral (OpenAI-compatible) and return model text."""
    body = json.dumps({
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are OpenCompare, a neutral comparison engine. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1400,
    }).encode()
    req = urllib.request.Request(
        MISTRAL_URL, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
        return d["choices"][0]["message"]["content"]


def _parse_json(raw):
    raw = raw.strip()
    # strip markdown fences
    if "```" in raw:
        # keep only the first fenced block
        parts = raw.split("```")
        # parts[0] is before first fence, parts[1] is inside (may have "json" prefix)
        if len(parts) >= 2:
            raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    # extract first {...} block
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        # 1. stray backslashes before quotes/apostrophes (Mistral escapes ' -> \')
        cleaned = raw.replace("\\'", "'").replace('\\"', '"')
        # 2. trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        # 3. missing colon after key: "key"[ or "key"{  ->  "key":[ / "key":{
        cleaned = re.sub(r'("(?:winner|scores|a_pros|a_cons|b_pros|b_cons|metrics|best_for|summary|recommendation|alternatives|label|a|b)")\s*\[', r'\1:[', cleaned)
        cleaned = re.sub(r'("(?:winner|scores|a_pros|a_cons|b_pros|b_cons|metrics|best_for|summary|recommendation|alternatives|label|a|b)")\s*\{', r'\1:{', cleaned)
        try:
            d = json.loads(cleaned)
        except json.JSONDecodeError:
            # 4. unbalanced brackets — try to close them
            depth = {"{": 0, "[": 0}
            for ch in cleaned:
                if ch == "{": depth["{"] += 1
                elif ch == "}": depth["{"] = max(0, depth["{"] - 1)
                elif ch == "[": depth["["] += 1
                elif ch == "]": depth["["] = max(0, depth["["] - 1)
            fixed = cleaned + "]" * depth["["] + "}" * depth["{"]
            d = json.loads(fixed)
        # 5. normalize scores: model may use item names instead of a/b
        if isinstance(d, dict) and "scores" in d and isinstance(d["scores"], dict):
            sc = d["scores"]
            if "a" not in sc or "b" not in sc:
                vals = list(sc.values())
                if len(vals) >= 2:
                    d["scores"] = {"a": vals[0], "b": vals[1]}
        return d


CACHE = {}  # key: "a|b" -> result dict (in-memory cache)

# Preload past comparisons into CACHE so popular pairs are instant on cold start
for _item in COMPARES:
    _a = (_item.get("item_a") or "").lower()
    _b = (_item.get("item_b") or "").lower()
    _r = _item.get("result")
    if _a and _b and _r:
        CACHE[f"{_a}|{_b}"] = _r


def compare(a, b):
    if not MISTRAL_KEY:
        return _fallback(a, b)
    key = f"{a.lower()}|{b.lower()}"
    # cache hit -> no API call (saves quota)
    if key in CACHE:
        return CACHE[key]
    try:
        result = _parse_json(_call_mistral(f"Compare '{a}' vs '{b}'.\n{PROMPT}"))
        if "scores" not in result or "winner" not in result:
            raise ValueError("incomplete")
        result = _fill(result, a, b)
        CACHE[key] = result
        return result
    except Exception as e:
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
        # 429 = rate limited -> signal so UI can say "try later"
        if "429" in str(e) or "too many" in str(e).lower() or "rate" in str(e).lower():
            return {"_rate_limited": True, "a": a, "b": b}
        return _fallback(a, b)


def _fill(result, a, b):
    """Ensure all UI-expected text fields are present (never blank)."""
    if not isinstance(result, dict):
        return _fallback(a, b)
    if not result.get("best_for"):
        result["best_for"] = f"Depends on priorities — {a} suits some needs, {b} others."
    if not result.get("summary"):
        result["summary"] = f"{a} and {b} each have distinct strengths; the better pick depends on what you value most."
    if not result.get("recommendation"):
        result["recommendation"] = f"Choose {a} if its strengths align with your needs; pick {b} otherwise. Try both where possible."
    if not result.get("alternatives"):
        result["alternatives"] = []
    return result


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


def _find_by_slug(slug):
    for item in COMPARES:
        if item.get("slug") == slug:
            return item
    return None


def og_image(slug):
    """Generate a share-card SVG for a comparison (dependency-free)."""
    item = _find_by_slug(slug)
    if not item:
        # fallback blank card
        a, b, winner, sa, sb = "Item A", "Item B", "VS", "0", "0"
    else:
        a = item.get("item_a", "Item A")
        b = item.get("item_b", "Item B")
        r = item.get("result", {}) or {}
        sc = r.get("scores", {}) or {}
        sa = str(sc.get("a", ""))
        sb = str(sc.get("b", ""))
        w = r.get("winner", "")
        if w == "A":
            winner = f"{a} wins"
        elif w == "B":
            winner = f"{b} wins"
        elif w == "Tie":
            winner = "Too close to call"
        else:
            winner = "VS"
    # escape XML
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    a, b, winner = esc(a), esc(b), esc(winner)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#7c5cfc"/>
      <stop offset="100%" stop-color="#ff6b6b"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <text x="600" y="120" font-family="Arial, sans-serif" font-size="42" font-weight="bold" fill="#ffffff" text-anchor="middle" opacity="0.9">OpenCompare</text>
  <text x="600" y="300" font-family="Arial, sans-serif" font-size="84" font-weight="bold" fill="#ffffff" text-anchor="middle">{a} <tspan fill="#ffd9d9">vs</tspan> {b}</text>
  <text x="600" y="400" font-family="Arial, sans-serif" font-size="56" font-weight="bold" fill="#ffffff" text-anchor="middle">{winner}</text>
  <text x="430" y="500" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="#ffffff" text-anchor="middle">{sa}</text>
  <text x="770" y="500" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="#ffffff" text-anchor="middle">{sb}</text>
  <text x="600" y="585" font-family="Arial, sans-serif" font-size="28" fill="#ffffff" text-anchor="middle" opacity="0.85">Compare anything. Decide smarter.</text>
</svg>'''
    return svg


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
        elif p.path == "/robots.txt":
            try:
                with open(os.path.join(BASE_DIR, "robots.txt")) as f:
                    self._send(200, f.read(), "text/plain")
            except Exception:
                self._send(404, "not found")
        elif p.path == "/sitemap.xml":
            try:
                with open(os.path.join(BASE_DIR, "sitemap.xml")) as f:
                    self._send(200, f.read(), "application/xml")
            except Exception:
                self._send(404, "not found")
        elif p.path == "/api/compare":
            self._send(200, json.dumps(COMPARES))
        elif p.path == "/api/categories":
            self._send(200, json.dumps(CATEGORIES))
        elif p.path == "/api/debug":
            net_ok = "unknown"
            try:
                # test HF endpoint reachability
                test_url = "https://api.mistral.ai/v1/models"
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
                "mistral_key_present": bool(MISTRAL_KEY),
                "mistral_key_len": len(MISTRAL_KEY),
                "mistral_model": MISTRAL_MODEL,
                "port": os.environ.get("PORT", "unset"),
                "network_to_mistral": net_ok,
                "last_error": LAST_ERROR,
            }))
        elif p.path == "/health":
            self._send(200, '{"status":"ok"}')
        elif p.path.startswith("/c/"):
            slug = p.path[3:].strip("/")
            # render the SPA; frontend reads ?c=<slug> (or /c/<slug>) and loads that comparison
            self._send(200, render(), "text/html")
        elif p.path.startswith("/og/") and p.path.endswith(".svg"):
            slug = p.path[len("/og/"):-len(".svg")].strip("/")
            self._send(200, og_image(slug), "image/svg+xml")
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
            "slug": make_slug(a, b),
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
