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
    '"best_for_roles":[{"role":"e.g. Students","winner":"A|B","reason":"one short sentence why this item wins for that role"},{"role":"e.g. Coding","winner":"A|B","reason":"short reason"},{"role":"e.g. Budget","winner":"A|B","reason":"short reason"}],'
    '"summary":"3-4 sentence detailed neutral summary covering key differences",'
    '"recommendation":"3-4 sentence detailed buyer advice: clearly state which to pick for whom and why (mention specific strengths, budget, and use-case)",'
    '"alternatives":["alt1","alt2"]}'
    '\n\nEXAMPLE of exact format you must follow:\n'
    '{"winner":"Tie","scores":{"a":7.5,"b":7.5},"a_pros":["p1","p2","p3"],"a_cons":["c1","c2","c3"],"b_pros":["p1","p2","p3"],"b_cons":["c1","c2","c3"],"metrics":[{"label":"Price","a":"$999","b":"$899"},{"label":"Key Features","a":"x","b":"y"},{"label":"Performance","a":"x","b":"y"},{"label":"Ease of Use","a":"x","b":"y"},{"label":"Supported Platforms","a":"x","b":"y"},{"label":"Integrations","a":"x","b":"y"},{"label":"Security","a":"x","b":"y"}],"best_for":"phrase","best_for_roles":[{"role":"Students","winner":"A","reason":"better for learning"},{"role":"Coding","winner":"B","reason":"stronger dev tools"},{"role":"Budget","winner":"A","reason":"cheaper plan"}],"summary":"2-3 sentences","recommendation":"3-4 sentences","alternatives":["alt1","alt2"]}'
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
    if not result.get("best_for_roles"):
        result["best_for_roles"] = []
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


def og_image(slug, query=""):
    """Generate a share-card SVG for a comparison (dependency-free).
    Reads data from query string when present (stateless), else from stored COMPARES."""
    from urllib.parse import parse_qs
    q = parse_qs(query)
    def g(k, d=""):
        return q.get(k, [d])[0]
    a = g("a")
    b = g("b")
    winner = g("w")
    sa = g("sa")
    sb = g("sb")
    # if no query data, try stored comparison
    if not a and not b:
        item = _find_by_slug(slug)
        if item:
            a = item.get("item_a", "Item A")
            b = item.get("item_b", "Item B")
            r = item.get("result", {}) or {}
            sc = r.get("scores", {}) or {}
            sa = str(sc.get("a", ""))
            sb = str(sc.get("b", ""))
            w = r.get("winner", "")
            if w == "A": winner = f"{a} wins"
            elif w == "B": winner = f"{b} wins"
            elif w == "Tie": winner = "Too close to call"
            else: winner = "VS"
    if not a: a = "Item A"
    if not b: b = "Item B"
    if not winner: winner = "VS"
    a = str(a)[:40]; b = str(b)[:40]; winner = str(winner)[:40]
    sa = str(sa)[:6]; sb = str(sb)[:6]

    # ---- pure-Python PNG generator (no Pillow dependency) ----
    import zlib, struct
    W, H = 1200, 630
    # gradient background (violet #7c5cfc -> coral #ff6b6b) per column
    c1 = (124, 92, 252); c2 = (255, 107, 107)
    # 5x7 bitmap font for A-Z 0-9 space - . :  (each char = 7 rows x 5 cols, top-first, verified)
    FONT = {
        ' ':"00000000000000000000000000000000000",
        '-':"00000000000000011100000000000000000",
        '.':"00000000000000000000000000000000110",
        '0':"01110100111001110101110011100101110",
        '1':"00100011000010000100001000010001110",
        '2':"01110100010000100110010001000011111",
        '3':"11110000010000101110000010000111110",
        '4':"00010001100101010010111110001000010",
        '5':"11111100001111000001000011000101110",
        '6':"01110100001000011110100011000101110",
        '7':"11111000010001000100010000100001000",
        '8':"01110100011000101110100011000101110",
        '9':"01110100011000101111000010000101110",
        ':':"00000000000011000000000000011000000",
        'A':"01110100011000111111100011000110001",
        'B':"11110100011000111110100011000111110",
        'C':"01110100011000010000100001000101110",
        'D':"11110100011000110001100011000111110",
        'E':"11111100001000011110100001000011111",
        'F':"11111100001000011110100001000010000",
        'G':"01110100011000010111100011000101110",
        'H':"10001100011000111111100011000110001",
        'I':"01110001000010000100001000010001110",
        'J':"00111000100001000010000101001001100",
        'K':"10001100101010011000101001001010001",
        'L':"10000100001000010000100001000011111",
        'M':"10001110111010110101100011000110001",
        'N':"10001110011010110011100011000110001",
        'O':"01110100011000110001100011000101110",
        'P':"11110100011000111110100001000010000",
        'Q':"01110100011000110001101011001001101",
        'R':"11110100011000111110101001001010001",
        'S':"01111100001000001110000010000111110",
        'T':"11111001000010000100001000010000100",
        'U':"10001100011000110001100011000101110",
        'V':"10001100011000110001100010101000100",
        'W':"10001100011000110101101011101110001",
        'X':"10001100010101000100010101000110001",
        'Y':"10001100010101000100001000010000100",
        'Z':"11111000010001000100010001000011111",
    }
    def glyph_rows(ch):
        ch = ch.upper()
        bits = FONT.get(ch, FONT[' '])
        return [bits[i*5:(i+1)*5] for i in range(7)]

    def px(x, y):
        t = x / W
        return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t), int(c1[2]+(c2[2]-c1[2])*t))

    # build pixel buffer (RGB tuples)
    buf = [[px(x, y) for x in range(W)] for y in range(H)]
    WHITE = (255, 255, 255); PINK = (255, 217, 217)

    def draw_text(text, cx, cy, color, scale=8):
        rows = []
        for ch in text:
            rows.append(glyph_rows(ch))
        # total width
        char_w = 5 * scale + scale  # 5 cols + 1 space
        total_w = len(text) * char_w - scale
        x0 = int(cx - total_w / 2)
        y0 = int(cy - (7 * scale) / 2)
        for i, ch in enumerate(text):
            gr = glyph_rows(ch)
            ox = x0 + i * char_w
            for r in range(7):
                for c in range(5):
                    if gr[r][c] == '1':
                        for dy in range(scale):
                            for dx in range(scale):
                                xx = ox + c * scale + dx
                                yy = y0 + r * scale + dy
                                if 0 <= xx < W and 0 <= yy < H:
                                    buf[yy][xx] = color

    draw_text("OpenCompare", W//2, 90, WHITE, scale=6)
    draw_text(f"{a} vs {b}", W//2, 290, WHITE, scale=10)
    draw_text(winner, W//2, 410, PINK, scale=8)
    draw_text(f"{sa}", W//2 - 180, 500, WHITE, scale=8)
    draw_text(f"{sb}", W//2 + 180, 500, WHITE, scale=8)
    draw_text("Compare anything. Decide smarter.", W//2, 580, WHITE, scale=4)

    # encode PNG (truecolor 8-bit)
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type 0
        for x in range(W):
            r, g, b = buf[y][x]
            raw += bytes((r, g, b))
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    return png


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(body, bytes):
            self.wfile.write(body)
        else:
            self.wfile.write(str(body).encode())

    def do_GET(self):
        global LAST_ERROR
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
        elif p.path == "/api/trending":
            # most recent comparisons first (ephemeral store; best-effort)
            recent = list(reversed(COMPARES[-50:]))
            self._send(200, json.dumps(recent))
        elif p.path == "/trending":
            self._send(200, render(), "text/html")
        elif p.path.startswith("/cat/"):
            self._send(200, render(), "text/html")
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
            try:
                import PIL
                pil_ok = f"ok {PIL.__version__}"
            except Exception as pe:
                pil_ok = f"FAIL: {type(pe).__name__}: {str(pe)[:100]}"
            self._send(200, json.dumps({
                "mistral_key_present": bool(MISTRAL_KEY),
                "mistral_key_len": len(MISTRAL_KEY),
                "mistral_model": MISTRAL_MODEL,
                "port": os.environ.get("PORT", "unset"),
                "network_to_mistral": net_ok,
                "pil": pil_ok,
                "last_error": LAST_ERROR,
            }))
        elif p.path == "/health":
            self._send(200, '{"status":"ok"}')
        elif p.path.startswith("/c/"):
            slug = p.path[3:].strip("/")
            html = render()
            # server-side OG injection so social scrapers (WhatsApp/X/Telegram) see
            # the per-comparison image WITHOUT running JS
            q = urllib.parse.parse_qs(p.query)
            a = (q.get("a") or [None])[0]
            b = (q.get("b") or [None])[0]
            w = (q.get("w") or [None])[0]
            sa = (q.get("sa") or [None])[0]
            sb = (q.get("sb") or [None])[0]
            if a and b:
                og_url = f"/og/{slug}.png?a={urllib.parse.quote(a)}&b={urllib.parse.quote(b)}" \
                         f"&w={urllib.parse.quote(w or '')}&sa={urllib.parse.quote(sa or '')}&sb={urllib.parse.quote(sb or '')}"
                title = f"{a} vs {b} — OpenCompare verdict"
                desc = f"AI comparison: {a} vs {b}. {w or 'See the verdict'}."
                html = html.replace(
                    '<meta property="og:image" content="https://opencompare.onrender.com/og/opencompare-vs-opencompare.png" id="ogImage">',
                    f'<meta property="og:image" content="https://opencompare.onrender.com{og_url}" id="ogImage">'
                ).replace(
                    '<meta name="twitter:image" content="https://opencompare.onrender.com/og/opencompare-vs-opencompare.png" id="twImage">',
                    f'<meta name="twitter:image" content="https://opencompare.onrender.com{og_url}" id="twImage">'
                ).replace(
                    '<meta property="og:title" content="OpenCompare — Compare Anything. Decide Smarter.">',
                    f'<meta property="og:title" content="{title}">'
                ).replace(
                    '<meta name="twitter:title" content="OpenCompare — Compare Anything. Decide Smarter.">',
                    f'<meta name="twitter:title" content="{title}">'
                ).replace(
                    '<meta property="og:description" content="Free AI comparison engine. Drop in two items and get an instant, neutral, detailed side-by-side comparison.">',
                    f'<meta property="og:description" content="{desc}">'
                ).replace(
                    '<meta name="twitter:description" content="Free AI comparison engine. Drop in two items and get an instant, neutral, detailed side-by-side comparison.">',
                    f'<meta name="twitter:description" content="{desc}">'
                )
            self._send(200, html, "text/html")
        elif p.path.startswith("/og/") and p.path.endswith((".svg", ".png")):
            slug = p.path[len("/og/"):].rsplit(".", 1)[0].strip("/")
            try:
                data = og_image(slug, p.query)
                self._send(200, data, "image/png")
            except Exception as e:
                LAST_ERROR = f"og_image: {type(e).__name__}: {str(e)[:200]}"
                # fallback: 1x1 transparent PNG so social cards never 502
                fallback = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6360000002000154a24f500000000049454e44ae426082")
                self._send(200, fallback, "image/png")
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
