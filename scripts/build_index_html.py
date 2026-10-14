"""Extract VisionCare HTML from agent transcript and write frontend/index.html."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT = Path(
    r"C:/Users/sudha/.cursor/projects/c-Arafath-A/agent-transcripts"
    "/a57d9cd8-08e1-48fd-8779-769b25094a32/a57d9cd8-08e1-48fd-8779-769b25094a32.jsonl"
)
OUT = ROOT / "frontend" / "index.html"


def extract_html() -> str | None:
    if not TRANSCRIPT.exists():
        return None
    for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        texts: list[str] = []
        msg = obj.get("message", "")
        if isinstance(msg, str):
            texts.append(msg)
        elif isinstance(msg, dict):
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
        elif isinstance(msg, list):
            for part in msg:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
        for text in texts:
            if "<!DOCTYPE html>" not in text:
                continue
            start = text.find("<!DOCTYPE html>")
            end = text.rfind("</html>")
            if start >= 0 and end > start:
                return text[start : end + len("</html>")]
            match = re.search(r"<!DOCTYPE html>.*</html>", text, re.S | re.I)
            if match:
                return match.group(0)
    return None


def main() -> None:
    html = extract_html()
    if not html:
        raise SystemExit("HTML not found in transcript. Paste into frontend/index.html manually.")
    if "</body>" in html and "/js/api.js" not in html:
        html = html.replace("</body>", '<script src="/js/api.js"></script>\n</body>', 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({len(html)} chars)")


if __name__ == "__main__":
    main()
