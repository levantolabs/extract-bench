"""Fail if any tracked or staged text file contains a key-like string. Used by CI on every pull request.

  python scripts/check_secrets.py [paths...]      # default: every file `git ls-files` lists
"""
import re, subprocess, sys
from pathlib import Path

PATTERNS = {
    "openai/openrouter key": r"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}",
    "anthropic key": r"\bsk-ant-[A-Za-z0-9_-]{20,}",
    "github token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "hugging face token": r"\bhf_[A-Za-z0-9]{30,}",
    "aws access key": r"\bAKIA[0-9A-Z]{16}\b",
    "google api key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "bearer literal": r"(?i)authorization[\"']?\s*[:=]\s*[\"']bearer\s+[A-Za-z0-9._-]{20,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "assigned secret": r"(?i)\b(?:api[_-]?key|secret|token)\b\s*[:=]\s*[\"'][A-Za-z0-9._-]{24,}[\"']",
}
SKIP = re.compile(r"\.(png|jpe?g|webp|pdf|gz|zip|xz)$")


def main(paths):
    files = paths or subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    hits = 0
    for f in files:
        if SKIP.search(f) or not Path(f).is_file():
            continue
        try:
            text = Path(f).read_text(errors="ignore")
        except OSError:
            continue
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                print(f"{f}:{line}: possible {name}")
                hits += 1
    print("no key-like strings found" if not hits else f"{hits} possible secret(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
