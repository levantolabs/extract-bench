"""Print the structure of one VRDU dataset.jsonl record (read-only)."""
import gzip, json, sys
f = sys.argv[1]
with gzip.open(f, "rt") as fh:
    rec = json.loads(fh.readline())
def shape(o, depth=0, maxd=4):
    if depth > maxd: return "..."
    if isinstance(o, dict): return {k: shape(v, depth+1) for k, v in o.items()}
    if isinstance(o, list): return [shape(o[0], depth+1)] + ([f"...len={len(o)}"] if len(o) > 1 else []) if o else []
    if isinstance(o, str): return f"str[{len(o)}]"
    return type(o).__name__
print(json.dumps(shape(rec), indent=1)[:3000])
print("filename:", rec.get("filename"))
print("ocr keys:", list(rec["ocr"].keys()) if isinstance(rec.get("ocr"), dict) else type(rec.get("ocr")))
print("annotations:", json.dumps(rec["annotations"])[:2500])
o = rec["ocr"]
print("text head:", repr(o.get("text", "")[:600]))
