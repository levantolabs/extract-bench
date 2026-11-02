import csv, io, json, random, hashlib, base64, os, re, sys
from datetime import date
D="datasets/sroie"
tsvb=open(f"{D}/raw/cc-ocr/sroie2019_word_347.tsv","rb").read()
print("tsv sha", hashlib.sha256(tsvb).hexdigest(), len(tsvb))
csv.field_size_limit(1<<30)
rows={r["image_name"]:r for r in csv.DictReader(io.StringIO(tsvb.decode()),delimiter="\t")}
items=[json.loads(l) for l in open(f"{D}/items.jsonl")]
mir={}
for l in open(f"{D}/raw/jsdnrs-mirror/test_text.jsonl"):
    r=json.loads(l); mir[r["key"]]=r
print("mirror keys sample", list(mir.values())[0].keys())
MON={m:i for i,m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(),1)}
def mydate(s):
    s=s.strip()
    m=re.fullmatch(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})",s)
    if m: return date(int(m[1]),int(m[2]),int(m[3])).isoformat()
    m=re.fullmatch(r"(\d{1,2})\s*[/\-. ]\s*(\d{1,2})\s*[/\-. ]\s*(\d{2,4})",s)
    if m:
        y=int(m[3]); y=y+2000 if y<100 else y
        try: return date(y,int(m[2]),int(m[1])).isoformat()
        except ValueError: return "INVALID"
    m=re.fullmatch(r"(\d{1,2})\s*[/\-. ]?\s*([A-Za-z]{3})[A-Za-z]*\.?\s*[/\-. ,]?\s*(\d{2,4})",s)
    if m:
        y=int(m[3]); y=y+2000 if y<100 else y
        return date(y,MON[m[2].lower()],int(m[1])).isoformat()
    return "UNPARSED"
def mymoney(s):
    t=s.replace("RM","").replace("$","").replace(" ","").replace("MYR","")
    t=re.sub(r",(?=\d{3}\b)","",t)
    return f"{float(t):.2f}"
rnd=random.Random(7); pick=rnd.sample(range(len(items)),15)
mism=0
for i in pick:
    it=items[i]; name=it["meta"]["image_name"]; r=rows[name]; g=json.loads(r["answer"])
    stem=name[:-4]; me=mir[stem]["entities"]
    f={x["name"]:x["gold"] for x in it["fields"]}
    probs=[]
    for k in ("company","address"):
        if f[k]!=g[k]: probs.append(f"{k} tsv")
        if f[k]!=me.get(k): probs.append(f"{k} mirror")
    md=mydate(g["date"]); 
    if f["date"]!=md and not (isinstance(f["date"],list)): probs.append(f"date {f['date']} vs mine {md} raw {g['date']}")
    if f["total"]!=mymoney(g["total"]): probs.append(f"total {f['total']} vs {g['total']}")
    if g["date"]!=me.get("date") or g["total"]!=me.get("total"): probs.append("mirror date/total differ")
    if it["meta"]["gold_verbatim"]!=g: probs.append("gold_verbatim")
    cf={x["name"]:x["gold"] for x in it["meta"]["comparable"]["fields"]}
    if cf!=g: probs.append("comparable gold")
    if it["meta"]["comparable"]["prompt"]!=r["question"]: probs.append("prompt")
    ib=open(f"{D}/{it['images'][0]}","rb").read()
    if ib!=base64.b64decode(r["image"]): probs.append("image bytes")
    mism+=bool(probs)
    print(i, it["id"], json.dumps(g)[:160], "| typed date", f["date"], "total", f["total"], "|", probs or "OK")
print("items with mismatch", mism)
# whole-set independent date/money check
bad=[]
for it in items:
    g=it["meta"]["gold_verbatim"]; f={x["name"]:x["gold"] for x in it["fields"]}
    md=mydate(g["date"])
    if f["date"]!=md: bad.append((it["id"],g["date"],f["date"],md))
    if f["total"]!=mymoney(g["total"]): bad.append((it["id"],g["total"],f["total"]))
print("full-set typed diffs vs my converter:", bad)
