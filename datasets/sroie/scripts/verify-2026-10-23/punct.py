import json,re,unicodedata,sys
D="datasets/sroie"
def cc(text):
    text = re.sub(r'(?<![0-9])\s*([,.!?:;])\s*', r'\1 ', text)
    text = re.sub(r'(?<=[0-9])(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)
    return re.sub(r'\s+', ' ', text)
def nt(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold(); s=re.sub(r"\s+"," ",s).strip(); return s.strip(" .,;:!?\"'`()[]{}")
cnt={}
for l in open(f"{D}/items.jsonl"):
    it=json.loads(l)
    for k in ("company","address"):
        g=it["meta"]["gold_verbatim"][k]
        if nt(cc(g))!=nt(g): cnt.setdefault(k,[]).append(g)
for k,v in cnt.items(): print(k,len(v),v[:4])
