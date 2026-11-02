import json, os, re, collections
D="datasets/sroie"
items=[json.loads(l) for l in open(f"{D}/items.jsonl")]
errs=[]; c=collections.Counter()
ids=set()
TOP={"id","dataset","use_case","modality","text","text_source","images","fields","metric","meta"}
for it in items:
    if set(it)!=TOP: errs.append((it["id"],"keys",set(it)^TOP))
    if it["id"] in ids: errs.append((it["id"],"dup"))
    ids.add(it["id"])
    if it["dataset"]!="sroie": errs.append((it["id"],"dataset"))
    if it["modality"]!="image" or it["text"] is not None or it["text_source"]!="none": errs.append((it["id"],"text/modality"))
    if it["metric"] not in "exact_norm anls token_f1 numeric_tol date_exact list_f1 edit_sim".split(): errs.append((it["id"],"metric"))
    if not (1<=len(it["images"])<=4): errs.append((it["id"],"nimg"))
    for p in it["images"]:
        fp=os.path.join(D,p)
        if not os.path.isfile(fp): errs.append((it["id"],"missing",p)); continue
        b=open(fp,"rb").read(16)
        sz=os.path.getsize(fp)
        if sz>4*1024*1024: errs.append((it["id"],"big",sz))
        if not (b[:3]==b"\xff\xd8\xff" or b[:8]==b"\x89PNG\r\n\x1a\n" or (b[:4]==b"RIFF" and b[8:12]==b"WEBP")): errs.append((it["id"],"magic"))
        c["maxsz"]=max(c["maxsz"],sz)
    for f in it["fields"]:
        if set(f)!={"name","type","instructions","gold"}: errs.append((it["id"],"fkeys"))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",f["name"]): errs.append((it["id"],"fname"))
        if f["type"] not in "text number money date time datetime duration list".split(): errs.append((it["id"],"ftype"))
        if f["gold"] is None: c["nullgold"]+=1
        if isinstance(f["gold"],list): c["listgold"]+=1
        if f["type"]=="date":
            gs=f["gold"] if isinstance(f["gold"],list) else [f["gold"]]
            for g in gs:
                if not re.fullmatch(r"\d{4}-\d\d-\d\d",g): errs.append((it["id"],"date fmt",g))
        if f["type"]=="money" and not re.fullmatch(r"-?\d+\.\d\d",f["gold"]): errs.append((it["id"],"money",f["gold"]))
        c["fields"]+=1
    for f in it["meta"]["comparable"]["fields"]:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}",f["name"]) or f["type"]!="text": errs.append((it["id"],"cmp field"))
    c["holdout"]+=it["meta"]["holdout"]
print(len(items), dict(c), errs[:20], len(errs))
# check img dir has no extra files
print(len(os.listdir(f"{D}/img")))
# label.json consistency
lab=json.load(open(f"{D}/comparable/label.json")); print("labels",len(lab))
