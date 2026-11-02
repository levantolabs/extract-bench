"""Stats over the VRDU mixed-template test sets (read-only): entity multiplicity, absences, pages, line items."""
import gzip, json, sys
from collections import Counter
from pathlib import Path
raw = Path(sys.argv[1])
for corpus, split in (("registration-form", "FARA-lv2-mixed_template-train_10-test_300-valid_100-SD_0.json"),
                      ("ad-buy-form", "DeepForm-mixed_template-train_10-test_300-valid_100-SD_0.json")):
    test = set(json.loads((raw / corpus / "few_shot-splits" / split).read_text())["test"])
    meta = json.loads((raw / corpus / "main" / "meta.json").read_text())
    ents = list(meta["entity_name_to_match_func"])
    pat = meta.get("entity_appearance_pattern", {e: "unrepeated" for e in ents})
    present, multi_item, mentions, pages, nli, distinct_vals = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    nested_names = Counter(); n = 0; first_nested = None; file_paths = Counter(); seen = set()
    with gzip.open(raw / corpus / "main" / "dataset.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["filename"] not in test:
                continue
            n += 1; seen.add(r["filename"])
            pages[len(r["ocr"]["pages"])] += 1
            file_paths[r["file_path"].split("/")[0]] += 1
            by = Counter()
            k_line = 0
            for name, ments in r["annotations"]:
                if isinstance(name, list):
                    nested_names[tuple(sorted(name))] += 1; k_line += 1
                    if first_nested is None: first_nested = (name, ments)
                    continue
                by[name] += 1
                mentions[len(ments)] += 1
                vals = {" ".join(m[0].split()) for m in ments}
                if len(vals) > 1: distinct_vals[name] += 1
            for e in ents:
                if pat.get(e) == "unrepeated" and by[e]: present[e] += 1
                if by[e] > 1: multi_item[e] += 1
            nli[min(k_line, 50) if k_line < 50 else 50] += 1
    print(f"== {corpus}: test docs found {n} / {len(test)} (missing {len(test - seen)})")
    print("  pages:", sorted(pages.items()))
    print("  file_path prefix:", file_paths.most_common(3))
    print("  present (unrepeated):", {e: present[e] for e in ents if pat.get(e) == 'unrepeated'})
    print("  docs with >1 entity item of same name:", dict(multi_item))
    print("  mentions per entity item:", sorted(mentions.items()))
    print("  entity items whose mentions have >1 distinct text:", dict(distinct_vals))
    if nested_names:
        print("  nested name tuples:", nested_names.most_common(8))
        li = sorted(nli.items()); print("  line items per doc (50=50+):", li)
        print("  first nested example:", json.dumps(first_nested)[:900])
