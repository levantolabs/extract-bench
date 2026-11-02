# One-off: fill the numeric placeholders of CARD.md from build_report.json + selfcheck.out.json + du of raw/.
import json, os, subprocess, sys
D = sys.argv[1]
rep = json.load(open(os.path.join(D, "build_report.json")))
sc = json.load(open(os.path.join(D, "selfcheck.out.json")))
card = open(os.path.join(D, "CARD.md")).read()
rows = []
for t, m in rep["tag_marginals"].items():
    rows.append(f"  | {t} | {m['val']:,} | {m['expected_in_P']} | {m['in_P']} | {m['in_all_350']} |")
g = rep["gold_reachable_sample"]
raw_mb = int(subprocess.check_output(["du", "-sk", os.path.join(D, "raw")]).split()[0]) / 1024
rep_txt = {
    "  TAGTABLE": "\n".join(rows),
    "REACH_VAL": f"{rep['gold_reachable_val_pct']}% of 5,349",
    "REACH_SAMPLE": (f"{g['all_350']['pct']}% of all 350; P block {g['proportional']['pct']}%; "
                     f"handwritten items {g['handwritten_all']['pct']}% (n={g['handwritten_all']['n']}); "
                     f"non-handwritten {g['non_handwritten']['pct']}% (n={g['non_handwritten']['n']})"),
    "REACH_ALNUM_VAL": f"{rep['gold_reachable_alnum_val_pct']}%",
    "IMGLINE": (f"Re-encoded: {rep['images_reencoded']} of {rep['n_docs']}. "
                f"Largest file {rep['image_bytes_max'] / 2**20:.2f} MiB."),
    "TOKLINE": (f"p50 {sc['text_o200k']['p50']}, p95 {sc['text_o200k']['p95']}, max {sc['text_o200k']['max']}, "
                f"mean {sc['text_o200k']['mean']}. With the question added: p50 {sc['text_plus_instructions_o200k']['p50']}, "
                f"p95 {sc['text_plus_instructions_o200k']['p95']}"),
    "IMGTOT": f"{sc['n_images']} PNGs, {sc['image_bytes']['total_mb']} MB in total, p50 {sc['image_bytes']['p50'] / 1e3:.0f} KB per page",
    "RAWMB": f"{raw_mb:.0f} MB",
}
for k, v in rep_txt.items():
    if k not in card: sys.exit(f"placeholder {k} missing")
    card = card.replace(k, v)
open(os.path.join(D, "CARD.md"), "w").write(card)
print("filled")
