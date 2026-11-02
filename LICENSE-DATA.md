# Data licences

The code and docs are under [Apache-2.0](LICENSE).

Each dataset in `datasets/` keeps the licence of its source. We offer our own additions under CC BY 4.0, as far as we
hold rights in them; we'll confirm this before the public release. Our additions are the field mappings, the
`instructions`, the gold audits, the splits, the OCR text we produced and the predictions in `results/`. Each
dataset's `CARD.md` has the evidence for its row below.

| Dataset | Source | Licence | What this repo ships |
|---|---|---|---|
| FATURA | Limam et al. 2023, Zenodo 10.5281/zenodo.10371464 | CC BY 4.0 (verified) | Items, Tesseract text, one example image in `docs/images/` |
| CORD | Park et al. 2019, clovaai/cord | CC BY 4.0 (verified) | Items, Tesseract text, CC-OCR comparable labels |
| TAT-QA | Zhu et al. 2021 | CC BY 4.0 (verified) | Items |
| SQuAD 2.0 | Rajpurkar et al. 2018 | CC BY-SA 4.0 (verified) | Items |
| Emails | Willis, LLM-Extraction-Challenge | MIT for the repo; we didn't check the original senders' terms | Items |
| OmniAI OCR | getomni-ai/ocr-benchmark | MIT, but the images include third-party documents that MIT can't relicense | Items and OCR text, no images |
| RealDoc | Extend, RealDocBench | CC BY 4.0 for the questions and answers; the source PDFs' rights are mixed or unverified | Items, Tesseract text, and the PDF text layer where one exists |
| SROIE | ICDAR 2019, via CC-OCR | CC BY 4.0 as a mirror claims it; UNKNOWN at the source | Items, Tesseract text |
| DocVQA | Mathew et al. 2021 | Mirrors say MIT or Apache-2.0; the original terms are UNKNOWN | Items with the official OCR |
| VRDU | Wang et al. 2023 (Google) | UNKNOWN. The repo states no licence; the evaluator code is Apache-2.0 | Items with the official OCR |
| Kleister-NDA | Stanisławek et al. 2021 (Applica) | UNKNOWN. No licence is stated | Items with the PDF text layer |
| Kleister-Charity | Stanisławek et al. 2021 (Applica) | UNKNOWN. No licence is stated | Items with the official OCR |

The repo doesn't ship page images or raw source downloads. Each `build.py` downloads them again from the pinned source
and checks their hashes.

If you hold rights in any of these documents and want something changed, open an issue.
