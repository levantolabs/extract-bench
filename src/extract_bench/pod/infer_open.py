"""GPU side of `extract-bench open`: run one open extraction model over an exported requests file.

  python -m extract_bench.pod.infer_open <model: nuextract3|schematron3b|gliner25> <requests.jsonl> <results.jsonl>

Needs a CUDA GPU and `pip install vllm "gliner2[local]"` (the published results: vLLM 0.32.0, bf16, one NVIDIA A40;
gliner2 2.1.0). Standalone: it imports nothing from the rest of the package, so it can be copied to the GPU box alone.

Inference only: prompts/templates/schemas were built by the harness; this script renders the model's own chat template,
generates greedily (temperature 0), and writes the raw output per item. vLLM for the two LLMs, gliner2 for GLiNER2.5.
"""
import json, sys, time

MODEL, REQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HF = {"nuextract3": "numind/NuExtract3", "schematron3b": "inference-net/Schematron-3B", "gliner25": "fastino/gliner2.5-base-v1"}[MODEL]
MAX_LEN, MAX_NEW = 32768, 4096
recs = [json.loads(l) for l in open(REQ)]
todo = [r for r in recs if not r.get("empty")]
results = {}
pod = {"model": HF}


def put(r, **kw):
    results[r["id"]] = {"id": r["id"], "dataset": r["dataset"], "arm": r["arm"], "text_source": r.get("text_source"),
                        "empty": r.get("empty", False), "pod": pod, **kw}


def main():
    if MODEL in ("nuextract3", "schematron3b"):
        import vllm
        from vllm import LLM, SamplingParams
        pod["vllm"] = vllm.__version__
        if MODEL == "nuextract3":
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
            render = lambda q: proc.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True,
                                                        template=q["template"], instructions=q["instructions"], enable_thinking=False)
            tok = proc.tokenizer
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(HF)
            render = lambda q: tok.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True)
        llm = LLM(model=HF, trust_remote_code=True, max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=20)

        def sp(q):
            if MODEL != "schematron3b":
                return SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
            try:
                from vllm.sampling_params import StructuredOutputsParams
                return SamplingParams(temperature=0.0, max_tokens=MAX_NEW, structured_outputs=StructuredOutputsParams(json=q["json_schema"]))
            except ImportError:
                from vllm.sampling_params import GuidedDecodingParams
                return SamplingParams(temperature=0.0, max_tokens=MAX_NEW, guided_decoding=GuidedDecodingParams(json=q["json_schema"]))

        prompts, params, keep = [], [], []
        for r in todo:
            p = render(r["request"])
            n = len(tok(p, add_special_tokens=False)["input_ids"])
            if n > MAX_LEN - MAX_NEW:
                put(r, error=f"prompt too long ({n} tokens)", n_calls=0)
                continue
            prompts.append(p); params.append(sp(r["request"])); keep.append((r, n))
        t0 = time.time()
        outs = llm.generate(prompts, params)
        per = (time.time() - t0) * 1000 / max(1, len(outs))
        for (r, n), o in zip(keep, outs):
            put(r, text=o.outputs[0].text, finish=o.outputs[0].finish_reason, prompt_tokens=n,
                out_tokens=len(o.outputs[0].token_ids), wall_ms=per, n_calls=1, error=None)
    else:
        from gliner2 import AutoExtractor
        m = AutoExtractor.from_pretrained(HF, map_location="cuda")
        pod["gliner2"] = __import__("gliner2").__version__ if hasattr(__import__("gliner2"), "__version__") else "unknown"
        CH, OV = 1800, 300  # characters per chunk / overlap (encoder window is a few hundred tokens)
        for r in todo:
            q = r["request"]; text = q["text"]; t0 = time.time()
            starts = list(range(0, max(1, len(text) - OV), CH - OV)) or [0]
            spans, calls = {k: [] for k in q["labels"]}, 0
            try:
                for s in starts:
                    chunk = text[s:s + CH]
                    res = m.extract_entities(chunk, q["labels"], include_confidence=True, include_spans=True)
                    calls += 1
                    for lab, lst in ((res or {}).get("entities") or {}).items():
                        for e in lst or []:
                            spans.setdefault(lab, []).append({"text": e.get("text"), "start": (e.get("start") or 0) + s,
                                                              "end": (e.get("end") or 0) + s, "confidence": e.get("confidence")})
                put(r, spans=spans, wall_ms=(time.time() - t0) * 1000, n_calls=calls, error=None)
            except Exception as ex:  # noqa: BLE001
                put(r, error=f"{type(ex).__name__}: {str(ex)[:200]}", n_calls=calls)
    for r in recs:
        if r.get("empty"):
            put(r)
    with open(OUT, "w") as fh:
        for r in recs:
            fh.write(json.dumps(results[r["id"]], ensure_ascii=False) + "\n")
    print(MODEL, "done", len(results), "results; errors", sum(1 for v in results.values() if v.get("error")))


if __name__ == "__main__":  # vLLM starts its engine with spawn, which re-imports this file
    main()
