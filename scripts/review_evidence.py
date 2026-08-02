"""Evidence loader for the 478-agent review corpus: the source of record for the refactor.

Each refactor phase reads its own cluster's raw agent output through here rather than through any
summary. The post-processed JSON is a ~52% compression of the journal: it is lossless for confirmed
findings and carries the lens-tagged corrections, but it drops 412 of 456 verifier ``reasoning``
records (~1.31M chars), which is where the greps, executed reproductions, and falsification attempts
that prove a defect at HEAD actually live. So: spine from the JSON, evidence from the journal.

    python scripts/review_evidence.py                 # corpus integrity self-check
    python scripts/review_evidence.py K5              # one cluster, summarised
    python scripts/review_evidence.py K5 --full       # + every verifier's full reasoning

    from scripts.review_evidence import cluster, evidence, CLUSTERS
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / ".review-corpus"
JOURNAL = CORPUS / "journal.jsonl"
POST = CORPUS / "post_processed.json"

# Path prefixes, kept abbreviated in CLUSTERS so the table stays readable.
_ABBREV = {
    "MCP/": "packages/tcip-mcp/src/tcip_mcp/",
    "WEB/": "packages/tcip-web/src/tcip_web/",
    "FE/": "packages/tcip-web/frontend/src/",
    "ANN/": "packages/tcip-annotation/src/tcip_annotation/",
}


def _expand(path: str) -> str:
    for short, full in _ABBREV.items():
        if path.startswith(short):
            return full + path[len(short):]
    return path


_LENS = re.compile(r"^(refute|grounding-fit|fix-safety): ")


def _load():
    records = [json.loads(line) for line in JOURNAL.open(encoding="utf-8")]
    results = [d for d in records if d.get("type") == "result"]
    surveys = [d for d in results if "findings" in d["result"]]
    verifiers = [d for d in results if "refuted" in d["result"]]
    post = json.loads(POST.read_text(encoding="utf-8"))["result"]

    # surface_assessment text is the only link from a survey record back to its surface key.
    a2k = {s["assessment"]: s["key"] for s in post["surfaces"]}
    findings = {
        (a2k[s["result"]["surface_assessment"]], f["file"], f["line"]):
            dict(surface=a2k[s["result"]["surface_assessment"]], **f)
        for s in surveys for f in s["result"]["findings"]
    }
    confirmed = {(c["surface"], c["file"], c["line"]): c for c in post["confirmed"]}
    # A correction string is verbatim from the verifier's record, with its lens prefixed.
    by_correction = {(v["result"].get("correction") or ""): v for v in verifiers}
    return findings, confirmed, by_correction, post


FINDINGS, CONFIRMED, _BY_CORRECTION, POST = _load()


def evidence(key: tuple[str, str, int]) -> dict:
    """Full evidence for one finding: its 10 raw fields + all 3 verifiers with reasoning."""
    conf = CONFIRMED[key]
    out = []
    for tagged in conf.get("corrections", []):
        rec = _BY_CORRECTION.get(_LENS.sub("", tagged))
        lens = tagged.split(":", 1)[0]
        if rec is None:
            out.append({"lens": lens, "reasoning": None, "correction": _LENS.sub("", tagged)})
            continue
        r = rec["result"]
        out.append({"lens": lens, "refuted": r["refuted"], "confidence": r.get("confidence"),
                    "reasoning": r.get("reasoning"), "correction": r.get("correction")})
    return {"finding": FINDINGS[key], "refute_count": conf.get("refute_count"), "verifiers": out}


def cluster(name: str) -> list[dict]:
    return [evidence(k) for k in CLUSTERS[name]]


def refuted_full(substr: str = "") -> list[dict]:
    """The 21 refuted findings. Their bodies exist only in the journal; the JSON keeps a title."""
    out = []
    for r in POST["refuted"]:
        match = [f for k, f in FINDINGS.items()
                 if k[0] == r["surface"] and k[1] == r["file"] and k not in CONFIRMED]
        if substr and substr not in r["title"]:
            continue
        out.append({"title": r["title"], "file": r["file"], "refute_count": r["refute_count"],
                    "why_refuted": r.get("why_refuted"), "journal_body": match})
    return out


# ── Clusters ────────────────────────────────────────────────────────────────
# Derived from co-occurrence in the corpus: shared file, root cause named by >=2 independent
# surveys, or shared proposed fix. Exact partition of the 131 confirmed findings.
CLUSTERS: dict[str, list[tuple[str, str, int]]] = {
"K1": [("pipelines-derivations","MCP/tools/inference_tools.py",45),
       ("validation-rails","MCP/tools/inference_tools.py",46),
       ("defensibility-provenance","MCP/tools/inference_tools.py",45),
       ("goldens-and-tests","tests/test_char_goldens_measurement.py",119),
       ("data-layer","MCP/pipelines/data/splits.py",51),
       ("data-layer","MCP/tools/data_tools.py",324)],
"K2": [("skills-training-eval","MCP/pipelines/operating_point.py",156),
       ("pipelines-derivations","MCP/pipelines/operating_point.py",175),
       ("pipelines-derivations","MCP/pipelines/operating_point.py",158),
       ("pipelines-derivations","MCP/pipelines/operating_point.py",163),
       ("pipelines-derivations","MCP/pipelines/operating_point.py",150),
       ("pipelines-derivations","MCP/pipelines/training/evaluation.py",420),
       ("validation-rails","MCP/pipelines/operating_point.py",174),
       ("validation-rails","MCP/pipelines/operating_point.py",103),
       ("validation-rails","MCP/pipelines/feedback/review_calibration.py",77),
       ("validation-rails","MCP/pipelines/feedback/review_calibration.py",155),
       ("defensibility-provenance","MCP/pipelines/feedback/review_calibration.py",127),
       ("web-backend-routes","WEB/routes/review.py",719),
       ("web-backend-routes","WEB/routes/review.py",717)],
"K3": [("skills-phenology-delivery","MCP/tools/phenology_tools.py",296),
       ("measurement-phenology","MCP/tools/phenology_tools.py",295),
       ("measurement-core","MCP/pipelines/postprocessing/aggregation.py",229)],
"K4": [("measurement-core","MCP/pipelines/postprocessing/aggregation.py",121),
       ("measurement-core","MCP/pipelines/postprocessing/aggregation.py",133),
       ("measurement-phenology","MCP/pipelines/postprocessing/phenology.py",133),
       ("measurement-phenology","MCP/pipelines/postprocessing/phenology.py",252),
       ("measurement-phenology","MCP/pipelines/postprocessing/phenology.py",142),
       ("goldens-and-tests","tests/test_char_goldens_measurement.py",193)],
"K5": [("skills-phenology-delivery","MCP/tools/phenology_tools.py",246),
       ("skills-phenology-delivery","WEB/routes/results.py",260),
       ("mcp-tool-surface","MCP/tools/phenology_tools.py",246),
       ("measurement-phenology","MCP/tools/phenology_tools.py",246),
       ("residual-elongated-class-id","MCP/tools/phenology_tools.py",246),
       ("residual-elongated-class-id","MCP/pipelines/postprocessing/phenology.py",177),
       ("residual-elongated-class-id","MCP/pipelines/postprocessing/phenology.py",284),
       ("residual-elongated-class-id","MCP/pipelines/postprocessing/phenology.py",75),
       ("residual-elongated-class-id","WEB/routes/results.py",260)],
"K6": [("skills-phenology-delivery","WEB/routes/results.py",185),
       ("honest-scope","WEB/routes/results.py",185),
       ("honest-scope","WEB/routes/results.py",121),
       ("web-backend-routes","WEB/routes/results.py",121),
       ("web-backend-routes","WEB/routes/results.py",14),
       ("residual-elongated-class-id","WEB/routes/results.py",121),
       ("measurement-phenology","MCP/tools/phenology_tools.py",249),
       ("skills-meta-docs","README.md",121)],
"K7": [("skills-toolkit-inventory",".github/skills/toolkit-inventory/SKILL.md",51),
       ("skills-toolkit-inventory","MCP/pipelines/inference/generic_predictor.py",257),
       ("skills-toolkit-inventory","MCP/pipelines/components/heads.py",181),
       ("skills-toolkit-inventory","MCP/pipelines/components/losses.py",293),
       ("skills-toolkit-inventory","MCP/pipelines/components/detectors.py",51),
       ("pipelines-model-build","MCP/pipelines/components/detectors.py",51),
       ("pipelines-model-build","MCP/pipelines/components/heads.py",181),
       ("pipelines-model-build","MCP/pipelines/model_contract.py",66),
       ("pipelines-model-build","MCP/pipelines/components/backbones.py",138),
       ("pipelines-derivations","MCP/pipelines/derivations.py",69),
       ("mcp-tool-surface","MCP/tools/feedback_tools.py",110)],
"K8": [("pipelines-model-build","MCP/pipelines/components/detectors.py",48),
       ("honest-scope","MCP/pipelines/components/detectors.py",48),
       ("honest-scope","MCP/pipelines/data/datasets.py",393),
       ("data-layer","MCP/pipelines/data/datasets.py",393)],
"K9": [("skills-training-eval","MCP/pipelines/training/generic_trainer.py",305),
       ("skills-training-eval",".github/skills/training/SKILL.md",8),
       ("skills-training-eval",".github/skills/evaluation/SKILL.md",30),
       ("skills-training-eval",".github/skills/evaluation/SKILL.md",57),
       ("skills-training-eval",".github/skills/evaluation/SKILL.md",45),
       ("honest-scope","MCP/pipelines/training/generic_trainer.py",303),
       ("training-loop-hpo","MCP/pipelines/training/generic_trainer.py",625),
       ("training-loop-hpo","MCP/pipelines/training/generic_trainer.py",69),
       ("training-loop-hpo",".github/skills/training/SKILL.md",27),
       ("training-loop-hpo",".github/skills/training/SKILL.md",50),
       ("mcp-tool-surface","MCP/tools/model_tools.py",78)],
"K10":[("mcp-tool-surface","MCP/tools/training_tools.py",909),
       ("goldens-and-tests","MCP/tools/training_tools.py",911),
       ("pipelines-derivations","MCP/pipelines/resolution.py",51)],
"K11":[("training-loop-hpo","MCP/tools/training_tools.py",544),
       ("training-loop-hpo","MCP/tools/training_tools.py",546),
       ("training-loop-hpo","MCP/tools/training_tools.py",420),
       ("mcp-tool-surface","MCP/tools/training_tools.py",545),
       ("training-loop-hpo","MCP/pipelines/training/generic_trainer.py",219),
       ("training-loop-hpo","MCP/pipelines/training/envelope.py",351)],
"K12":[("pipelines-model-build","MCP/pipelines/model_build.py",196),
       ("defensibility-provenance","MCP/pipelines/model_build.py",206),
       ("defensibility-provenance","MCP/tools/training_tools.py",580),
       ("defensibility-provenance","MCP/experiments.py",169),
       ("defensibility-provenance","MCP/tools/inference_tools.py",289),
       ("web-backend-routes","MCP/audit.py",77),
       ("goldens-and-tests","tests/test_char_goldens_measurement.py",485),
       ("skills-meta-docs",".github/skills/self-improvement/SKILL.md",16)],
"K13":[("data-layer","MCP/pipelines/data/datasets.py",246),
       ("data-layer","MCP/pipelines/data/datasets.py",755),
       ("skills-annotation-visual","FE/App.tsx",198),
       ("skills-annotation-visual",".github/skills/annotation/SKILL.md",156),
       ("skills-annotation-visual",".github/skills/annotation/SKILL.md",27)],
"K14":[("skills-pipeline-design",".github/skills/pipeline-design/SKILL.md",20),
       ("skills-pipeline-design",".github/skills/pipeline-design/SKILL.md",171),
       ("skills-pipeline-design",".github/skills/pipeline-design/SKILL.md",188),
       ("skills-toolkit-inventory","MCP/pipelines/components/temporal.py",30),
       ("skills-toolkit-inventory","MCP/pipelines/components/backbones.py",209),
       ("pipelines-model-build","MCP/pipelines/components/backbones.py",209),
       ("skills-phenology-delivery",".github/skills/phenology/SKILL.md",42),
       ("skills-meta-docs",".github/skills/project-setup/SKILL.md",66),
       ("skills-crops",".github/skills/crops/elderberry/SKILL.md",164)],
"K15":[("gui-breeder-ux","FE/api/inference.ts",180),
       ("residual-elongated-class-id","FE/api/inference.ts",174),
       ("web-backend-routes","WEB/routes/results.py",274),
       ("gui-breeder-ux","FE/tabs/InferenceTab.tsx",34),
       ("gui-breeder-ux","FE/tabs/TuningTab.tsx",13),
       ("gui-breeder-ux","FE/tabs/ReviewTab.tsx",50),
       ("gui-breeder-ux","FE/tabs/ReviewTab.tsx",838),
       ("gui-breeder-ux","FE/tabs/ReviewTab.tsx",936),
       ("gui-breeder-ux","FE/tabs/ResultsTab.tsx",376),
       ("web-backend-routes","FE/tabs/ResultsTab.tsx",51)],
"K16":[("skills-annotation-visual","ANN/state.py",13),
       ("keypoint-capability-gap","ANN/state.py",27),
       ("skills-annotation-visual","MCP/tools/annotation_tools.py",557),
       ("keypoint-capability-gap","MCP/tools/annotation_tools.py",557),
       ("keypoint-capability-gap","MCP/pipelines/measurement/mask_geometry.py",3),
       ("keypoint-capability-gap",".github/skills/annotation/SKILL.md",81),
       ("keypoint-capability-gap","FE/store/types.ts",9)],
"K17":[("measurement-core","MCP/pipelines/measurement/mask_geometry.py",105),
       ("measurement-core","MCP/pipelines/measurement/mask_geometry.py",31),
       ("measurement-core","MCP/pipelines/postprocessing/aggregation.py",243),
       ("measurement-core","MCP/pipelines/postprocessing/aggregation.py",93)],
"K18":[("skills-crops",".github/skills/crops/chestnut/SKILL.md",44),
       ("skills-crops","scripts/verify_skill_traits.py",63),
       ("skills-crops",".github/skills/crops/crops.yml",644),
       ("skills-crops",".github/skills/crops/black-locust/SKILL.md",66),
       ("skills-crops",".github/skills/crops/elderberry/SKILL.md",87)],
"K19":[("web-backend-routes","WEB/routes/classes.py",230),
       ("web-backend-routes","WEB/agent_bash_guard.py",43),
       ("web-backend-routes","WEB/routes/training.py",109)],
}
CLUSTERS = {k: [(s, _expand(f), ln) for s, f, ln in v] for k, v in CLUSTERS.items()}


def selfcheck() -> int:
    """Prove the corpus and the cluster partition are intact. Exit non-zero if not."""
    problems = []
    assigned = [k for keys in CLUSTERS.values() for k in keys]
    if len(assigned) != len(set(assigned)):
        problems.append("a finding is assigned to more than one cluster")
    missing = [k for k in assigned if k not in CONFIRMED]
    if missing:
        problems.append(f"{len(missing)} cluster keys do not resolve: {missing[:5]}")
    unassigned = [k for k in CONFIRMED if k not in set(assigned)]
    if unassigned:
        problems.append(f"{len(unassigned)} confirmed findings unassigned: {unassigned[:5]}")
    reasoning = sum(1 for k in CONFIRMED for v in evidence(k)["verifiers"] if v.get("reasoning"))

    print(f"surveys/verifiers      : 22 / {len([1 for _ in _BY_CORRECTION])} correction-keyed")
    print(f"findings (unique keys) : {len(FINDINGS)}")
    print(f"confirmed / refuted    : {len(CONFIRMED)} / {len(POST['refuted'])}")
    print(f"clusters               : {len(CLUSTERS)} covering {len(assigned)} findings")
    print(f"verifier reasoning recovered from journal: {reasoning}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print("OK" if not problems else "FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(selfcheck())
    name = args[0]
    full = "--full" in sys.argv
    for item in cluster(name):
        f = item["finding"]
        print(f"\n{'='*78}\n[{f['severity']}] {f['file']}:{f['line']}  ({f['surface']})")
        print(f"  {f['title']}")
        print(f"  refuted by {item['refute_count']}/3")
        print(f"  evidence : {f['evidence'][:300]}")
        print(f"  fix      : {f['proposed_fix'][:300]}")
        for v in item["verifiers"]:
            print(f"  -- {v['lens']} (refuted={v.get('refuted')}, {v.get('confidence')})")
            if v.get("correction"):
                print(f"     correction: {v['correction'][:400]}")
            if full and v.get("reasoning"):
                print(f"     reasoning : {v['reasoning']}")
