"""Run the GLNet monitoring studies from the research checkout on a CUDA host."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("execution-monitoring", "adapter-screening"))
    parser.add_argument("--check", action="store_true", help="Check GPU and training dependencies and exit")
    args = parser.parse_args()
    from glnet_trace_monitoring.research_actor import preflight, run_parent_pilot
    preflight()
    if args.check:
        import torch
        print(torch.cuda.get_device_name())
        return
    from glnet_trace_monitoring.research_common import OUTPUT, config, prepare_data, save_json
    from glnet_trace_monitoring.research_monitors import qualify
    prepare_data(config())
    if not (OUTPUT / "006/monitor_qualification.json").exists():
        qualify()
    if args.study == "execution-monitoring":
        from glnet_trace_monitoring.research_trace import run_trace_study
        from glnet_trace_monitoring.research_weights import run_weight_study
        save_json(OUTPUT / "model.json", json.loads((ROOT / "configs/model.json").read_text()))
        if not (OUTPUT / "007/results/summary.json").exists():
            run_parent_pilot()
        if not (OUTPUT / "008/results/summary.json").exists():
            run_trace_study()
        run_weight_study()
    else:
        from huggingface_hub import snapshot_download
        from glnet_trace_monitoring.research_feedback_016b import run_acquisition
        from adapter_monitoring.generate import build_data, stable_hash
        recipe = json.loads((ROOT.parent / "adapter_monitoring/evidence/recipe.json").read_text())["016b"]
        blocks = build_data(recipe["source"], {tuple(pair) for pair in recipe["historical_exclusions"]})
        source = json.loads((ROOT / "configs/source_016.json").read_text())
        if stable_hash(blocks) != source["data_sha256"]:
            raise ValueError("Source corpus differs from the frozen protocol")
        for identity in source["models"].values():
            identity["snapshot"] = snapshot_download(identity["repo_id"], revision=identity["revision"])
        save_json(OUTPUT / "016/data/blocks.json", blocks)
        save_json(OUTPUT / "016/protocol.json", source)
        run_acquisition()


if __name__ == "__main__":
    main()
