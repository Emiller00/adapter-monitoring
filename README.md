# Adapter Monitoring

Recompute the held-out results from **Execution Monitoring with Traces and Adapter Weights** and **Weight-Only Screening of LoRA Adapters** on a CPU. The checkout includes the evidence and requires only Python 3.10 or later.

The experiments apply GLNet, a weight-space architecture introduced by Putterman et al., to synthetic LoRA adapter monitoring. This repository provides the experiment evidence and verification tools. [Putterman et al., *Learning on LoRAs*](https://arxiv.org/abs/2410.04207)

```sh
python3 -m adapter_monitoring
```

For an installed command, run `python3 -m pip install .`, then run `adapter-monitoring audit`. Add `--json` to the audit command for per-monitor counts, thresholds, and individual qualification failures. Run `adapter-monitoring audit /path/to/evidence` to check a compatible bundle.

| Study | Monitor | Detected | False alarms |
| --- | --- | ---: | ---: |
| Execution Monitoring with Traces and Adapter Weights | Trace-only | 211/407 | 0/1,129 |
| Execution Monitoring with Traces and Adapter Weights | Trace plus weight | 393/407 | 10/1,129 |
| Weight-Only Screening of LoRA Adapters, Qwen3-0.6B | GLNet weight-only | 6/6 | 0/6 |
| Weight-Only Screening of LoRA Adapters, Qwen3-1.7B | GLNet weight-only | 6/6 | 0/6 |

The execution monitoring study measures execution-level violations over six held-out adapter blocks. The name “trace-only” follows the study and includes the prompt trigger indicator. The two monitors use the same held-out executions. The weight-only screening study measures adapter-level detection over three held-out blocks per model size. The 0.6B run had eight qualification failures across 48 adapters. All 24 held-out adapters across the two model sizes qualified.

## Rerun the experiments on an NVIDIA GPU

From the downloaded repository, use a Python environment with CUDA-enabled PyTorch and a compatible NVIDIA driver. Install the training dependencies and check the device before starting a run.

```sh
python -m pip install '.[gpu]'
python gpu/run.py execution-monitoring --check
python gpu/run.py execution-monitoring
python gpu/run.py adapter-screening
```

Each study command trains LoRA adapters and generates model responses on CUDA. The original analysis also uses CPU operations. The launcher obtains pinned Qwen snapshots and writes checkpoints, records, and results under `gpu/experiments/trace_safety_v1/`. Run one study at a time. The screening study schedules 192 training jobs. These commands are for a source checkout. The default verifier reads the bundled evidence and takes seconds.

## Regenerate the synthetic prompts

```sh
python3 -m adapter_monitoring generate-data ./generated-data
```

The recipe includes seeds, generation settings, and the historical arithmetic expressions excluded from the 016 source pool. Generation verifies the original 011 partition hashes and both the 016 and 016b corpus hashes before writing output. It generates 480 training prompts for 011 and 5,760 for 016b, along with their original validation, qualification, or evaluation partitions as applicable.

## Evidence format

The evidence directory has a version 1 `manifest.json` and one JSON Lines file per run. The manifest lists the SHA-256 of each file, disjoint block assignments, label names, calibration rule, false-alarm target, and saved thresholds. Each row supplies a unique `id` within the run, an integer `block`, a `label`, and a map of monitor names to numeric `scores`. Rows must cover threshold and evaluation blocks. For adapter qualification, provide per-label minimums in the manifest and measured rates or scalar values under each row's `qualification`, with its recorded `passed` status. The bundled manifest provides examples.

The auditor reapplies the calibration rules, compares their thresholds with the saved thresholds, and counts scores at or above the saved thresholds on the held-out blocks. An optional `expected` object records the published `detected` and `false_alarms` numerator/denominator pairs for each monitor. Custom bundles can omit it. The CLI exits nonzero when validation fails. Checksums verify consistency with the manifest. Use the manifest from a trusted copy of the release when checking authenticity.

## Evidence provenance and full reruns

- **Execution Monitoring with Traces and Adapter Weights (run `011`).** The source study retained its execution records, adapter files and hashes, GLNet checkpoint, and summary. The fused execution probabilities were reconstructed on CPU using the saved checkpoint and refitted logistic heads. The split uses blocks 0–11 for weight fitting, 12–14 for fusion fitting, 15–17 for threshold calibration, and 18–23 for evaluation. Reconstruction used PyTorch 2.12.1 and scikit-learn 1.7.2. The audit uses reconstructed probabilities at the original saved thresholds. The evidence manifest records hashes for the source summary, config, scorer source and checkpoint.
- **Weight-Only Screening of LoRA Adapters (run `016b`).** The bundle uses saved GLNet weight scores and actor qualification measurements. Screening sets its threshold on benign adapters from blocks 6–8. Evaluation uses blocks 9–11. The manifest records the protocol, weight-score and source-file hashes, training seed, and exact model revisions. Qualification checks establish agreement with recorded actor decisions from the saved measurements.

The GPU launcher incorporates the research training and evaluation code. Full training reruns from this release remain unverified.
