# UAV Defence AI Perception Platform

A modular research prototype for the frozen M.Tech scope:

**EO/RGB + IR → registration → cross-modal attention → BiFPN-style multi-scale fusion → RT-DETR → ByteTrack → OSNet Re-ID → temporal context → confidence/human verification → AI assistant/RAG → knowledge distillation → ONNX/TensorRT edge benchmarking.**

This repository is deliberately modular. Each phase can be run and tested independently before the next phase is enabled.

## Important scope

This is a perception and operator-support research prototype. It does **not** implement weapon control, autonomous engagement, or fire-control functions.

## Recommended environment

Use **Python 3.11** on Windows/Linux. CUDA-enabled PyTorch should be installed from the official PyTorch selector for the user's GPU/OS before the remaining packages when GPU training is required.

### Windows quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional edge stack:

```powershell
pip install -r requirements-edge.txt
```

Optional assistant/RAG stack:

```powershell
pip install -r requirements-assistant.txt
```

## Phase checklist

| Phase | Status | Entry point |
|---|---|---|
| 1. Environment + dataset | Ready | `scripts/check_environment.py` |
| 2. RT-DETR baseline | Ready | `scripts/train_detector.py`, `scripts/infer_image.py` |
| 3. EO/IR registration | Ready | `scripts/register_pair.py` |
| 4. Cross-modal fusion | Ready | `scripts/train_fusion.py` / `scripts/fuse_pair.py` |
| 5. BiFPN-style fusion adapter | Ready | `src/fusion/bifpn.py` |
| 6. RT-DETR inference on fused input | Ready | `scripts/infer_fused.py` |
| 7. ByteTrack | Ready | `scripts/run_video.py` |
| 8. OSNet Re-ID adapter | Ready/optional dependency | `src/reid/` |
| 9. Temporal context | Ready | `src/context/` |
| 10. Confidence + human review | Ready | `src/confidence/` |
| 11. State/Event store | Ready | `src/state/` |
| 12. AI assistant + RAG | Ready with deterministic fallback | `scripts/run_assistant.py` |
| 13. Dashboard | Ready | `streamlit run app/dashboard.py` |
| 14. Knowledge distillation prototype | Ready | `scripts/train_distillation.py` |
| 15. ONNX/TensorRT export/benchmark | Ready where backend installed | `scripts/export_onnx.py`, `scripts/benchmark.py` |

## First smoke test without a real dataset

The repository includes a synthetic EO/IR pair generator. This validates registration, fusion, state storage, context, confidence and assistant components without downloading data.

```powershell
python tools/generate_demo_data.py
python scripts/register_pair.py --eo data/raw/eo/demo_eo.png --ir data/raw/ir/demo_ir.png
python scripts/fuse_pair.py --eo data/raw/eo/demo_eo.png --ir data/raw/ir/demo_ir.png
python scripts/run_assistant.py --question "Give me the current scene summary"
```

For detector inference/training, install `ultralytics` through `requirements.txt` and provide either pretrained RT-DETR weights or your dataset YAML.

## Dataset layout for Ultralytics detection training

```text
data/
  processed/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
  annotations/
  splits/
```

An example dataset YAML is provided under `config/dataset.yaml`.

## Actual research workflow

1. Establish RGB-only RT-DETR baseline.
2. Establish IR-only baseline where labels are available.
3. Establish non-learned/simple EO+IR fusion baseline.
4. Train/evaluate the cross-modal fusion adapter.
5. Compare detection metrics.
6. Run video tracking and calculate IDF1/HOTA through the chosen evaluation tooling.
7. Add Re-ID and measure identity recovery after occlusion.
8. Record trajectories and derive contextual events.
9. Route low-confidence cases to human review.
10. Add the assistant over the structured state store and approved documents.
11. Distill a smaller model.
12. Export and benchmark FP32/FP16/INT8 where the target GPU/TensorRT stack is available.

## Current implementation note

The research code separates the multimodal fusion adapter from the RT-DETR detector because Ultralytics exposes RT-DETR training/inference as a detector API rather than a turnkey custom cross-modal architecture. The adapter therefore produces a fused 3-channel image that is passed to RT-DETR. This keeps the pipeline runnable and makes the fusion module independently trainable. A later experiment can replace the adapter with an end-to-end feature-level RT-DETR fork without changing the downstream tracking/context/assistant interfaces.

## Evidence and reproducibility

Every experiment should save:

- configuration
- model/checkpoint path
- dataset split identifier
- metrics JSON/CSV
- predictions
- timing measurements
- environment information

Do not report target metrics until they are actually measured on your chosen dataset split.
