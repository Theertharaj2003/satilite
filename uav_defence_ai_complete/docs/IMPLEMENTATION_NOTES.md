# Implementation notes

## RT-DETR
Ultralytics currently exposes RT-DETR through a Python API for prediction and training. The baseline scripts use that supported interface.

## Tracking
For an initial runnable pipeline, `ByteTrackAdapter` exposes a stable internal track interface. The current code keeps a deterministic IoU fallback so development can proceed even if the external tracker stack changes. **For final thesis numbers, configure and record a canonical ByteTrack implementation/version; never report fallback IoU-tracker results as ByteTrack results.**

## Re-ID
The OSNet adapter is optional because OSNet packaging/checkpoint conventions differ across environments. The pipeline has a deterministic appearance-embedding fallback for development only. Final Re-ID experiments should use a fixed OSNet checkpoint and record preprocessing/checkpoint details.

## Fusion
The cross-modal adapter is trained with a reconstruction-oriented objective so the project can be executed and debugged before an end-to-end detector-aware loss is added. Final research experiments should compare simple fusion against the trained adapter and, later, a detector-aware joint objective if compute permits.

## Edge
TensorRT installation is environment-specific. Run the ONNX export first. Then build the TensorRT engine with the exact CUDA/TensorRT version installed on the target machine, and record that environment with the benchmark.

## Assistant
The assistant is deliberately read-only. It operates on structured perception state and approved documents. A deterministic rule-based provider is included so the application runs without an LLM server. Ollama integration is optional.
