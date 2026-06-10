# vision-mlops-pipeline

[![ci](https://github.com/vishalblz10/vision-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalblz10/vision-mlops-pipeline/actions/workflows/ci.yml)

An end-to-end MLOps pipeline for computer-vision models that covers both sides of the job:
the **models** (PyTorch training, confusion matrices, mAP, drift statistics) and the
**infrastructure** (evaluation gates in CI, an MLflow registry, a Prometheus-instrumented
model server, Kubernetes/KServe manifests with canary rollout, Terraform for EKS).

The reference model is a small CNN classifying synthetic shape images. That choice is
deliberate: the dataset is generated in-process and fully seeded, so **every stage of the
pipeline — training, gating, registry promotion, ONNX export, quantization, drift
detection, serving — runs end-to-end on a laptop CPU in under a minute**, and the test
suite proves each piece works. The pipeline itself is model-agnostic; the YOLO adapter
and detection-metrics module show the same machinery applied to object detection.

## Pipeline

```
 make_dataset ──▶ train ──▶ evaluate ──▶ gate ──▶ MLflow registry ──▶ export ──▶ serve
  (seeded,        (PyTorch   (confusion   (CI       (versions +        (ONNX +     (FastAPI +
   synthetic)      CNN)       matrix,      blocker)  @production       INT8        Prometheus,
                              F1, conf.)             alias)            quant)      K8s/KServe)
                                                                                      │
                drift monitor ◀── production traffic features (brightness, contrast) ◀┘
                (PSI + KS test)
```

## What maps to what

| Concern | Where |
|---|---|
| PyTorch model + training | `vision_mlops/model.py`, `vision_mlops/train.py` |
| Evaluation: confusion matrix, per-class P/R/F1 | `vision_mlops/evaluate.py` |
| Detection metrics: IoU, greedy matching, mAP@0.5 (YOLO-style eval) | `vision_mlops/detection.py` |
| YOLO adapter (ultralytics behind a stable interface) | `vision_mlops/yolo.py` |
| Drift monitoring: PSI + Kolmogorov–Smirnov | `vision_mlops/drift.py` |
| Evaluation gates that block promotion in CI | `vision_mlops/gates.py`, `config/gates.yaml` |
| MLflow tracking, registry, alias-based promotion | `vision_mlops/registry.py` |
| ONNX export + parity check, dynamic INT8 quantization | `vision_mlops/export.py` |
| Model server: /predict, /healthz, /readyz, /metrics | `vision_mlops/serving/api.py` |
| Kubernetes deployment (probes, HPA, scrape annotations) | `deploy/k8s/` |
| KServe InferenceService with 10% canary | `deploy/kserve/inferenceservice.yaml` |
| Prometheus alert rules + Grafana dashboard | `deploy/monitoring/` |
| EKS + VPC + artifact bucket (Terraform) | `deploy/terraform/` |
| CI + gated model-promotion workflow | `.github/workflows/` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest             # 56 tests across every module (CPU-only, no downloads)
vmops demo         # full lifecycle, printed stage by stage
```

`vmops demo` output (abridged, real run):

```
=== 1/6 Train the reference PyTorch model =======================
  epoch 1: loss 0.9662  train accuracy 0.5280
  epoch 4: loss 0.1989  train accuracy 0.9347

=== 2/6 Evaluate on the holdout window ==========================
  accuracy 0.9760  f1_macro 0.9756  mean confidence 0.9064
  confusion matrix: [[150, 2, 3], [4, 169, 0], [3, 0, 169]]

=== 3/6 Run the promotion gate ==================================
  [PASS] min_accuracy: accuracy 0.9760 (floor 0.9000)
  [PASS] min_f1_macro: f1_macro 0.9756 (floor 0.9000)
  [PASS] min_recall[circle]: recall 0.9677 (floor 0.8000)
  [PASS] min_recall[square]: recall 0.9769 (floor 0.8000)
  [PASS] min_recall[triangle]: recall 0.9826 (floor 0.8000)

=== 4/6 Log to MLflow and promote in the registry ===============
  run 5af8389d49ed40daafcb98d9aa9dfb55
  registered 'shapes-classifier' v1, alias '@production' updated

=== 5/6 Export for deployment (ONNX + INT8 quantization) ========
  models/model.pt + models/model.onnx (ONNX parity: max abs diff 2.38e-06)
  quantized: 610 KiB -> 226 KiB (62.9% smaller)

=== 6/6 Drift check: clean window vs shifted camera exposure ====
  [DRIFT] brightness: PSI 10.2340
  [DRIFT] contrast: PSI 0.5107
  drifted: True (a PSI alert would page the on-call here)
```

## The pieces

### Evaluation gates (`vmops gate`)

A model is promoted only if it clears absolute floors **and** doesn't regress against the
currently-serving baseline:

```bash
vmops train --out models/candidate.pt --metrics-out models/candidate-metrics.json
vmops gate --candidate models/candidate-metrics.json --baseline models/serving-metrics.json
echo $?   # non-zero exit blocks the CI job, and with it the promotion
```

Thresholds live in [`config/gates.yaml`](config/gates.yaml) (min accuracy/F1, per-class
recall floor, max accuracy drop vs baseline). The
[`model-promotion`](.github/workflows/model-promotion.yml) workflow wires this into GitHub
Actions: train → evaluate → gate → export ONNX, with the gate step failing the pipeline
exactly the way a production promotion should be stopped.

### MLflow registry

`vmops train --mlflow --register shapes-classifier --promote production` logs params and
metrics, registers a new version, and moves the `@production` alias. Serving code resolves
`models:/shapes-classifier@production`, so a promotion (or rollback) is just an alias move —
no image rebuild.

### Drift monitoring

`vision_mlops/drift.py` computes **PSI** (quantile-binned against the reference window)
and a **KS test** per feature. `vmops drift --shift 0.25` simulates a misconfigured camera
exposure; the same statistics run offline against logged production features. The serving
layer complements this with a low-confidence Prometheus alert
(`deploy/monitoring/prometheus-rules.yaml`) — a cheap online drift symptom that pages
before label-based metrics can.

### Serving

```bash
MODEL_PATH=models/model.pt uvicorn vision_mlops.serving.api:app --port 8000
curl -F "file=@shape.png" localhost:8000/predict
# {"label":"circle","confidence":0.98...,"probabilities":{...},"latency_ms":1.42,...}
```

- `/healthz` (liveness) vs `/readyz` (503 until the model loads) — wired to the probes in
  `deploy/k8s/deployment.yaml`, so a pod that can't predict never receives traffic.
- `/metrics` exports prediction counts by class, a latency histogram, and a **confidence
  histogram** whose drift toward low buckets drives the alerting above.

### Kubernetes, KServe canary, Terraform

`deploy/k8s/` runs the FastAPI image directly (Deployment + Service + HPA).
`deploy/kserve/inferenceservice.yaml` serves the ONNX export through KServe with
`canaryTrafficPercent: 10`: the previous revision keeps 90% of traffic while the candidate
takes 10%, watched via the same Prometheus metrics; promotion is removing the field,
rollback is setting it to 0. `deploy/terraform/` stands up the EKS cluster, VPC and the
versioned S3 bucket the `storageUri` points at.

These manifests are written to be applied to a real cluster, but this repo's CI doesn't
provision one — everything model-side is exercised by the test suite; the infra layer is
reviewable configuration.

### YOLO / detection

`vision_mlops/detection.py` implements IoU, greedy score-ordered matching (duplicate
detections count as false positives), PR curves and VOC2010-style mAP@0.5 — the evaluation
half of a detection pipeline, unit-tested against hand-computed cases.
`vision_mlops/yolo.py` adapts ultralytics YOLO models (`pip install ".[yolo]"`) to that
interface; the post-processing is dependency-free so it's tested without weights.

### Quantization / on-device

`vmops export-onnx --quantize models/model-int8.pt` produces the ONNX artifact KServe
serves plus a dynamic-INT8 PyTorch variant (62.9% smaller at equal predictions on the
holdout) — the first step of an on-device path where a Core ML / LiteRT conversion would
slot into the same export stage.

## Layout

```
vision_mlops/          # the package: data, model, train, evaluate, detection,
                       # drift, gates, registry, export, yolo, serving/, cli
config/gates.yaml      # promotion thresholds
tests/                 # 56 tests, all CPU, no downloads
deploy/                # k8s/, kserve/, monitoring/, terraform/
.github/workflows/     # ci.yml, model-promotion.yml
```

## License

MIT
