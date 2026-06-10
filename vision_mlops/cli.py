"""``vmops`` command-line interface for the pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _stage(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def cmd_train(args: argparse.Namespace) -> int:
    from vision_mlops.data import make_dataset
    from vision_mlops.evaluate import evaluate_model
    from vision_mlops.model import save_model
    from vision_mlops.train import train_model

    print(
        f"Generating {args.samples} training and {args.holdout} holdout images "
        f"(seed {args.seed})"
    )
    X_train, y_train = make_dataset(args.samples, seed=args.seed)
    X_holdout, y_holdout = make_dataset(args.holdout, seed=args.seed + 1)

    model, history = train_model(X_train, y_train, epochs=args.epochs, seed=args.seed)
    for epoch in history:
        print(
            f"  epoch {epoch['epoch']}: loss {epoch['loss']:.4f}  "
            f"train accuracy {epoch['accuracy']:.4f}"
        )

    metrics = evaluate_model(model, X_holdout, y_holdout)
    print(
        f"Holdout: accuracy {metrics['accuracy']:.4f}  f1_macro {metrics['f1_macro']:.4f}  "
        f"mean confidence {metrics['mean_confidence']:.4f}"
    )

    save_model(model, args.out)
    print(f"Model checkpoint -> {args.out}")
    if args.metrics_out:
        _write_json(args.metrics_out, metrics)
        print(f"Metrics -> {args.metrics_out}")

    if args.mlflow:
        from vision_mlops.registry import log_training_run, promote, register_model_version

        params = {"samples": args.samples, "epochs": args.epochs, "seed": args.seed}
        run_id = log_training_run(model, params, metrics, tracking_uri=args.tracking_uri)
        print(f"MLflow run logged: {run_id}")
        if args.register:
            version = register_model_version(run_id, args.register)
            print(f"Registered model '{args.register}' version {version}")
            if args.promote:
                promote(args.register, version, args.promote)
                print(f"Alias '@{args.promote}' now points at version {version}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from vision_mlops.data import make_dataset
    from vision_mlops.evaluate import evaluate_model
    from vision_mlops.model import load_model

    model = load_model(args.model)
    X, y = make_dataset(args.samples, seed=args.seed)
    metrics = evaluate_model(model, X, y)

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    if args.out:
        _write_json(args.out, metrics)
        print(f"Metrics -> {args.out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from vision_mlops.gates import GateConfig, evaluate_gate

    candidate = json.loads(Path(args.candidate).read_text())
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    config = GateConfig.from_yaml(args.config) if args.config else GateConfig()

    decision = evaluate_gate(candidate, config, baseline)
    for check in decision.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")
    if args.out:
        _write_json(args.out, decision.to_dict())

    if decision.passed:
        print("Gate PASSED - candidate may be promoted.")
        return 0
    print(f"Gate FAILED ({len(decision.failures)} failing check(s)) - promotion blocked.")
    return 1


def cmd_drift(args: argparse.Namespace) -> int:
    from vision_mlops.data import extract_features, make_dataset
    from vision_mlops.drift import detect_drift

    print(
        f"Simulating windows: reference vs current with brightness shift "
        f"{args.shift:+.2f} ({args.samples} images each)"
    )
    X_ref, _ = make_dataset(args.samples, seed=args.seed)
    X_cur, _ = make_dataset(args.samples, seed=args.seed + 1, brightness_shift=args.shift)

    report = detect_drift(
        extract_features(X_ref),
        extract_features(X_cur),
        psi_threshold=args.psi_threshold,
    )
    for feature in report.features:
        flag = "DRIFT" if feature.drifted else "ok"
        print(
            f"  [{flag:>5}] {feature.feature}: PSI {feature.psi:.4f}  "
            f"KS {feature.ks_statistic:.4f} (p={feature.ks_pvalue:.2e})"
        )
    if args.out:
        _write_json(args.out, report.to_dict())

    if report.drifted:
        print("Drift detected - investigate the upstream data before trusting predictions.")
        return 1
    print("No significant drift detected.")
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    from vision_mlops.export import (
        export_onnx,
        quantize_dynamic,
        serialized_size_bytes,
        verify_onnx,
    )
    from vision_mlops.model import load_model, save_model

    model = load_model(args.model)
    path = export_onnx(model, args.out)
    max_diff = verify_onnx(model, path)
    print(f"ONNX -> {path} (parity vs PyTorch: max abs diff {max_diff:.2e})")

    if args.quantize:
        quantized = quantize_dynamic(model)
        before = serialized_size_bytes(model)
        after = serialized_size_bytes(quantized)
        save_model(quantized, args.quantize)
        print(
            f"INT8 dynamic quantization -> {args.quantize} "
            f"({before / 1024:.0f} KiB -> {after / 1024:.0f} KiB, "
            f"{(1 - after / before) * 100:.1f}% smaller)"
        )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from vision_mlops.data import extract_features, make_dataset
    from vision_mlops.drift import detect_drift
    from vision_mlops.evaluate import evaluate_model
    from vision_mlops.export import (
        export_onnx,
        quantize_dynamic,
        serialized_size_bytes,
        verify_onnx,
    )
    from vision_mlops.gates import GateConfig, evaluate_gate
    from vision_mlops.model import save_model
    from vision_mlops.train import train_model

    _stage("1/6 Train the reference PyTorch model")
    X_train, y_train = make_dataset(1500, seed=args.seed)
    X_holdout, y_holdout = make_dataset(500, seed=args.seed + 1)
    model, history = train_model(X_train, y_train, epochs=4, seed=args.seed)
    for epoch in history:
        print(
            f"  epoch {epoch['epoch']}: loss {epoch['loss']:.4f}  "
            f"train accuracy {epoch['accuracy']:.4f}"
        )

    _stage("2/6 Evaluate on the holdout window")
    metrics = evaluate_model(model, X_holdout, y_holdout)
    print(
        f"  accuracy {metrics['accuracy']:.4f}  f1_macro {metrics['f1_macro']:.4f}  "
        f"mean confidence {metrics['mean_confidence']:.4f}"
    )
    print(f"  confusion matrix: {metrics['confusion_matrix']}")

    _stage("3/6 Run the promotion gate")
    config = GateConfig.from_yaml(args.config) if args.config else GateConfig()
    decision = evaluate_gate(metrics, config)
    for check in decision.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    if not decision.passed:
        print("Gate failed - stopping the demo here, exactly as CI would.")
        return 1

    _stage("4/6 Log to MLflow and promote in the registry")
    if args.skip_mlflow:
        print("  (skipped with --skip-mlflow)")
    else:
        from vision_mlops.registry import (
            DEFAULT_MODEL_NAME,
            log_training_run,
            promote,
            register_model_version,
        )

        run_id = log_training_run(
            model,
            {"samples": 1500, "epochs": 4, "seed": args.seed},
            metrics,
            tracking_uri=args.tracking_uri,
        )
        version = register_model_version(run_id, DEFAULT_MODEL_NAME)
        promote(DEFAULT_MODEL_NAME, version, "production")
        print(f"  run {run_id}")
        print(f"  registered '{DEFAULT_MODEL_NAME}' v{version}, alias '@production' updated")

    _stage("5/6 Export for deployment (ONNX + INT8 quantization)")
    save_model(model, "models/model.pt")
    onnx_path = export_onnx(model, "models/model.onnx")
    max_diff = verify_onnx(model, onnx_path)
    quantized = quantize_dynamic(model)
    before = serialized_size_bytes(model)
    after = serialized_size_bytes(quantized)
    print(f"  models/model.pt + {onnx_path} (ONNX parity: max abs diff {max_diff:.2e})")
    print(
        f"  quantized: {before / 1024:.0f} KiB -> {after / 1024:.0f} KiB "
        f"({(1 - after / before) * 100:.1f}% smaller)"
    )

    _stage("6/6 Drift check: clean window vs shifted camera exposure")
    X_clean, _ = make_dataset(500, seed=args.seed + 2)
    X_shifted, _ = make_dataset(500, seed=args.seed + 3, brightness_shift=0.25)
    report = detect_drift(extract_features(X_clean), extract_features(X_shifted))
    for feature in report.features:
        flag = "DRIFT" if feature.drifted else "ok"
        print(f"  [{flag:>5}] {feature.feature}: PSI {feature.psi:.4f}")
    print(f"  drifted: {report.drifted} (a PSI alert would page the on-call here)")

    print("\nDemo complete.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmops",
        description="MLOps pipeline for computer-vision models: train, evaluate, "
        "gate, register, export and monitor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the reference model and log/register it")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--holdout", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/model.pt")
    p.add_argument("--metrics-out", default="models/metrics.json")
    p.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--register", default=None, help="register the model under this name")
    p.add_argument("--promote", default=None, help="alias to point at the new version")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint on a fresh holdout")
    p.add_argument("--model", default="models/model.pt")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("gate", help="run promotion checks; non-zero exit blocks CI")
    p.add_argument("--candidate", required=True, help="candidate metrics JSON")
    p.add_argument("--baseline", default=None, help="serving baseline metrics JSON")
    p.add_argument("--config", default=None, help="gate thresholds YAML")
    p.add_argument("--out", default=None, help="write the decision JSON here")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("drift", help="simulate a drift check between two windows")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--shift", type=float, default=0.25, help="brightness shift of the current window")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--psi-threshold", type=float, default=0.25)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("export-onnx", help="export a checkpoint to ONNX (and optionally INT8)")
    p.add_argument("--model", default="models/model.pt")
    p.add_argument("--out", default="models/model.onnx")
    p.add_argument("--quantize", default=None, help="also write an INT8-quantized checkpoint here")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("demo", help="run the whole lifecycle end-to-end, printing each stage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="config/gates.yaml")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--skip-mlflow", action="store_true")
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
