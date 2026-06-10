"""Evaluation gates: decide whether a candidate model may be promoted.

Used as the promotion check in CI/CD — the CLI's ``gate`` command exits
non-zero when any check fails, which blocks the pipeline. Two kinds of checks
run: absolute quality floors, and (when a production baseline is supplied)
non-regression against the model currently serving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GateConfig:
    min_accuracy: float = 0.90
    min_f1_macro: float = 0.90
    min_per_class_recall: float = 0.80
    max_accuracy_drop: float = 0.02

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GateConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    checks: tuple[Check, ...]

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


def evaluate_gate(
    candidate: dict, config: GateConfig, baseline: dict | None = None
) -> GateDecision:
    checks: list[Check] = []

    accuracy = candidate["accuracy"]
    checks.append(
        Check(
            "min_accuracy",
            accuracy >= config.min_accuracy,
            f"accuracy {accuracy:.4f} (floor {config.min_accuracy:.4f})",
        )
    )

    f1 = candidate["f1_macro"]
    checks.append(
        Check(
            "min_f1_macro",
            f1 >= config.min_f1_macro,
            f"f1_macro {f1:.4f} (floor {config.min_f1_macro:.4f})",
        )
    )

    for cls, metrics in candidate.get("per_class", {}).items():
        recall = metrics["recall"]
        checks.append(
            Check(
                f"min_recall[{cls}]",
                recall >= config.min_per_class_recall,
                f"recall {recall:.4f} (floor {config.min_per_class_recall:.4f})",
            )
        )

    if baseline is not None:
        drop = baseline["accuracy"] - accuracy
        checks.append(
            Check(
                "max_accuracy_drop",
                drop <= config.max_accuracy_drop,
                f"accuracy {accuracy:.4f} vs baseline {baseline['accuracy']:.4f} "
                f"(drop {drop:+.4f}, tolerance {config.max_accuracy_drop:.4f})",
            )
        )

    return GateDecision(passed=all(c.passed for c in checks), checks=tuple(checks))
