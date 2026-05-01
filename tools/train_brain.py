#!/usr/bin/env python3
"""
Train a minimal text classifier (intent routing) and export Core ML + vocabulary + response map.

Sklearn CountVectorizer is not supported by coremltools, so we export only the trained
LogisticRegression (float features). Token→bag-of-words matching sklearn is done on-device
in BrainEngine using brain_vocab.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Tiny synthetic dataset: enough to learn token hints for "code" vs chat.
DEFAULT_EXAMPLES: list[tuple[str, str]] = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("thanks", "greeting"),
    ("bye", "greeting"),
    ("what is a loop", "explain"),
    ("explain recursion", "explain"),
    ("what does async mean", "explain"),
    ("define a variable", "explain"),
    ("how does memory work", "explain"),
    ("write hello world in python", "code"),
    ("python function to add two numbers", "code"),
    ("show me a for loop in swift", "code"),
    ("implement binary search", "code"),
    ("snippet for reading a file", "code"),
    ("fibonacci code", "code"),
    ("def foo():", "code"),
    ("import numpy", "code"),
]


DEFAULT_RESPONSES: dict[str, str] = {
    "greeting": "Hi! I’m a tiny on-device brain (logistic regression over word counts). "
    "Ask for code or explanations.",
    "explain": "Short take: break the idea into steps, try a minimal example, then extend. "
    "If you want a concrete snippet, ask “write code for …”.",
    "code": "Minimal Python hello world:\n\nprint(\"hello, world\")\n\n"
    "Ask something more specific and I’ll still route it as the code intent.",
}


def load_extra_examples(repo_root: Path) -> list[tuple[str, str]]:
    path = repo_root / "brain" / "extra_training.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for row in data.get("examples", []):
        text = row.get("text", "").strip()
        label = row.get("intent", "").strip()
        if text and label:
            out.append((text, label))
    return out


def load_response_overrides(repo_root: Path) -> dict[str, str]:
    path = repo_root / "brain" / "intent_responses.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("responses", {})
    return {str(k): str(v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for Brain.mlpackage, brain_vocab.json, brain_responses.json",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    texts, labels = zip(*(DEFAULT_EXAMPLES + load_extra_examples(repo_root)))
    responses = {**DEFAULT_RESPONSES, **load_response_overrides(repo_root)}

    vec = CountVectorizer(max_features=256, ngram_range=(1, 2))
    X = vec.fit_transform(list(texts))
    clf = LogisticRegression(max_iter=500, random_state=0)
    clf.fit(X, list(labels))

    feature_names = vec.get_feature_names_out()
    tokens_by_col = feature_names.tolist()
    n_features = len(tokens_by_col)
    if int(clf.coef_.shape[1]) != n_features:
        raise RuntimeError("Classifier feature count does not match vectorizer output")

    input_tensor = ct.TensorType(shape=(n_features,), dtype=np.float32)
    mlmodel = ct.converters.sklearn.convert(
        clf,
        input_features=[("features", input_tensor)],
        output_feature_names="intent",
    )
    mlmodel.author = "AIApp train_brain.py"
    mlmodel.short_description = "Intent classifier (logistic regression on bag-of-words)"

    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name

    mlpackage_path = out_dir / "Brain.mlpackage"
    if mlpackage_path.exists():
        import shutil

        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))

    vocab_payload = {
        "version": 1,
        "feature_count": n_features,
        "classes": [str(c) for c in clf.classes_.tolist()],
        "tokens": tokens_by_col,
        "ngram_range": list(vec.ngram_range),
        "lowercase": bool(vec.lowercase),
        "binary": bool(vec.binary),
        "token_pattern": vec.token_pattern,
        "model_input_name": input_name,
    }
    (out_dir / "brain_vocab.json").write_text(json.dumps(vocab_payload, indent=2), encoding="utf-8")

    if sys.platform == "darwin":
        import subprocess

        subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                str(mlpackage_path),
                str(out_dir),
            ],
            check=True,
        )
        print(f"Compiled {out_dir / 'Brain.mlmodelc'}")
    else:
        print("Note: Brain.mlmodelc not compiled (needs macOS + xcrun coremlcompiler).")

    (out_dir / "brain_responses.json").write_text(
        json.dumps({"intents": responses, "labels": sorted(set(labels))}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {mlpackage_path}")
    print(f"Wrote {out_dir / 'brain_vocab.json'}")
    print(f"Wrote {out_dir / 'brain_responses.json'}")


if __name__ == "__main__":
    main()
