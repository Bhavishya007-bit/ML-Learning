"""Run with python -m claims --help from the project directory."""
import argparse
import json
from pathlib import Path

import joblib

from .data import ingest, load_batch, read_csv, synthetic
from .model import predict, segment, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run clearly-labelled synthetic data")
    demo.add_argument("--rows", type=int, default=5000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", default="outputs/demo")
    fit = commands.add_parser("train", help="Ingest and train on your own Allstate-format CSV")
    fit.add_argument("--csv", required=True)
    fit.add_argument("--source", required=True, help="Honest dataset name/provenance")
    fit.add_argument("--output", default="outputs/allstate")
    fit.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="Predict severity and bands for new rows")
    infer.add_argument("--model", required=True)
    infer.add_argument("--csv", required=True)
    infer.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    if args.command == "predict":
        # Only load artifacts you created/trust: joblib is a pickle format.
        bundle = joblib.load(args.model)
        frame, _ = read_csv(args.csv, training=False)
        values = predict(bundle["model"], frame[bundle["features"]])
        result = frame[["id"]].assign(predicted_loss=values,
            severity_band=segment(values, bundle["thresholds"]), model_source=bundle["source"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} predictions to {args.output}")
        return
    output = Path(args.output)
    # Avoid accidentally overwriting an earlier experiment or its evidence.
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "demo":
        csv_path = output / "synthetic.csv"
        synthetic(args.rows, args.seed).to_csv(csv_path, index=False)
        source = "synthetic_demo"
    else:
        csv_path, source = Path(args.csv), args.source
    db = output / "claims.sqlite"
    batch, quality = ingest(csv_path, db, source)
    (output / "quality.json").write_text(json.dumps(quality, indent=2))
    report = train(load_batch(db, batch), output, source, args.seed)
    print((output / "evaluation.md").read_text())


if __name__ == "__main__":
    main()
