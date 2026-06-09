import argparse
import sys
import pandas as pd
from pathlib import Path
from src.config import load_config
from src.prediction import load_artifacts, predict_new_data
from src.data_loader import load_data


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input data file (CSV or Excel)",
    )
    parser.add_argument(
        "--model",
        default="slp",
        choices=["slp", "logistic_regression", "random_forest", "xgboost"],
        help="Trained model to use (default: slp)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save predictions (default: print to stdout)",
    )
    args = parser.parse_args()

    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])

    model_path = models_dir / f"{args.model}_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    lang_mapping_path = models_dir / "lang_mapping.pkl"

    for p in [model_path, scaler_path, lang_mapping_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run the pipeline or train first.")
            sys.exit(1)

    print(f"Loading artifacts from {models_dir}...")
    model, scaler, lang_mapping = load_artifacts(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        lang_mapping_path=str(lang_mapping_path),
    )

    print(f"Loading input data from {args.input}...")
    df_input = load_data(args.input)

    print("Making predictions...")
    result_df = predict_new_data(df_input, model, scaler, lang_mapping)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".csv":
            result_df.to_csv(output_path, index=False)
        else:
            result_df.to_excel(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print("\nPredictions:")
        print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
