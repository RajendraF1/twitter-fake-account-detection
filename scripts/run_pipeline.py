import argparse
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the full detection pipeline"
    )
    parser.add_argument(
        "--model",
        default="slp",
        choices=["slp", "logistic_regression", "random_forest", "xgboost"],
        help="Model to train (default: slp)",
    )
    parser.add_argument(
        "--train-path",
        default=None,
        help="Override training data path",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Override testing data path",
    )
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.run(
        train_path=args.train_path,
        test_path=args.test_path,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
