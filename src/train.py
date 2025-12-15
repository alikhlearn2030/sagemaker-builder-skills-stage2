train_script = r"""
import os
import argparse
import glob
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def first_csv(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {path}")
    return files[0]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-x-dir", type=str, default="/opt/ml/input/data/train_x")
    p.add_argument("--train-y-dir", type=str, default="/opt/ml/input/data/train_y")
    p.add_argument("--test-x-dir", type=str, default="/opt/ml/input/data/test_x")
    p.add_argument("--test-y-dir", type=str, default="/opt/ml/input/data/test_y")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    x_train_path = first_csv(args.train_x_dir)
    y_train_path = first_csv(args.train_y_dir)
    x_test_path  = first_csv(args.test_x_dir)
    y_test_path  = first_csv(args.test_y_dir)

    print("Reading files:")
    print("X_train:", x_train_path)
    print("y_train:", y_train_path)
    print("X_test :", x_test_path)
    print("y_test :", y_test_path)

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test  = pd.read_csv(x_test_path)
    y_test  = pd.read_csv(y_test_path).squeeze()

    model = LogisticRegression(C=args.C, max_iter=args.max_iter)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("TEST_ACCURACY:", acc)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
"""
with open("train.py", "w", encoding="utf-8") as f:
    f.write(train_script)

print("Updated train.py (robust file loading).")
