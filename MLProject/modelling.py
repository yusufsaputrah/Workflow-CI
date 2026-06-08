import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import argparse

def train(n_estimators):
    df = pd.read_csv('dataset_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="MLProject_RandomForest"):
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics dan params secara eksplisit
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        # Log model secara eksplisit - PENTING agar artifact 'model' selalu ada
        mlflow.sklearn.log_model(clf, "model")

        print(f"Accuracy with {n_estimators} estimators: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    train(args.n_estimators)
