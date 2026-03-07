"""Iris flower classification with SVM.

A beginner-friendly multi-class classification model using
Support Vector Machine with RBF kernel on the classic Iris dataset.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train(ctx):
    hp = ctx.hyperparameters
    C = float(hp.get("C", 1.0))
    kernel = hp.get("kernel", "rbf")
    gamma = hp.get("gamma", "scale")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)),
    ])

    ctx.log_metric("progress", 10)

    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target

    ctx.log_metric("progress", 30)

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    for i, score in enumerate(scores):
        ctx.log_metric("accuracy", float(score), epoch=i + 1)
        ctx.log_metric("loss", float(1.0 - score), epoch=i + 1)

    ctx.log_metric("progress", 70)

    model.fit(X, y)
    train_acc = float(model.score(X, y))
    ctx.log_metric("accuracy", train_acc, epoch=len(scores) + 1)
    ctx.log_metric("progress", 100)


def infer(ctx):
    data = ctx.get_input_data()
    if "features" not in data:
        ctx.set_output({"error": "No 'features' key in input_data"})
        return

    X = np.array(data["features"])
    if X.ndim == 1:
        X = X.reshape(1, -1)

    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    iris = load_iris()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42)),
    ])
    model.fit(iris.data, iris.target)

    predictions = model.predict(X).tolist()
    probabilities = model.predict_proba(X).tolist()
    class_names = iris.target_names.tolist()

    ctx.set_output({
        "predictions": predictions,
        "class_names": [class_names[p] for p in predictions],
        "probabilities": probabilities,
    })
