"""
EXPERIMENT 7 - Naive Bayes and Memorization Methods for Classification
AIM: To compare Naive Bayes with memorization methods (k-Nearest Neighbours
     and Decision Tree) and select the best model.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------- 1. Load data and stratified split ----------
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("Wine Quality Classification: Naive Bayes vs Memorization Methods")
print("=" * 65)
print(
    "Samples:",
    X.shape[0],
    "| Features:",
    X.shape[1],
    "| Classes:",
    list(wine.target_names),
)
print("Class counts:", np.bincount(y).tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------- 2. Models ----------
models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "k-Nearest Neighbours (k=5)": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree (depth=3)": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
}

# ---------- 3. Train / test / 5-fold CV comparison ----------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X, y, cv=kfold)
    summary.append([name, train_acc, test_acc, cv.mean(), cv.std()])

table = pd.DataFrame(
    summary, columns=["Model", "Train Acc", "Test Acc", "CV Mean", "CV Std"]
)
print("\nModel comparison:")
print(table.round(4).to_string(index=False))

best = table.sort_values(["CV Mean", "Test Acc"], ascending=False).iloc[0][
    "Model"
]
print("\nSelected model (highest CV accuracy):", best)

# ---------- 4. Confusion matrix and classification report ----------
pred = models[best].predict(X_test)
print("\nConfusion matrix of selected model:")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=wine.target_names))

# ---------- 5a. Effect of k in kNN (memorization) ----------
print("Effect of k on kNN (5-fold CV accuracy):")
for k in [1, 3, 5, 7, 11, 15]:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    print(f"  k = {k:>2} -> {cross_val_score(knn, X, y, cv=kfold).mean():.4f}")

# ---------- 5b. Rules learnt by the decision tree ----------
print("\nDecision tree rules:")
print(
    export_text(models["Decision Tree (depth=3)"], feature_names=list(X.columns))
)
