import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def evaluate(name, clf, X_tr, X_te, y_tr, y_te):
    """
    Fits a classifier, computes predictions on test set, prints classification report and accuracy.
    """
    clf.fit(np.array(X_tr, float), np.array(y_tr))
    preds = clf.predict(np.array(X_te, float))
    acc   = accuracy_score(y_te, preds)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(classification_report(y_te, preds))
    print(f"  accuracy: {acc:.4f}")
    return acc, clf
