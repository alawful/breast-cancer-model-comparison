# Breast cancer model comparison

This is a compact machine-learning exercise using scikit-learn's Wisconsin breast-cancer dataset. It compares several familiar classifiers under the same train/test split and reports accuracy, weighted F1, a confusion matrix, and five-fold cross-validation accuracy.

The models are logistic regression, an RBF support-vector machine, k-nearest neighbors, random forest, decision tree, and Gaussian naive Bayes. Scaling is applied to the models that depend strongly on feature magnitude. During cross-validation, preprocessing is fitted separately inside each fold to avoid validation leakage.

## Running

```bash
python -m pip install -r requirements.txt
python src/evaluate_models.py
```

The current script prints a concise summary to the console. It uses the dataset bundled with scikit-learn, so no patient records or external medical data are included.

## Scope

This is an educational model comparison, not a diagnostic system. Its results should not be interpreted as clinical performance or medical advice.
