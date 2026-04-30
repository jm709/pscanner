"""ML training pipeline for the copy-trade gate model.

Consumes ``training_examples`` from ``data/corpus.sqlite3`` and produces
a versioned XGBoost model artifact. Inference is out of scope for v1.
"""
