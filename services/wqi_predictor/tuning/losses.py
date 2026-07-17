import numpy as np

def log_cosh_obj(labels: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # custom log-cosh objective for xgb and lightgbm
    x = preds - labels
    grad = np.tanh(x)
    hess = 1.0 - np.tanh(x)**2

    return grad, hess
