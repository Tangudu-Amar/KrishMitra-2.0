import numpy as np

class RFXGEnsemble:
    def __init__(self, rf_model, xgb_model):
        self.rf_model = rf_model
        self.xgb_model = xgb_model

    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        avg_proba = [(rf_p[:, 1] + xgb_p[:, 1]) / 2 for rf_p, xgb_p in zip(rf_proba, xgb_proba)]
        return np.vstack(avg_proba)

    def predict(self, X):
        avg_proba_stacked = self.predict_proba(X)
        predictions = (avg_proba_stacked > 0.5).astype(int)
        return predictions.T
