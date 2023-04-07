import pickle
import pandas as pd


class Model:

    def __init__(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def predict(self, param1, param2) -> dict:
        x_test = pd.DataFrame([[param1, param2]])
        y = self.model.predict(x_test)[0]  # just get single value
        prob = str(self.model.predict_proba(x_test)[0].tolist())  # send to list for return
        return {'prediction': y.item(), "confidence": prob}
