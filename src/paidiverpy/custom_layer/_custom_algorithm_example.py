import ast
from paidiverpy.custom_layer.base_custom_algorithm import BaseCustomAlgorithm
from sklearn import preprocessing

class MyMethod(BaseCustomAlgorithm):
    def process(self):
        feature_range = ast.literal_eval(self.params.feature_range)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        self.image_data = min_max_scaler.fit_transform(self.image_data)
        return self.image_data
