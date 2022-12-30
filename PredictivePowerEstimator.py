from sklearn.feature_selection import mutual_info_regression


class PredictivePowerEstimator:
    def __init__(self, df, target_name):
        self.X = df.drop(columns=[target_name])
        self.y = df[target_name]

    def get_predictive_power(self):
        mi = mutual_info_regression(self.X, self.y, random_state=0)
        feature_importances = zip(self.X.columns, mi)
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        return feature_importances

    def print_predictive_power(self):
        feature_importances = self.get_predictive_power()
        print("Feature | Predictive Power")
        print("--------|----------------")
        for feature, importance in feature_importances:
            print(f"{feature:10} | {importance:.3f}")