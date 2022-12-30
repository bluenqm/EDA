import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataTransformer:
    def __init__(self):
        return

    def transform_variable(self, df, var, mapping_dict):
        df_mapped = df.copy()
        df_mapped[var] = df_mapped[var].map(mapping_dict)
        return df_mapped

    def scale_variable(self, df, var, method='standard', feature_range=(0, 1)):
        df_scaled = df.copy()
        column = df_scaled[[var]]

        scaler = StandardScaler()
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)

        column_scaled = scaler.fit_transform(column)
        df_scaled[var] = column_scaled
        return df_scaled

    def scale_variables(self, df, method='standard', feature_range=(0, 1)):
        scaler = StandardScaler()
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    def one_hot_encode_variable(self, df, var):
        df_encoded = pd.get_dummies(data=df, columns=var)
        return df_encoded
