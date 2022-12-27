import pandas as pd

class DataDescriptor:
    def __init__(self, data):
        self.data = data

    def remove_duplicates(self):
        return self.data.drop_duplicates(self.data)

    def get_basic_description(self):
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        n_samples, n_features = X.shape

        print("Number of samples: ", n_samples)
        print("Number of features: ", n_features)
        print("Number of classes: ", len(y.unique()), y.unique())
        duplicated_values = self.data.duplicated().sum()
        print("Duplicate values: ", duplicated_values)

        if (self.data.duplicated().sum() > 0):
            self.data = self.data.drop_duplicates(self.data)
            print(duplicated_values, " duplicated rows have been dropped!")

        duplicated_values_X = self.data.iloc[:, :-1].duplicated().sum()
        print("Duplicate values in X: ", duplicated_values_X)
        print("Numerical features:")
        self.describe_numerical_data()
        print("Categorical features:")
        self.describe_categorical_data()

    def describe_numerical_data(self):
        print("{:<9}".format("Name"),
              "{:>6}".format("UVal"),
              "{:>8}".format("DType"),
              "{:>8}".format("Count"),
              "{:>8}".format("Mean"),
              "{:>8}".format("Std"),
              "{:>8}".format("Min"),
              "{:>8}".format("25%"),
              "{:>8}".format("50%"),
              "{:>8}".format("75%"),
              "{:>8}".format("Max"),
              "{:>8}".format("Skew"))
        for feature in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                self.print_numerical_feature(feature)

    def print_numerical_feature(self, numerical_feature):
        col = self.data[numerical_feature]
        name = col.name
        no_unique_values = col.nunique()
        data_type = col.dtype
        stats = col.describe()
        count = stats['count']
        mean = stats['mean']
        std = stats['std']
        min_val = stats['min']
        p25_val = stats['25%']
        p50_val = stats['50%']
        p75_val = stats['75%']
        max_val = stats['max']
        skewness = col.skew()
        print("{:<9}".format(name[0:9]),
              "{:>6}".format(no_unique_values),
              "{:>8}".format(str(data_type)),
              "{:>8}".format(str(int(count))),
              "{:>8.2f}".format(mean),
              "{:>8.2f}".format(std),
              "{:>8.2f}".format(min_val),
              "{:>8.2f}".format(p25_val),
              "{:>8.2f}".format(p50_val),
              "{:>8.2f}".format(p75_val),
              "{:>8.2f}".format(max_val),
              "{:>8.2f}".format(skewness))

    def describe_categorical_data(self, ):
        print("{:<9}".format("Name"),
              "{:>6}".format("UVal"),
              "{:>8}".format("DType"),
              "{:>8}".format("Count"),
              "{:>16}".format("Unique Values"))
        for feature in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[feature]):
                self.print_categorical_feature(feature)

    def print_categorical_feature(self, categorical_feature):
        col = self.data[categorical_feature]
        name = col.name
        no_unique_values = col.nunique()
        data_type = col.dtype
        stats = col.describe()
        count = stats['count']
        unique_values = col.unique()
        print("{:<9}".format(name[0:9]),
              "{:>6}".format(no_unique_values),
              "{:>8}".format(str(data_type)),
              "{:>8}".format(str(int(count))),
              "{:<16}".format(str(unique_values)[0:65]))
