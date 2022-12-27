import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DataDescriptor import DataDescriptor
from FeatureAnalyser import FeatureAnalyser

# Load the data from the CSV file
data_frame = pd.read_csv('data.csv', sep=';')

data_descriptor = DataDescriptor(data_frame)
data_descriptor.get_basic_description()
data_frame = data_descriptor.remove_duplicates()

#sns.catplot(x='job', y='age', hue='y', data=data_frame, kind="box", aspect=1.5)
#sns.catplot(x='job', y='y', hue='loan', data=data_frame, kind='bar')

feature_analyser = FeatureAnalyser(data_frame, savefig=True)
#for col1 in data_frame:
#    feature_analyser.pairwise_visualise(col1, 'y')
feature_analyser.visualise_num_cat_variables('pdays', 'y')

#feature_analyser.visualise_numerical_variable('emp.var.rate')
#feature_analyser.visualise_numerical_variable_against_target('job')
#for column in data_frame:
#    feature_analyser.visualise_variable(column)

# Plot histograms for each feature
#X.hist(figsize=(10, 10))
#plt.show()

# Plot scatter plots for each pair of features
#pd.plotting.scatter_matrix(X, figsize=(10, 10))
#plt.show()

# Plot box plots for each feature
#X.plot(kind='box', figsize=(10, 10))
#plt.show()

#correlation_tester = CorrelationTester(data_frame)
#correlation_tester.test_nominal_correlation('poutcome', 'y')

