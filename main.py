import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DataDescriptor import DataDescriptor
from FeatureAnalyser import FeatureAnalyser

data_frame = pd.read_csv('data.csv', sep=';')

data_descriptor = DataDescriptor(data_frame)
data_descriptor.get_basic_description()
data_frame = data_descriptor.remove_duplicates()

feature_analyser = FeatureAnalyser(data_frame, savefig=True)
#for column in data_frame:
#    feature_analyser.visualise_variable(column)
#    feature_analyser.visualise_num_cat_variables_fill(column, 'y')

''' Month is special, workaround because of a bug in the seaborn library https://github.com/mwaskom/seaborn/issues/2261 
row_order=['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df = data_frame[data_frame['month'] == 'mar']
for label in row_order:
    df2 = data_frame[data_frame['month'] == label]
    df = df.append(df2)
feature_analyser2 = FeatureAnalyser(df, savefig=True)
feature_analyser2.visualise_num_cat_variables_fill('month', 'y')'''

#feature_analyser.visualise_variable('month')
#feature_analyser.visualise_num_cat_variables_fill('month', 'y')

#feature_analyser.visualise_cat_pie_chart('y')
#feature_analyser.visualise_num_cat_variables('age', 'y', stat='proportion')


#feature_analyser.visualise_num_cat_variables_fill('duration', 'y')
#feature_analyser.visualise_num_cat_variables_fill('campaign', 'y')
#feature_analyser.visualise_num_cat_variables_fill('pdays', 'y')


#feature_analyser.visualise_numerical_variable('emp.var.rate')
#feature_analyser.visualise_numerical_variable_against_target('job')


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

