import pandas as pd
import numpy as np

from DataDescriptor import DataDescriptor
from DataTransformer import DataTransformer
from FeatureAnalyser import FeatureAnalyser
from ModelEvaluator import ModelEvaluator
from PredictivePowerEstimator import PredictivePowerEstimator


data_frame = pd.read_csv('data.csv', sep=';')

data_descriptor = DataDescriptor(data_frame)
data_descriptor.get_basic_description()
data_frame = data_descriptor.remove_duplicates()

''' pdays is special, we need to move 999 values to another column'''
data_frame['pre_camp_contacted'] = np.where(data_frame['pdays'] == 999, 0, 1)
new_pdays_placeholder = -1
data_frame['pdays'] = np.where(data_frame['pdays'] == 999, new_pdays_placeholder, data_frame['pdays'])
column_names = ['duration']
data_frame.drop(columns=column_names, inplace=True)

feature_analyser = FeatureAnalyser(data_frame, savefig=True)
for column in data_frame:
    feature_analyser.visualise_variable(column)
    feature_analyser.visualise_num_cat_variables_fill(column, 'y')
feature_analyser.corr_heatmap(method='pearson')

''' For plotting pdays (only consider pdays > 0)
new_df = data_frame.drop(data_frame[data_frame['pdays'] < 0].index, inplace=False)
data_descriptor = DataDescriptor(new_df)
data_descriptor.get_basic_description()
feature_analyser2 = FeatureAnalyser(new_df, savefig=True)
feature_analyser2.visualise_numerical_variable('pdays')'''

data_transformer = DataTransformer()
df = data_transformer.transform_variable(data_frame, 'y', {'no': 0, 'yes': 1})
df = data_transformer.one_hot_encode_variable(df, ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome'])
df = data_transformer.transform_variable(df, 'education', {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 6})
df = data_transformer.transform_variable(df, 'day_of_week', {'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6})
df = data_transformer.transform_variable(df, 'month', {'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
df = data_transformer.scale_variables(df, method='minmax')

predictive_power_estimator = PredictivePowerEstimator(df, 'y')
predictive_power_estimator.print_predictive_power()

evaluator = ModelEvaluator(df, 'y')
evaluator.evaluate()
evaluator.print_results()

columns_list = ['cons.price.idx', 'cons.conf.idx', 'pdays', 'pre_camp_contacted', 'month', 'poutcome_success', 'poutcome_nonexistent', 'poutcome_failure', 'y']
new_df = df[columns_list].copy()
evaluator = ModelEvaluator(new_df, 'y')
evaluator.evaluate()
evaluator.print_results()
