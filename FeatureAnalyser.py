import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt

class FeatureAnalyser:
    def __init__(self, data, savefig=False):
        self.data = data
        self.savefig = savefig
        sns.set_style('darkgrid')
        sns.set_palette('Set2')

    def visualise_numerical_variable(self, numerical_variable, kde=False):
        plt.title(label=numerical_variable)
        col = self.data[numerical_variable]
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.5, 2)})
        mean = np.array(col).mean()

        sns.boxplot(data=col, orient='h', width=0.6, ax=ax_box)
        ax_box.axvline(mean, color='r', linestyle='--')

        binwidth = None
        if pd.api.types.is_integer_dtype(self.data[numerical_variable]):
            binwidth = 1
        sns.histplot(data=col, kde=kde, stat='proportion', binwidth=binwidth, ax=ax_hist)
        ax_hist.axvline(mean, color='r', linestyle='--')

        plt.legend({'Mean': mean})
        ax_box.set(xlabel='')
        self.display_plot(numerical_variable)

    def display_plot(self, plot_name=None):
        if self.savefig:
            plt.savefig(plot_name + '.png')
        else:
            plt.show()

    def visualise_categorical_variable(self, categorical_variable, order=None):
        sns.countplot(x=categorical_variable, data=self.data, order=order)
        self.set_x_labels()
        self.display_plot(categorical_variable)

    def set_x_labels(self):
        tick_labels = [item.get_text() for item in plt.xticks()[1]]
        if (len(tick_labels) > 5):
            new_tick_labels = [label[:4] for label in tick_labels]
            plt.xticks(range(len(tick_labels)), new_tick_labels)

    def visualise_variable(self, variable):
        plt.clf()
        if pd.api.types.is_numeric_dtype(self.data[variable]):
            self.visualise_numerical_variable(variable)
        elif pd.api.types.is_object_dtype(self.data[variable]):
            self.visualise_categorical_variable(variable)
        else:
            print(f'Error: {variable} is not a supported variable.')

    def pairwise_visualise(self, first_var, second_var):
        plt.clf()
        if pd.api.types.is_numeric_dtype(self.data[first_var]) and pd.api.types.is_numeric_dtype(self.data[second_var]):
            sns.scatterplot(x=first_var, y=second_var, data=self.data)
        elif pd.api.types.is_numeric_dtype(self.data[first_var]) and pd.api.types.is_object_dtype(self.data[second_var]):
            sns.catplot(x=second_var, y=first_var, data=self.data, kind="box", aspect=1.5)
            self.set_x_labels()
        elif pd.api.types.is_object_dtype(self.data[first_var])  and pd.api.types.is_numeric_dtype(self.data[second_var]):
            sns.catplot(x=first_var, y=second_var, data=self.data, kind="box")
            self.set_x_labels()
        elif pd.api.types.is_object_dtype(self.data[first_var])  and pd.api.types.is_object_dtype(self.data[second_var]):
            sns.histplot(x=first_var, hue=second_var, data=self.data, stat="percent", multiple='fill')
            self.set_x_labels()
        else:
            print('Not a supported variables')
        self.display_plot(first_var + '_' + second_var + '.png')

    def visualise_num_cat_variables(self, num_var, cat_var):
        f, (ax_box, ax_hist) = plt.subplots(1, 2, sharey=True)
        sns.boxplot(x=cat_var, y=num_var, data=self.data, ax=ax_box)
        sns.histplot(x=cat_var, y=num_var, data=self.data, stat="proportion", multiple='layer', ax=ax_hist)
        self.set_x_labels()
        self.display_plot(num_var + '_' + cat_var + '.png')
        return
