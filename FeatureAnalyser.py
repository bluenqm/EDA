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
        self.display_plot(numerical_variable)

    def display_plot(self, plot_name=None):
        if self.savefig:
            plt.savefig(plot_name + '.png')
        else:
            plt.show()

    def visualise_categorical_variable(self, categorical_variable, order=None):
        sns.countplot(x=categorical_variable, data=self.data)
        self.set_x_labels()
        self.display_plot(categorical_variable)

    def set_x_labels(self):
        tick_labels = [item.get_text() for item in plt.xticks()[1]]
        if (len(tick_labels) > 8):
            new_tick_labels = [label[:4] for label in tick_labels]
            plt.xticks(range(len(tick_labels)), new_tick_labels)
        elif (len(tick_labels) == 8):
            new_tick_labels = [label[:8] for label in tick_labels]
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

    def visualise_num_cat_variables(self, num_var, cat_var, stat='count'):
        plt.clf()
        binwidth = None
        if pd.api.types.is_integer_dtype(self.data[num_var]):
            binwidth = 1
        y_labels = self.data[cat_var].unique()
        color_idx = 0
        for label in y_labels:
            data = self.data[self.data[cat_var] == label][num_var]
            color = sns.color_palette('Set2')[color_idx]
            sns.histplot(data=data, color=color, stat=stat, binwidth=binwidth, kde=True, label=label)
            color_idx = color_idx + 1
        plt.legend(loc='upper right')
        self.display_plot(num_var + '_' + cat_var + '.png')
        return

    def visualise_num_cat_variables_fill(self, num_var, cat_var):
        plt.clf()
        binwidth = None
        if pd.api.types.is_integer_dtype(self.data[num_var]):
            binwidth = 1

        sns.histplot(x=num_var, hue=cat_var, data=self.data, stat="percent", multiple='fill', binwidth=binwidth, hue_order=['no', 'yes'])

        self.display_plot(num_var + '_' + cat_var + '.png')
        return

    def visualise_cat_pie_chart(self, cat_var):
        s = self.data[cat_var].value_counts()
        plt.pie(s, labels=s.index, autopct=self.autopct_format(s), colors=sns.color_palette('Set2'))
        self.display_plot(cat_var + '_pie.png')

    def autopct_format(self, values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

