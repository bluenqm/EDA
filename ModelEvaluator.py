import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class ModelEvaluator:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)
        smote = SMOTE(random_state=0)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        self.results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

        y_pred_baseline = self.y_test.copy()
        y_pred_baseline.values[:] = 1
        acc = accuracy_score(self.y_test, y_pred_baseline)
        prec = precision_score(self.y_test, y_pred_baseline)
        rec = recall_score(self.y_test, y_pred_baseline)
        f1 = f1_score(self.y_test, y_pred_baseline)
        self.results = pd.concat([self.results, pd.DataFrame(
            {'Model': 'baseline', 'Accuracy': ["{:.2f}".format(acc)], 'Precision': ["{:.2f}".format(prec)], 'Recall': ["{:.2f}".format(rec)], 'F1 Score': ["{:.2f}".format(f1)]})])

    def evaluate(self):
        models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN', 'Neural Network']

        for model in models:
            if model == 'Logistic Regression':
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(random_state=0)
            elif model == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(random_state=0)
            elif model == 'SVM':
                from sklearn.svm import SVC
                m = SVC(random_state=0)
            elif model == 'Decision Tree':
                from sklearn.tree import DecisionTreeClassifier
                m = DecisionTreeClassifier(random_state=0)
            elif model == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                m = KNeighborsClassifier()
            elif model == 'Neural Network':
                from sklearn.neural_network import MLPClassifier
                m = MLPClassifier(random_state=0)
            else:
                raise ValueError(f'Invalid model: {model}')

            m.fit(self.X_train, self.y_train)
            y_pred = m.predict(self.X_test)

            # Calculate the accuracy, precision, recall, and F1 score
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            self.results = pd.concat([self.results, pd.DataFrame(
                {'Model': [model], 'Accuracy': ["{:.2f}".format(acc)], 'Precision': ["{:.2f}".format(prec)], 'Recall': ["{:.2f}".format(rec)], 'F1 Score': ["{:.2f}".format(f1)]})])

    def print_results(self):
        print(self.results)