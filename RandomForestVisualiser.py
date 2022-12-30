from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


class RandomForestVisualiser:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)
        smote = SMOTE(random_state=0)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def visualise(self):
        m = RandomForestClassifier(random_state=0)
        m.fit(self.X_train, self.y_train)

        # Extract single tree
        estimator = m.estimators_[5]
        export_graphviz(estimator, out_file='tree.dot',
                        feature_names=self.X_train.columns,
                        class_names=True,
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
