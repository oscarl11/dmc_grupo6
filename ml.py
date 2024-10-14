from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLSystem:
    def __init__(self, model=None):
        if model is None:
            self.model = RandomForestClassifier()
        else:
            self.model = model

    def load_data(self, X, y, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def predict(self, X_new):
        return self.model.predict(X_new)