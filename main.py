import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# read dataset
loan = pd.read_csv("loan_data.csv")

# encode 'purpose' column using label encoder
le = LabelEncoder()
loan['purpose'] = le.fit_transform(loan['purpose'])

X = loan.iloc[:, 1:-1]
y = loan.iloc[:, -1]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=79)

# function for decision tree
def train_decision_tree(X_train, X_test, y_train, y_test):

    dt_clf = DecisionTreeClassifier()

    param_grid = {'max_depth': [4, 8, 12, 16, 20]}

    grid = GridSearchCV(dt_clf, param_grid, scoring='recall', cv=5)

    grid.fit(X_train, y_train)

    # Predictions
    y_pred_train = grid.predict(X_train)
    y_pred_test = grid.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return grid, y_pred_test, train_accuracy, test_accuracy

# function for random forest
def train_random_forest(X_train, X_test, y_train, y_test):

    rf_clf = RandomForestClassifier(n_estimators=250)
    rf_clf.fit(X_train, y_train)

    y_pred_train = rf_clf.predict(X_train)
    y_pred_test = rf_clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return rf_clf, y_pred_test, train_accuracy, test_accuracy

# function for gradient boosting
def train_gradient_boosting(X_train, X_test, y_train, y_test):

    gb_clf = GradientBoostingClassifier(learning_rate=0.05)
    gb_clf.fit(X_train, y_train)

    y_pred_train = gb_clf.predict(X_train)
    y_pred_test = gb_clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return gb_clf, y_pred_test, train_accuracy, test_accuracy

