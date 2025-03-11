import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error




# Define a function to load the dataset
def load_dataset(dataset_source):
    if dataset_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv'])
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            iris = datasets.load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            return df
    elif dataset_source == "UCI Machine Learning Repository":
        dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])
        if dataset_name == "Iris":
            iris = datasets.load_iris()
            df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            return df
        elif dataset_name == "Wine":
            wine = datasets.load_wine()
            df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
            df['target'] = wine.target
            return df
        elif dataset_name == "Breast Cancer":
            breast_cancer = datasets.load_breast_cancer()
            df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
            df['target'] = breast_cancer.target
            return df
    

# Define a function to preprocess the data
def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']

    # Determine if the target is categorical or numerical
    is_classification = False
    if y.dtype == 'object' or len(y.unique()) <= 10:
        is_classification = True
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Separate numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing
    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(), categorical_cols)
            ]
        )
    elif len(numerical_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols)
            ]
        )
    elif len(categorical_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_cols)
            ]
        )
    else:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor and transform the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test

# Define a function to train the model
def train_model(X_train, X_test, y_train, y_test, model_type, C, epsilon, n_estimators, kernel, max_depth, n_neighbors, regularization, max_iter):
    if model_type == "Support Vector Classification (SVC)":
        model = SVC(C=C, kernel=kernel, probability=True)
    elif model_type == "Support Vector Regression (SVR)":
        model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(penalty=regularization, max_iter=max_iter)
    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "Decision Tree Regressor":
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elif model_type == "K-Nearest Neighbors Classifier":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == "K-Nearest Neighbors Regressor":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    return model

# Define a function to evaluate the model
def evaluate_model(model, X_test, y_test, model_type):
    y_pred = model.predict(X_test)

    if model_type in ["Support Vector Classification (SVC)", "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "K-Nearest Neighbors Classifier"]:
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        fig, ax = plt.subplots()
        cm.plot(ax=ax)
        st.pyplot(fig)
    else:
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Define a function to visualize the data
def visualize_data(df, model_type, model, X_test, y_test):
    if model_type in ["Support Vector Classification (SVC)", "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "K-Nearest Neighbors Classifier"]:
        # Plot the predicted probabilities
        y_pred_proba = model.predict_proba(X_test)
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Actual')
        ax.plot(y_pred_proba[:, 1], label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Plot the ROC curve
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            st.pyplot(fig)
        else:
            auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            st.write(f"ROC AUC Score: {auc:.2f}")

        # Plot the precision-recall curve
        if len(np.unique(y_test)) == 2:
            precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_proba[:, 1])
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            st.pyplot(fig)
        else:
            st.write("Precision-Recall Curve is not supported for multiclass problems")
    else:
        # Plot the predicted values
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.legend()
        st.pyplot(fig)

        # Plot the residuals
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.plot(residuals)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Residual')
        st.pyplot(fig)

        # Plot the Q-Q plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)

# Main function
def main():
    st.title("Machine Learning Playground")

    # Sidebar for dataset selection
    st.sidebar.header("Dataset")
    dataset_source = st.sidebar.selectbox("Select Dataset Source", ["Upload CSV", "UCI Machine Learning Repository"])

    df = load_dataset(dataset_source)

    st.write("Dataset Shape:")
    st.write(df.shape)
    st.write("Dataset Summary:")
    st.write(df.describe())

    # Select target variable
    st.subheader("Target Variable")
    target = 'target'

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Model selection
    st.subheader("Model Selection")
    model_type = st.selectbox("Select Model Type", 
                             ["Support Vector Classification (SVC)", 
                              "Support Vector Regression (SVR)", 
                              "Logistic Regression", 
                              "Decision Tree Classifier", 
                              "Decision Tree Regressor", 
                              "Random Forest Classifier", 
                              "Random Forest Regressor", 
                              "K-Nearest Neighbors Classifier",
                              "K-Nearest Neighbors Regressor"])

    # Hyperparameter selection
    st.subheader("Hyperparameter Selection")
    C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
    epsilon = st.slider("Epsilon (Margin of error)", 0.1, 10.0, 1.0)
    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
    max_depth = st.slider("Max Depth", 1, 20, 5)
    n_neighbors = st.slider("Number of Neighbors", 1, 10, 5)
    regularization = st.selectbox("Regularization", ["l1", "l2"])
    max_iter = st.slider("Max Iterations", 100, 1000, 200)

    # Train the model
    if st.button("Train Model"):
        model = train_model(X_train, X_test, y_train, y_test, model_type, C, epsilon, n_estimators, kernel, max_depth, n_neighbors, regularization, max_iter)
        evaluate_model(model, X_test, y_test, model_type)

    # Visualize the data
    if st.button("Visualize Data"):
        model = train_model(X_train, X_test, y_train, y_test, model_type, C, epsilon, n_estimators, kernel, max_depth, n_neighbors, regularization, max_iter)
        visualize_data(df, model_type, model, X_test, y_test)

if __name__ == "__main__":
    main()