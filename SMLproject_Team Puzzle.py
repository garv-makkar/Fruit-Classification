import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load data
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Separate target variable
X = data.drop(["ID", "category"], axis=1)
y = data["category"]
X_test = test_data.drop(["ID"], axis=1)

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dimensionality Reduction (Only PCA in this version)
pca = PCA(n_components=min(X_train.shape[1], 100))  # Adjust based on explained variance
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
print("PCA completed")

# Clustering as additional feature
kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000, algorithm="elkan")
X_train_clusters = kmeans.fit_predict(X_train_pca).reshape(-1, 1)
X_val_clusters = kmeans.predict(X_val_pca).reshape(-1, 1)
X_test_clusters = kmeans.predict(X_test_pca).reshape(-1, 1)

# Combine original and clustered data
X_train_combined = np.hstack((X_train_pca, X_train_clusters))
X_val_combined = np.hstack((X_val_pca, X_val_clusters))
X_test_combined = np.hstack((X_test_pca, X_test_clusters))
print("Clustering done")

# Define models for ensemble
models = [
    ('bagging', BaggingClassifier(base_estimator=LogisticRegression(max_iter=500), n_estimators=50, random_state=42)),
    ('adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('logreg', LogisticRegression(max_iter=500, solver='liblinear')),
]

# Voting classifier to combine models
ensemble_model = VotingClassifier(estimators=models, voting='hard')

# Train VotingClassifier and evaluate accuracy
ensemble_model.fit(X_train_combined, y_train)
val_predictions = ensemble_model.predict(X_val_combined)
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy: ", val_accuracy)

# Predict on test data using best ensemble model
test_predictions = ensemble_model.predict(X_test_combined)
test_predictions = label_encoder.inverse_transform(test_predictions)

# Save predictions to CSV
submission = pd.DataFrame({"ID": test_data["ID"], "Category": test_predictions})
submission.to_csv("submission.csv", index=False)