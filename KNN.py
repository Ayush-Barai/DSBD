 # 8 assignment (KNN) TP,TN,FP,FN
 import pandas as pd
 import numpy as np
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import LabelEncoder
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
 # Load the iris dataset
 url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
 column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
 df = pd.read_csv(url, header=None, names=column_names)
 # Encode target variable (species) to numeric values
 label_encoder = LabelEncoder()
 df['species'] = label_encoder.fit_transform(df['species'])
 # Split the dataset into features and target variable
 X = df.drop('species', axis=1)  # Features
 y = df['species']  # Target variable
 # Split the dataset into training and test sets
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 # Initialize KNN classifier with k=3 (you can change k as needed)
 knn = KNeighborsClassifier(n_neighbors=3)
 # Fit the classifier on the training data
 knn.fit(X_train, y_train)
 # Predict on the test set
 y_pred = knn.predict(X_test)
 # Compute confusion matrix
 cm = confusion_matrix(y_test, y_pred)
 # Extract TP, FP, TN, FN from confusion matrix
 TP = cm[0, 0]  # True Positive
 FP = cm[0, 1]  # False Positive
 TN = cm[1, 1]  # True Negative
 FN = cm[1, 0]  # False Negative
 # Compute other metrics
 accuracy = accuracy_score(y_test, y_pred)
 error_rate = 1 - accuracy
 precision = precision_score(y_test, y_pred, average='weighted')
 recall = recall_score(y_test, y_pred, average='weighted')
 f1 = f1_score(y_test, y_pred, average='weighted')
 # Print results
 print("Confusion Matrix:")
 print(cm)
 print("\nTrue Positives (TP):", TP)
 print("False Positives (FP):", FP)
 print("True Negatives (TN):", TN)
 print("False Negatives (FN):", FN)
 print("\nAccuracy:", accuracy)
 print("Error Rate:", error_rate)
 print("Precision:", precision)
 print("Recall:", recall)