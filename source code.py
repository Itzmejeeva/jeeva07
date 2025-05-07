
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('movies_dataset.csv')

# Data Preprocessing: Assuming you have features such as ratings, genres, etc.
# For simplicity, let's use a binary classification (e.g., will the user like the movie or not)

# Example: Preprocessing (customize as per your dataset)
df['liked'] = df['rating'].apply(lambda x: 1 if x >= 3 else 0)  # 1 for liked, 0 for disliked

# Feature selection: Assume you have user and movie features (customize based on your dataset)
features = ['user_id', 'movie_id', 'genre', 'other_feature']
X = df[features]
y = df['liked']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Support Vector Machine
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict with both models
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

# Evaluate models
print("Random Forest Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\nSupport Vector Machine Evaluation:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Visualizing the Confusion Matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Visualizing the Confusion Matrix for SVM
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
