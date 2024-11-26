import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the Product Quality Dataset (adjust the dataset path as needed)
# The dataset contains features related to product quality and the target variable 'quality'
data = pd.read_csv('Product_quality-classification.csv')

# Split data into features (X) and target (y)
# X contains all columns except the last one, which is assumed to be the target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable (Product Quality)

# Encode the target variable if it is categorical
# LabelEncoder is used to convert non-numeric labels into numeric form
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature Selection using Variance Threshold
# VarianceThreshold removes all features with low variance.
# Features with variance less than 0.01 are removed.
selector_variance = VarianceThreshold(threshold=0.01)
X_selected_variance = selector_variance.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected_variance, y_encoded, test_size=0.2, random_state=42
)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualize the Decision Tree (Optional)
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.columns[:-1], class_names=label_encoder.classes_)
plt.title("Decision Tree Visualization")
plt.show()
