# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.io as pio
import time
import random

# --- Set Plotly Renderer ---
pio.renderers.default = "browser"

# --- Utility Function for Styling ---
def print_separator(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50 + "\n")

# --- Load Dataset ---
print_separator("Reading Dataset")

df = pd.read_csv(r"C:\Users\PC\Documents\Summer 2025\Ellabs\Arduino project\simulated_circuit_data.csv") 

# --- Preview Dataset ---
print_separator("Previewing Dataset")
print(df.head(10))
print(df.describe())
print("Dataset Shape:", df.shape)
print("Null values check:\n", df.isnull().sum())
print("Target Labels:", df["Status"].unique())

# --- Shuffle Data ---
print_separator("Shuffling Dataset")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data shuffled successfully.")

# --- Extra Visualization: Histogram ---
print_separator("Visualizing Features Distribution")
for col in ["Pressure", "Speed", "Temperature"]:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# --- 3D Scatter Plot ---
print_separator("3D Visualization")
fig = px.scatter_3d(df, x="Pressure", y="Speed", z="Temperature", color="Status", title="3D Scatter of Features")
fig.show()

# --- Feature and Label Separation ---
print_separator("Separating Features and Target")
X = df[["Pressure", "Speed", "Temperature"]]  # Features
y = df["Status"]                             # Target labels

# --- Extra Analysis ---
print("Feature Ranges:")
for col in X.columns:
    print(f"{col}: min = {X[col].min()}, max = {X[col].max()}, mean = {X[col].mean()}")

# --- Data Split ---
print_separator("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Set Size:", X_train.shape)
print("Test Set Size:", X_test.shape)

# --- Model Creation ---
print_separator("Creating Classifier")
knn = KNeighborsClassifier(n_neighbors=3)
print("Model created with k=3")

# --- Training Model ---
print_separator("Training Model")
start_time = time.time()
knn.fit(X_train, y_train)
end_time = time.time()
print("Training completed in {:.4f} seconds".format(end_time - start_time))

# --- Prediction ---
print_separator("Making Predictions")
y_pred = knn.predict(X_test)
print("Predictions generated.")


# --- Accuracy Score ---
print_separator("Evaluating Accuracy")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Model:", round(accuracy * 100, 2), "%")

# --- Classification Report ---
print_separator("Classification Report")
report = classification_report(y_test, y_pred)
print(report)

# --- Confusion Matrix ---
print_separator("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# --- Extra: Manual Prediction Simulation ---
print_separator("Manual Predictions Simulation")
for _ in range(5):
    sample = X_test.sample(1)
    prediction = knn.predict(sample)
    print(f"Sample:\n{sample.to_string(index=False)}\n=> Predicted Status: {prediction[0]}")
    time.sleep(1)

# --- Ending Note ---
print_separator("Summary")
print("Model trained, evaluated, and tested with various visualizations and metrics.")
print("Further optimization (like hyperparameter tuning) can be considered in the future.")
