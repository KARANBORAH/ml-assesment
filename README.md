# ml-assesment1. Data Preprocessin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle Infinity
df.dropna(inplace=True)  # Remove rows with missing values

# Encode labels (BENIGN=0, DDoS=1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Split features (X) and labels (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
2. SVM Model
python
Copy
# Initialize SVM with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svm.predict(X_test)
print("SVM Performance:")
print(classification_report(y_test, y_pred_svm))
Hyperparameter Tuning (Optional)
python
Copy
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, scoring='f1')
grid.fit(X_train, y_train)
best_svm = grid.best_estimator_
3. Neural Network (Keras)
python
Copy
# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Evaluate
y_pred_nn = (model.predict(X_test) > 0.5).astype(int)
print("\nNeural Network Performance:")
print(classification_report(y_test, y_pred_nn))
4. Key Considerations
Class Imbalance:

Use class_weight in SVM or sample_weight in Keras to balance classes.

Example for SVM:

python
Copy
svm = SVC(class_weight='balanced', ...)
Feature Engineering:

Add derived features like Packet Rate or Byte Rate if needed.

Model Optimization:

For SVM, tune C (regularization) and gamma (kernel width).

For the neural network, adjust layers, units, dropout rates, and learning rate.

Evaluation Metrics:

Prioritize recall to minimize false negatives (missed attacks).

Sample Output
Copy
SVM Performance:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1000
           1       0.98      0.98      0.98       500

    accuracy                           0.99      1500
   macro avg       0.99      0.99      0.99      1500
weighted avg       0.99      0.99      0.99      1500

Neural Network Performance:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1000
           1       0.98      0.98      0.98       500

    accuracy                           0.99      1500
   macro avg       0.99      0.99      0.99      1500
weighted avg       0.99      0.99      0.99      1500
