import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # ADDED StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib # for saving the LabelEncoder and Scaler

# --- 1. Load features ---
print("Loading features...")
try:
    X = np.load("X.npy")
    y = np.load("y.npy")
except FileNotFoundError:
    print("Error: X.npy or y.npy not found. Did you run extract_features.py?")
    exit()

print(f"Data shapes: X{X.shape}, y{y.shape}")

# --- 2. Encode labels (string -> number -> one-hot) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)
num_classes = y_onehot.shape[1]

# --- 3. Feature Scaling (CRITICAL STEP) ---
print("Applying Standard Scaling...")
scaler = StandardScaler()
# Fit the scaler on all data, then transform it
X_scaled = scaler.fit_transform(X)

# --- 4. Train/test split ---
# We use X_scaled for the split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# --- 5. Build the Improved Deep Learning Model (MLP) ---
print("Building and compiling model...")
model = Sequential([
    # Input layer: 40 features (MFCCs)
    Dense(512, activation="relu", input_shape=(X.shape[1],)), # Increased size
    Dropout(0.4), 
    
    # Hidden layers
    Dense(256, activation="relu"),
    Dropout(0.4), 
    Dense(128, activation="relu"), 
    Dropout(0.3), 
    
    # Output layer: number of emotions
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- 6. Train ---
print("Starting training with 100 epochs...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, # Increased epochs for better learning
    batch_size=32,
    verbose=1
)

# --- 7. Evaluate & Save ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🔥 Test Accuracy: {acc*100:.2f}%")

# Save the trained model in the recommended native Keras format
model.save("emotion_model.keras") 
# Save the LabelEncoder and the Scaler
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib') # SAVING THE SCALER

print("✅ Model (emotion_model.keras), label encoder, and SCALER saved!")