import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib

df = pd.read_csv(r"./MIT-BIH Arrhythmia Database.csv")
df = df.iloc[:, 1:]

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=32)
X_reduced = pca.fit_transform(X_scaled)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "label_encoder.pkl")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reduced, y_encoded)

y_categorical = to_categorical(y_resampled)

X_reshaped = X_resampled.reshape((X_resampled.shape[0], 1, X_resampled.shape[1]))

X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Bidirectional(LSTM(32, return_sequences=True, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]))),
    BatchNormalization(),
    Dropout(0.4),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=100, 
    batch_size=32, 
    verbose=1, 
    callbacks=[early_stopping]
)

model.save("rnn_arrythmia_model.h5")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))
