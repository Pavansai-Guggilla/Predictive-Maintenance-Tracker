# Uncomment the following line if running for the first time:
# !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.utils import to_categorical

# For reproducibility
np.random.seed(42)

#######################################
# 1. Load and Explore the Dataset
#######################################
df = pd.read_csv('predictive_maintenance.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataframe Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

print("\nUnique values in 'Type':", df['Type'].unique())

# Count plot for the target variable "Failure Type"
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Failure Type', order=df['Failure Type'].value_counts().index)
plt.title('Count of Each Failure Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histograms for numeric features
numeric_features = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[numeric_features].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Boxplots for numeric features
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(8, 6))
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

#######################################
# 2. Data Preprocessing (Original)
#######################################
features = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Failure Type'

X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode "Type" (ordinal: L, M, H)
oe = OrdinalEncoder(categories=[['L', 'M', 'H']])
X_train.loc[:, 'Type'] = oe.fit_transform(X_train[['Type']]).astype(int)
X_test.loc[:, 'Type'] = oe.transform(X_test[['Type']]).astype(int)

# Save the OrdinalEncoder
joblib.dump(oe, 'ordinal_encoder.joblib')

# Ensure all feature columns are numeric (float32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Custom encoding for target variable
categories = ['No Failure', 'Heat Dissipation Failure', 'Power Failure',
              'Overstrain Failure', 'Tool Wear Failure', 'Random Failures']
custom_encoder = {cat: i for i, cat in enumerate(categories)}
y_train_encoded = np.array([custom_encoder.get(cat, len(categories)) for cat in y_train])
y_test_encoded = np.array([custom_encoder.get(cat, len(categories)) for cat in y_test])

# One-hot encode target for Keras models
num_classes = len(categories)
y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

#######################################
# 3. Model 1: Random Forest Classifier (Baseline)
#######################################
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test_encoded, y_pred_rf)
print("\n=== Random Forest ===")
print(f"Test Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_rf, target_names=categories))
print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_rf))

#######################################
# 4. Prepare Data for Deep Learning Models
#######################################
# For LSTM and Transformer, we reshape to 3D: (samples, timesteps, features)
X_train_dl = np.expand_dims(X_train.values, axis=1)
X_test_dl = np.expand_dims(X_test.values, axis=1)
print("\nDeep Learning input shape:", X_train_dl.shape)

# For MLP, we use the original 2D data:
X_train_mlp = X_train.values
X_test_mlp = X_test.values

#######################################
# 5. Keras Tuner Setup for Deep Learning Models
#######################################
def build_lstm_model(hp):
    """
    Builds an LSTM model using hyperparameters from Keras Tuner.
    """
    model = Sequential()
    units = hp.Int("lstm_units", min_value=32, max_value=128, step=32, default=50)
    model.add(LSTM(units, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), activation='tanh'))
    
    dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_transformer_model(hp):
    """
    Builds a simple Transformer-like model using MultiHeadAttention and Keras Tuner.
    """
    # Input shape is (timesteps=1, features=6) for this dataset reshape
    inputs = Input(shape=(X_train_dl.shape[1], X_train_dl.shape[2]))
    
    # Hyperparameters for the attention mechanism
    num_heads = hp.Int('num_heads', 2, 8, step=2, default=4)
    key_dim = hp.Int('key_dim', 16, 64, step=16, default=32)
    
    # MultiHeadAttention requires query, key, and value. Here, we use the same input for simplicity.
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    
    # Optionally, you might add a dropout or residual connection. For simplicity, let's proceed:
    # (in practice, you'd typically add layer normalization + feed-forward network as well)
    pooled_output = GlobalAveragePooling1D()(attention_output)
    
    # Dense layers
    units_ff = hp.Int("transformer_ff_units", min_value=32, max_value=128, step=32, default=64)
    x = Dense(units_ff, activation='relu')(pooled_output)
    dropout_rate = hp.Float('dropout_transformer', min_value=0.0, max_value=0.5, step=0.1, default=0.1)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    learning_rate = hp.Choice('learning_rate_transformer', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_mlp_model(hp):
    """
    Builds a simple Multi-Layer Perceptron model using Keras Tuner.
    """
    model = Sequential()
    
    # We'll allow the user to choose how many units in the hidden layer(s):
    hidden_units = hp.Int("mlp_units", min_value=32, max_value=128, step=32, default=64)
    model.add(Dense(hidden_units, activation='relu', input_shape=(X_train_mlp.shape[1],)))
    
    dropout_rate = hp.Float('dropout_mlp', min_value=0.0, max_value=0.5, step=0.1, default=0.1)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # A second hidden layer (optional):
    hidden_units_2 = hp.Int("mlp_units_2", min_value=16, max_value=64, step=16, default=32)
    model.add(Dense(hidden_units_2, activation='relu'))
    
    dropout_rate_2 = hp.Float('dropout_mlp_2', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
    if dropout_rate_2 > 0:
        model.add(Dropout(dropout_rate_2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    learning_rate = hp.Choice('learning_rate_mlp', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------------------------------------------------------------------
# Below is a minimal placeholder showing how you might store final
# best models and their accuracies (so the rest of the script runs).
# In a real scenario, you'd run a KerasTuner search for each model.
# ------------------------------------------------------------------------

# Let's pretend we found these accuracies after tuning:
lstm_accuracy = 0.95       # Example placeholder 0.80
transformer_accuracy = 0.95  # Example placeholder0.82
mlp_accuracy = 0.95       # Example placeholder0.78

# Pretend "best" final models after tuning:
# (In real usage, you'd do something like: tuner = kt.RandomSearch(build_lstm_model, ...),
#  tuner.search(...), best_lstm_model = tuner.get_best_models(num_models=1)[0], etc.)
best_lstm_model = build_lstm_model(kt.HyperParameters())
best_transformer_model = build_transformer_model(kt.HyperParameters())
best_mlp_model = build_mlp_model(kt.HyperParameters())

#######################################
# 6. Model Comparison and Save Best Model (Original)
#######################################
model_accuracies_original = {
    "Random Forest": rf_accuracy,
    # Add other models' accuracy here if you want
}
best_model_name_original = max(model_accuracies_original, key=model_accuracies_original.get)
best_model_score_original = model_accuracies_original[best_model_name_original]
print(f"\n[Original Section] Best Model: {best_model_name_original} with Accuracy {best_model_score_original * 100:.2f}%")

if best_model_name_original == "Random Forest":
    joblib.dump(rf_model, "best_model.joblib")
# Add saving logic for other models (original placeholder)

metadata_original = {
    "model_type": "sklearn" if best_model_name_original == "Random Forest" else "keras",
    "input_shape": "3D" if best_model_name_original in ["LSTM", "Transformer"] else "2D",
    "best_model_name": best_model_name_original,
    "classes": categories
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata_original, f)

with open("target_classes.json", "w") as f:
    json.dump(categories, f)

#######################################
# 2. Data Preprocessing (Added Section)
#######################################
import json  # repeated import as requested

# (We re-show the same logic to demonstrate the "added" request, though it's repetitive.)
features = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Failure Type'

X = df[features].copy()
y = df[target].copy()

from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode "Type" (ordinal: L, M, H)
from sklearn.preprocessing import OrdinalEncoder
oe2 = OrdinalEncoder(categories=[['L', 'M', 'H']])
X_train2.loc[:, 'Type'] = oe2.fit_transform(X_train2[['Type']]).astype(int)
X_test2.loc[:, 'Type'] = oe2.transform(X_test2[['Type']]).astype(int)

# Save the OrdinalEncoder
joblib.dump(oe2, 'ordinal_encoder.joblib')  # same file name

# ... (rest of preprocessing remains the same)
# You might re-encode y, etc. in the same manner if needed.
# Ensure all feature columns are numeric (float32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Custom encoding for target variable
categories = ['No Failure', 'Heat Dissipation Failure', 'Power Failure',
              'Overstrain Failure', 'Tool Wear Failure', 'Random Failures']
custom_encoder = {cat: i for i, cat in enumerate(categories)}
y_train_encoded = np.array([custom_encoder.get(cat, len(categories)) for cat in y_train])
y_test_encoded = np.array([custom_encoder.get(cat, len(categories)) for cat in y_test])

# One-hot encode target for Keras models
from tensorflow.keras.utils import to_categorical
num_classes = len(categories)
y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes=num_classes)

#######################################
# 3. Model 1: Random Forest Classifier (Baseline)
#######################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test_encoded, y_pred_rf)
print("\n=== Random Forest ===")
print(f"Test Accuracy: {rf_accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_rf, target_names=categories))
print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred_rf))

#######################################
# 4. Prepare Data for Deep Learning Models
#######################################
# For LSTM and Transformer, we reshape to 3D: (samples, timesteps, features)
X_train_dl = np.expand_dims(X_train.values, axis=1)
X_test_dl = np.expand_dims(X_test.values, axis=1)
print("\nDeep Learning input shape:", X_train_dl.shape)

# For MLP, we use the original 2D data:
X_train_mlp = X_train.values
X_test_mlp = X_test.values

#######################################
# 5. Keras Tuner Setup for Deep Learning Models
#######################################
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, GlobalAveragePooling1D

##############################
# 5a. LSTM Hypermodel Builder
##############################
def build_lstm_model(hp):
    model = Sequential()
    # Tune the number of LSTM units
    units = hp.Int("lstm_units", min_value=32, max_value=128, step=32, default=50)
    model.add(LSTM(units, input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), activation='tanh'))
    # Optionally add dropout
    dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner_lstm = kt.RandomSearch(
    build_lstm_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='kt_lstm',
    project_name='lstm_tuning'
)
print("\nTuning LSTM model...")
tuner_lstm.search(X_train_dl, y_train_cat, epochs=10, validation_split=0.2, verbose=1)
best_hp_lstm = tuner_lstm.get_best_hyperparameters(num_trials=1)[0]
best_lstm_model = build_lstm_model(best_hp_lstm)
history_lstm = best_lstm_model.fit(X_train_dl, y_train_cat, epochs=20, batch_size=64, validation_split=0.2, verbose=1)
lstm_loss, lstm_accuracy = best_lstm_model.evaluate(X_test_dl, y_test_cat, verbose=0)
print(f"\n=== Tuned LSTM Model ===\nTest Loss: {lstm_loss:.4f}, Test Accuracy: {lstm_accuracy:.4f}")
y_pred_lstm = best_lstm_model.predict(X_test_dl)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
print("LSTM Classification Report:")
print(classification_report(y_test_encoded, y_pred_lstm_classes, target_names=categories))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Val Accuracy')
plt.title('Tuned LSTM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Val Loss')
plt.title('Tuned LSTM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

#######################################
# 5b. Transformer Hypermodel Builder
#######################################
def build_transformer_model(hp):
    inputs = Input(shape=(X_train_dl.shape[1], X_train_dl.shape[2]))
    # Tune number of attention heads and key_dim
    num_heads = hp.Int("num_heads", min_value=2, max_value=4, step=1, default=2)
    key_dim = hp.Choice("key_dim", values=[8, 16, 32], default=16)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    dropout_rate = hp.Float("dropout", 0.0, 0.5, step=0.1, default=0.0)
    if dropout_rate > 0:
        attention_output = Dropout(dropout_rate)(attention_output)
    pooled_output = GlobalAveragePooling1D()(attention_output)
    dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=16, default=32)
    x = Dense(dense_units, activation='relu')(pooled_output)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner_transformer = kt.RandomSearch(
    build_transformer_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='kt_transformer',
    project_name='transformer_tuning'
)
print("\nTuning Transformer model...")
tuner_transformer.search(X_train_dl, y_train_cat, epochs=10, validation_split=0.2, verbose=1)
best_hp_transformer = tuner_transformer.get_best_hyperparameters(num_trials=1)[0]
best_transformer_model = build_transformer_model(best_hp_transformer)
history_transformer = best_transformer_model.fit(X_train_dl, y_train_cat, epochs=20, batch_size=64, validation_split=0.2, verbose=1)
transformer_loss, transformer_accuracy = best_transformer_model.evaluate(X_test_dl, y_test_cat, verbose=0)
print(f"\n=== Tuned Transformer Model ===\nTest Loss: {transformer_loss:.4f}, Test Accuracy: {transformer_accuracy:.4f}")
y_pred_transformer = best_transformer_model.predict(X_test_dl)
y_pred_transformer_classes = np.argmax(y_pred_transformer, axis=1)
print("Transformer Classification Report:")
print(classification_report(y_test_encoded, y_pred_transformer_classes, target_names=categories))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_transformer.history['accuracy'], label='Train Accuracy')
plt.plot(history_transformer.history['val_accuracy'], label='Val Accuracy')
plt.title('Tuned Transformer Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_transformer.history['loss'], label='Train Loss')
plt.plot(history_transformer.history['val_loss'], label='Val Loss')
plt.title('Tuned Transformer Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

#######################################
# 5c. MLP Hypermodel Builder
#######################################
def build_mlp_model(hp):
    model = Sequential()
    units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32, default=64)
    model.add(Dense(units_1, input_dim=X_train_mlp.shape[1], activation='relu'))
    dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.3)
    model.add(Dropout(dropout_rate))
    units_2 = hp.Int("units_2", min_value=16, max_value=64, step=16, default=32)
    model.add(Dense(units_2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner_mlp = kt.RandomSearch(
    build_mlp_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='kt_mlp',
    project_name='mlp_tuning'
)
print("\nTuning MLP model...")
tuner_mlp.search(X_train_mlp, y_train_cat, epochs=10, validation_split=0.2, verbose=1)
best_hp_mlp = tuner_mlp.get_best_hyperparameters(num_trials=1)[0]
best_mlp_model = build_mlp_model(best_hp_mlp)
history_mlp = best_mlp_model.fit(X_train_mlp, y_train_cat, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
mlp_loss, mlp_accuracy = best_mlp_model.evaluate(X_test_mlp, y_test_cat, verbose=0)
print(f"\n=== Tuned MLP Model ===\nTest Loss: {mlp_loss:.4f}, Test Accuracy: {mlp_accuracy:.4f}")
y_pred_mlp = best_mlp_model.predict(X_test_mlp)
y_pred_mlp_classes = np.argmax(y_pred_mlp, axis=1)
print("MLP Classification Report:")
print(classification_report(y_test_encoded, y_pred_mlp_classes, target_names=categories))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['accuracy'], label='Train Accuracy')
plt.plot(history_mlp.history['val_accuracy'], label='Val Accuracy')
plt.title('Tuned MLP Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_mlp.history['loss'], label='Train Loss')
plt.plot(history_mlp.history['val_loss'], label='Val Loss')
plt.title('Tuned MLP Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

#######################################
# 6. Model Comparison and Save Best Model (Added Section)
#######################################
print("\n=== Model Comparison (Added Section) ===")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Tuned LSTM Accuracy:    {lstm_accuracy * 100:.2f}%")
print(f"Tuned Transformer Accuracy: {transformer_accuracy * 100:.2f}%")
print(f"Tuned MLP Accuracy:     {mlp_accuracy * 100:.2f}%")

model_accuracies = {
    "Random Forest": rf_accuracy,
    "LSTM": lstm_accuracy,
    "Transformer": transformer_accuracy,
    "MLP": mlp_accuracy
}
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model_score = model_accuracies[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy {best_model_score * 100:.2f}%")

# Save the best model
if best_model_name == "Random Forest":
    joblib.dump(rf_model, "best_model.joblib")
elif best_model_name == "LSTM":
    best_lstm_model.save("best_model.h5")
elif best_model_name == "Transformer":
    best_transformer_model.save("best_model.h5")
elif best_model_name == "MLP":
    best_mlp_model.save("best_model.h5")

# Save metadata
metadata = {
    "model_type": "sklearn" if best_model_name == "Random Forest" else "keras",
    "input_shape": "3D" if best_model_name in ["LSTM", "Transformer"] else "2D",
    "best_model_name": best_model_name,
    "classes": categories
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

# Save target classes
with open("target_classes.json", "w") as f:
    json.dump(categories, f)
