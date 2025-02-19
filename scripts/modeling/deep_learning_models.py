import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from scripts.modeling.experiment_tracking import log_experiment

def build_mlp(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_shape):
    model = keras.Sequential([
        layers.SimpleRNN(64, activation='relu', input_shape=(input_shape, 1)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(input_shape, 1)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for fraud detection."""
    def __init__(self, input_dim: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Add CNN, RNN, LSTM, Transformer
def get_dl_model(model_name: str, input_dim: int):
    models = {
        "mlp": MLP(input_dim),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported DL model: {model_name}")
    return models[model_name]

def train_dl_model(X, y, model_name, epochs=DL_TRAIN_PARAMS["epochs"], batch_size=DL_TRAIN_PARAMS["batch_size"]):
    """Train a deep learning model using PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    model = get_dl_model(model_name, input_dim).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_TRAIN_PARAMS["lr"])

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    log_experiment(model_name, {"epochs": epochs}, {"loss": loss.item()})

    return model


def train_dl_models(model, X_train, y_train, X_test, y_test, model_name, epochs=10, batch_size=32):
    """
    Train a deep learning model and log it to MLflow.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape input for CNN, RNN, and LSTM models
    if len(model.input_shape) == 3:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

        # Save model in local directory
        os.makedirs("../resources/models", exist_ok=True)
        model.save(f"../resources/models/{model_name}.h5")

        # Log model to MLflow
        mlflow.tensorflow.log_model(model, model_name)

        # Log performance metrics to MLflow
        log_experiment(model, model_name, X_test, y_test, model_type="deep")

    print(f"Training complete for {model_name}. Model saved and logged to MLflow.")
