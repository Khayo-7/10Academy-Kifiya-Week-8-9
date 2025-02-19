import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict, Any

class ModelFactory:
    """Generates model architectures for different algorithm types"""
    
    @staticmethod
    def logistic_regression() -> LogisticRegression:
        return LogisticRegression(class_weight='balanced',
                                solver='lbfgs',
                                max_iter=1000)

    @staticmethod
    def random_forest() -> RandomForestClassifier:
        return RandomForestClassifier(class_weight='balanced',
                                    n_estimators=100,
                                    n_jobs=-1)

    @staticmethod
    def mlp(input_shape: tuple) -> Sequential:
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model

    @staticmethod
    def cnn(input_shape: tuple) -> Sequential:
        model = Sequential([
            Reshape((input_shape[0], 1)), 
            Conv1D(64, 3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model

    @staticmethod
    def lstm(input_shape: tuple) -> Sequential:
        model = Sequential([
            Reshape((1, input_shape[0])), 
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model