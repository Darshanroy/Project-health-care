import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger
import pickle
import os

logger = get_logger(__name__)


@step
def model_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "knn" ) -> Annotated[
     ClassifierMixin,
     ArtifactConfig(name="sklearn_classifier", is_model_artifact=True),
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        model_type: The type of model to train.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
        :param model_type:
        :param X_train:
        :param y_train:
    """
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    if model_type == "sgd":
        model = SGDClassifier()
    elif model_type == "rf":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "knn":
        model = KNeighborsClassifier()
    elif model_type == "dt":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f"Unknown model type {model_type}")

    logger.info(f"Training {model_type} model...")

    model.fit(X_train, y_train)

    # Define the model file path
    model_file_path = os.path.join("artifacts", f"{model_type}_best_model.pkl")

    # Pickle the trained model
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    # MLflow integration
    with mlflow.start_run():
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", signature=signature
        )

    return model
