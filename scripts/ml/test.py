import os
import mlflow
import tensorflow as tf


def load_model_from_artifact(artifact_path):
    model_path = os.path.join(artifact_path, "model.h5")
    return tf.keras.models.load_model(model_path)


def select_best_model(artifact_paths):
    best_accuracy = 0.0
    best_model = None

    for artifact_path in artifact_paths:
        model = load_model_from_artifact(artifact_path)
        # Assuming you logged accuracy as a metric during training
        run_id = artifact_path.split("/")[-1]  # Extract the run ID from the URI
        accuracy = mlflow.get_metric_history(run_id=run_id, key="accuracy")[-1]["value"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, best_accuracy


if __name__ == "__main__":
    # Get artifact paths from MLflow
    artifact_paths = mlflow.search_runs().artifact_uri

    # Select the best model
    best_model, best_accuracy = select_best_model(artifact_paths)

    print(f"Best model accuracy: {best_accuracy}")
    print("Best model saved to 'best_model.h5'")
    best_model.save("best_model.h5")
