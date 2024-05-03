import os
import mlflow

def print_mlflow_info():
    # Enable tracking to our remote MLFlow server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
    run_infos = mlflow.search_runs(search_all_experiments=True).to_dict('records')

    for run_info in run_infos:
        run_id = run_info["run_id"]  # Access run_id from dictionary
        metrics = mlflow.get_run(run_id).data.metrics
        print(f"Run ID: {run_id}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n")

def select_best_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
    # Get run information for recent runs
    run_infos = mlflow.search_runs(search_all_experiments=True).to_dict('records')
    best_accuracy = 0.0
    best_run_id = None
    best_model_path = None

    for run_info in run_infos:
        run_id = run_info["run_id"]  # Access run_id from dictionary
        metrics = mlflow.get_run(run_id).data.metrics
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id
            # Get the path to the checkpoint file
            best_model_path = f"/mlflow/artifacts/{run_id}/artifacts/checkpoint.h5"
            # Load the model from the checkpoint file
            #best_model_checkpoint = load_model(checkpoint_path)
            # best_model = mlflow.keras.load_model(f"runs:/{run_id}/model")

    #return best_run_id, best_accuracy
    return best_model_path, best_run_id, best_accuracy

if __name__ == "__main__":
    # Print MLflow information
    print_mlflow_info()

    # Select the best model
    best_model_path, best_run_id, best_accuracy = select_best_model()
    #best_run_id, best_accuracy = select_best_model()

    if best_run_id is not None:
        print(f"Best model accuracy: {best_accuracy}")
        print(f"Best model run ID: {best_run_id}")
        print(f"Best model access: {best_model_path}")
    else:
        print("No model found with recorded accuracy metric.")
