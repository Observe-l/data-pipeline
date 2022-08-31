import mlflow
from pprint import pprint

mlflow.set_tracking_uri("http://192.168.1.118:5000")
model_name = "mnist_best_model"
client = mlflow.tracking.MlflowClient()
# for rm in client.list_registered_models():
#     pprint(dict(rm), indent=4)

latest_version = int(client.get_latest_versions(name='mnist_best_model')[0].version)
client.transition_model_version_stage(
    name='mnist_best_model',
    version= latest_version,
    stage='Production',
    archive_existing_versions=True
)