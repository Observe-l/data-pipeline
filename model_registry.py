import mlflow
from pprint import pprint

mlflow.set_tracking_uri("http://redpc:5000")
model_name = "mnist_best_model"
client = mlflow.tracking.MlflowClient()

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Production"
)


# def input_test(arg1:str=None):
#     if arg1:
#         print(arg1)
#     else:
#         print("Nothing")

# input_test()
# input_test("Have input")