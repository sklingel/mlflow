import os
from mlflow.pyfunc import scoring_server
from mlflow.pyfunc import load_model


def load_app(auth_token):
    return scoring_server.init(
        load_model(os.environ[scoring_server._SERVER_MODEL_PATH]),
        auth_token
    )
