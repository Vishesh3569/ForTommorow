import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    """
    Deserializes the model from the .pkl file extracted to model_dir.
    """
    model_path = os.path.join(model_dir, "my_heart_model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """
    Parses the incoming request payload.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Assuming payload is {"inputs": [[val1, val2, ...]]}
        return np.array(data["inputs"])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generates predictions against the loaded model.
    """
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    """
    Formats the output prediction to be sent back to the client.
    """
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction.tolist()})
    raise ValueError(f"Unsupported content type: {response_content_type}")
