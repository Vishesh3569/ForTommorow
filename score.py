import os
import json
import joblib

def init():
    """
    Called once when the container starts. Loads the model into memory.
    """
    global model
    # Update the filename here to match your uploaded file
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'my_heart_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    """
    Called for every invocation of the endpoint.
    """
    try:
        # Assuming request payload is: {"data": [[val1, val2...]]}
        data = json.loads(raw_data)['data']
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
