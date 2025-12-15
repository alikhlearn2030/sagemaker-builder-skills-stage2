inference_code = r"""
import os
import joblib
import numpy as np

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    # Expecting CSV
    if request_content_type == "text/csv":
        data = np.fromstring(request_body, sep=",")
        return data.reshape(1, -1) if data.ndim == 1 else data
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    if response_content_type == "text/plain":
        return str(int(prediction[0])), response_content_type
    return str(prediction.tolist()), "application/json"
"""
with open("inference.py", "w", encoding="utf-8") as f:
    f.write(inference_code)

print("Created inference.py")
