import numpy as np
import requests
import json


class TritonClient:
    def __init__(self, url: str = "localhost:8000"):
        if not url.startswith("http"):
            url = f"http://{url}"
        self.url = url

    def infer(self, model_name: str, input_data: np.ndarray):
        endpoint = f"{self.url}/v2/models/{model_name}/infer"

        payload = {
            "inputs": [
                {
                    "name": "images",
                    "shape": input_data.shape,
                    "datatype": "FP32",
                    "data": input_data.tolist(),
                }
            ]
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=60.0)

            if response.status_code != 200:
                print(f"TRITON ERROR RESPONSE: {response.text}")
                response.raise_for_status()
            result = response.json()
            output_data = result["outputs"][0]["data"]
            dims = result["outputs"][0]["shape"]
            return np.array(output_data).reshape(dims)

        except requests.exceptions.RequestException as e:
            print(f"CONNECTION ERROR: {e}")
            raise e
