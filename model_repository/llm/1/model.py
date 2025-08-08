# model_repository/gemma3/1/model.py
import sys
import os
import numpy as np
from pathlib import Path
import triton_python_backend_utils as pb_utils

repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from config import FAST_MODE, STANDARD_MODE, FULL_FEATURE_MODE, MAX_TOKENS, MODEL_TYPE, MODEL_SIZE
from llm_model import get_llm_models

class TritonPythonModel:
    def initialize(self, args):
        self.model = get_llm_models()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode("utf-8")
            
            output = self.model.generate(prompt)
            
            output_np = np.array([output], dtype=object)
            output_tensor = pb_utils.Tensor("generated_text", output_np)

            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses