import numpy as np
import triton_python_backend_utils as pb_utils
from llama_cpp import Llama
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_path = "/data2/huggingface/hub/models--tgisaturday--Docsray/snapshots/1f96aea426e018521ce2958eddf65240b3009ba4/multilingual-e5-large-gguf/multilingual-e5-large-F16.gguf"

        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_ctx=0,
            logits_all=False,
            embedding=True,
            verbose=False
        )

    def execute(self, requests):
        request = requests[0]
        text = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode("utf-8")
        emb = self.model.create_embedding(text)["data"][0]["embedding"]
        return [pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("embedding_output", np.array(emb, dtype=np.float32).reshape(1, -1))
        ])]

