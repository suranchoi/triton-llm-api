from llama_cpp import Llama
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_path = "/your/model/path/bge-m3-gguf/bge-m3-F16.gguf"
    
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_ctx=0,
            logits_all=False,
            embedding=True,
            flash_attn=True,
            verbose=False
        )

    def execute(self, requests):
        request = requests[0]
        text = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode("utf-8")
        emb = self.model.create_embedding(text)["data"][0]["embedding"]
        return [pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("embedding_output", np.array(emb, dtype=np.float32).reshape(1, -1))
        ])]

