"""Optional CPU ONNX inference with the same predict interface as sklearn."""

import numpy as np


class OnnxPredictor:
    def __init__(self, model_path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, texts):
        values = np.asarray(list(texts), dtype=object).reshape(-1, 1)
        return self.session.run(None, {self.input_name: values})[0]
