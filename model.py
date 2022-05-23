import onnxruntime as ort
from scipy.special import softmax


class ModelInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def __call__(self, inputs, proba=False):
        predicts = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: inputs},
        )[0][0]
        if proba:
            predicts = softmax(predicts)
        return predicts
