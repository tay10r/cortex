import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession('scopenet-v1.0.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Preprocessing: image → tensor
def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # Add batch
    return arr

# Postprocessing: tensor → image
def postprocess(output: np.ndarray) -> Image.Image:
    output = output.squeeze(0)           # Remove batch
    output = output.transpose(1, 2, 0)   # CHW → HWC
    output = np.clip(output, 0, 1)       # Clamp
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)

# Inference
def reconstruct(image):
    input_tensor = preprocess(image)
    result = session.run([output_name], {input_name: input_tensor})[0]
    return postprocess(result)

# Gradio UI
iface = gr.Interface(fn=reconstruct,
                     inputs=gr.Image(type='pil', width=None, height=None),
                     outputs=gr.Image(type='pil', width=None, height=None),
                     title='Image Reconstruction (ONNX)')

iface.launch()
