import onnxruntime as ort
import torch

print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch CUDA device: {torch.cuda.get_device_name(0)}")

print(f"ONNX Runtime version: {ort.__version__}")
print(f"ONNX Runtime Providers: {ort.get_available_providers()}")

try:
    # Create a dummy session to test provider availability
    sess_options = ort.SessionOptions()
    # sess_options.log_severity_level = 0
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # Just creating a session might not trigger the provider check fully until inference, 
    # but let's see if it errors out or warns.
    # We don't have a model here, so we can't fully test inference.
    print("CUDAExecutionProvider is in available providers list:", 'CUDAExecutionProvider' in ort.get_available_providers())
except Exception as e:
    print(f"Error checking providers: {e}")
