import torch
import onnxruntime
from c.cConst import Const
from models import SCRFD, ArcFace

var = Const()

def load_model(det_weight=None, rec_weight=None, conf_thres=None, use_gpu=False):
    """
    Load Face Detection and Recognition models.
    
    Args:
        det_weight: Path to detection model (default: from Const)
        rec_weight: Path to recognition model (default: from Const)
        conf_thres: Confidence threshold (default: from Const)
        use_gpu: Try to use GPU for ONNX Runtime (requires onnxruntime-gpu)
    """
    # Use provided paths or fallback to Const
    det_weight = det_weight or var.det_weight
    rec_weight = rec_weight or var.rec_weight
    conf_thres = conf_thres or var.confidence_thresh
    
    # Choose providers based on use_gpu flag
    if use_gpu:
        available_providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("✅ Face API will attempt GPU (CUDA) - may fallback to CPU if driver issues")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️ onnxruntime-gpu not installed. Face API using CPU")
            print("   To enable GPU: pip uninstall onnxruntime && pip install onnxruntime-gpu")
    else:
        providers = ['CPUExecutionProvider']
        print("ℹ️ Face API using CPU (ReID & YOLO use GPU)")
    
    # Try to create sessions with error handling
    try:
        detector_session = onnxruntime.InferenceSession(det_weight, providers=providers)
        # Check which provider is actually being used
        actual_provider = detector_session.get_providers()[0]
        if actual_provider == 'CUDAExecutionProvider':
            print("   ✅ Face Detection running on GPU")
        else:
            print("   ⚠️ Face Detection running on CPU (GPU not available)")
    except Exception as e:
        print(f"   ⚠️ GPU failed for Face Detection, falling back to CPU: {e}")
        detector_session = onnxruntime.InferenceSession(det_weight, providers=['CPUExecutionProvider'])
    
    try:
        recognizer_session = onnxruntime.InferenceSession(rec_weight, providers=providers)
        actual_provider = recognizer_session.get_providers()[0]
        if actual_provider == 'CUDAExecutionProvider':
            print("   ✅ Face Recognition running on GPU")
        else:
            print("   ⚠️ Face Recognition running on CPU")
    except Exception as e:
        print(f"   ⚠️ GPU failed for Face Recognition, falling back to CPU: {e}")
        recognizer_session = onnxruntime.InferenceSession(rec_weight, providers=['CPUExecutionProvider'])

    detector = SCRFD(session=detector_session, model_path=det_weight, input_size=(640, 640), conf_thres=conf_thres)
    recognizer = ArcFace(session=recognizer_session)
    return detector, recognizer