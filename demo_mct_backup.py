import os
# Configure PyTorch memory management to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch
# Demo Mode: Speed over Reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image2map'))
from map import Map
import api_server
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler

# --- Custom Logger Setup ---
loggers = {}

def get_floor_logger(floor_id):
    if floor_id not in loggers:
        logger = logging.getLogger(f"Floor_{floor_id}")
        logger.setLevel(logging.INFO)
        
        # Check if handler exists to avoid duplicates
        if not logger.handlers:
            # Create handler
            log_file = f"./logs/floor_{floor_id}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8') # 5MB
            
            # Format: Timestamp - Message
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.propagate = False # Do not print to console
            
        loggers[floor_id] = logger
    return loggers[floor_id]

import argparse
import numpy as np
import sys
import threading
import time
import json
import queue
import traceback # Added for debug
from collections import deque
from PIL import Image
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.transforms as T
from ultralytics import YOLO

# --- Setup Paths for API_Face ---
# Add API_Face to system path to allow imports
sys.path.append(os.path.join(os.getcwd(), 'API_Face'))

# Import TransReID modules
from config import cfg
from model import make_model

# Import API_Face modules
try:
    from API_Face.load_model import load_model
    from API_Face.c.cConst import Const
    from API_Face.service.processing import build_targets
    from API_Face.face_utils.helpers import compute_similarity as face_compute_sim
    print("✅ Successfully imported API_Face modules")
except ImportError as e:
    print(f"❌ Error importing API_Face modules: {e}")

# Import Database Config Module (optional - for auto-loading from DB)
try:
    from database.db_config import load_all_config, generate_active_cameras_list
    DB_CONFIG_AVAILABLE = True
    print("✅ Database config module available")
except ImportError:
    from database.db_config import load_all_config, generate_active_cameras_list
    DB_CONFIG_AVAILABLE = True
    print("✅ Database config module available")
except ImportError:
    DB_CONFIG_AVAILABLE = False
    print("⚠️ Database config module not available (will use command-line args)")

# Import MCT Tracker for Database Logging
try:
    from database.mct_tracking import get_mct_tracker
    MCT_TRACKING_AVAILABLE = True
    print("✅ MCT Tracking module available")
except ImportError as e:
    MCT_TRACKING_AVAILABLE = False
    print(f"⚠️ MCT Tracking module not available: {e}")

# --- Threaded Stream Class ---
class ThreadedStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.frame = None
        self.status = False
        self.stopped = False
        
        # Check if opened
        if self.capture.isOpened():
            self.status = True
            # Read first frame
            self.status, self.frame = self.capture.read()
        
        # Start thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
                if not status:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.status, self.frame if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

# --- Logging Functions ---
def get_vn_time():
    """Get current time in Vietnam timezone"""
    vn_tz = timezone(timedelta(hours=7))
    return datetime.now(vn_tz)

def load_json_file(file_path):
    """Load JSON file, return empty dict/list if not exists or error"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_json_file(file_path, data):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def remove_expired_names(log_dir, expire_seconds=60):
    """Remove entries older than expire_seconds from face_logs.json"""
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    face_logs = load_json_file(face_logs_path)
    
    current_time = get_vn_time()
    updated_logs = {}
    
    for name, data in face_logs.items():
        try:
            # Handle both old format (string) and new format (dict)
            if isinstance(data, str):
                timestamp_str = data
            elif isinstance(data, dict):
                timestamp_str = data.get('timestamp', data.get('time', ''))
            else:
                continue
                
            log_time = datetime.fromisoformat(timestamp_str)
            if (current_time - log_time).total_seconds() < expire_seconds:
                # Keep in new format
                if isinstance(data, dict):
                    updated_logs[name] = data
                else:
                    updated_logs[name] = {"timestamp": timestamp_str, "reid_id": None}
        except:
            pass
    
    save_json_file(face_logs_path, updated_logs)
    return updated_logs

def log_face_detection(name, cam_name, log_dir, person_id=None):
    """
    Log face detection to face_logs.json and backup_db_*.json
    
    Args:
        name: Person name (e.g., 'nv001')
        cam_name: Camera name (e.g., 'cam01')
        log_dir: Directory to store logs (e.g., './logs/cam01')
        person_id: ReID person ID (e.g., 5)
    
    Returns:
        bool: True if logged (new detection), False if already logged recently
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove expired entries
    face_logs = remove_expired_names(log_dir, expire_seconds=60)
    
    # Check if already logged in last 60 seconds
    if name in face_logs:
        return False
    
    # Get current time
    current_time = get_vn_time()
    time_str = current_time.isoformat()
    
    # Update face_logs.json (temporary log for 1 minute)
    face_logs[name] = {
        "timestamp": time_str,
        "reid_id": person_id
    }
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    save_json_file(face_logs_path, face_logs)
    
    # Add to backup_db_*.json (permanent log)
    backup_filename = f"backup_db_{current_time.strftime('%m%Y')}.json"
    backup_path = os.path.join(log_dir, backup_filename)
    
    backup_data = load_json_file(backup_path)
    if not isinstance(backup_data, list):
        backup_data = []
    
    entry = {
        "WriteDate": time_str,
        "SN": cam_name,
        "Pin": name,
        "AttTime": time_str
    }
    
    # Add ReID ID if available
    if person_id is not None:
        entry["ReID_ID"] = person_id
    
    backup_data.append(entry)
    
    save_json_file(backup_path, backup_data)
    
    if person_id is not None:
        print(f"📝 Logged: {name} (ID:{person_id}) at {cam_name} - {current_time.strftime('%H:%M:%S')}")
        
        # --- MCT DB LOGGING: Face Recognition ---
        if MCT_TRACKING_AVAILABLE:
            try:
                tracker = get_mct_tracker()
                # Determine floor (simple heuristic or need passed arg)
                # cam_name is like "cam01", map to floor if possible or default to unknown
                tracker.save_face_recognition(
                    local_track_id=person_id,
                    usr_id=name,
                    floor="Unknown", # Will be updated if pos is tracked, or could pass floor here
                    camera_id=cam_name,
                    confidence=1.0 # Detected face
                )
            except Exception as e:
                print(f"Warning: Failed to log face to DB: {e}")
                
    else:
        print(f"📝 Logged: {name} at {cam_name} - {current_time.strftime('%H:%M:%S')}")
    return True

# --- Directory Monitoring Functions ---
def count_directories_and_files(directory):
    """Count number of directories and files in a directory"""
    if not os.path.exists(directory):
        return 0, 0
    
    folder_count = 0
    file_count = 0
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_count += 1
            # Count files in subdirectory
            try:
                file_count += len([f for f in os.listdir(item_path) 
                                 if os.path.isfile(os.path.join(item_path, f))])
            except:
                pass
        elif os.path.isfile(item_path):
            file_count += 1
    
    return folder_count, file_count

def monitor_face_directory(faces_dir, update_queue, check_interval=10):
    """
    Monitor faces directory for changes and trigger reload.
    Runs in separate thread.
    
    Args:
        faces_dir: Path to faces directory
        update_queue: Queue to put update signals
        check_interval: Check every N seconds (default: 10)
    """
    print(f"🔍 Starting face directory monitor (checking every {check_interval}s)...")
    
    initial_folders, initial_files = count_directories_and_files(faces_dir)
    print(f"📊 Initial state - Folders: {initial_folders}, Files: {initial_files}")
    
    while True:
        try:
            time.sleep(check_interval)
            
            current_folders, current_files = count_directories_and_files(faces_dir)
            
            if initial_folders != current_folders or initial_files != current_files:
                print(f"\n{'='*60}")
                print(f"📢 FACE DIRECTORY CHANGE DETECTED!")
                print(f"   Previous: {initial_folders} folders, {initial_files} files")
                print(f"   Current:  {current_folders} folders, {current_files} files")
                print(f"{'='*60}")
                
                # Signal update needed
                update_queue.put('reload_faces')
                
                # Update tracking state
                initial_folders, initial_files = current_folders, current_files
                
        except Exception as e:
            print(f"⚠️ Error in face directory monitor: {e}")
            time.sleep(check_interval)

# --- Utility Functions ---
def compute_iou(box1, box2):
    # box: x1, y1, x2, y2
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def is_face_inside_body(face_box, body_box):
    """
    Check if face is essentially inside the body box (focused on top half)
    face_box: [x1, y1, x2, y2]
    body_box: [x1, y1, x2, y2]
    """
    fx_center = (face_box[0] + face_box[2]) / 2
    fy_center = (face_box[1] + face_box[3]) / 2
    
    bx1, by1, bx2, by2 = body_box
    
    # Check center containment
    if bx1 < fx_center < bx2 and by1 < fy_center < by2:
        return True
    return False

# --- ReID Setup ---
class PendingTrack:
    def __init__(self, box, feat):
        self.box = box
        self.feat = feat
        self.count = 1
        self.features = [feat]  # Store all features for later batch add

class ConfirmedTrack:
    def __init__(self, person_id, box, feat):
        self.person_id = person_id      # The stable, confirmed ID
        self.box = box
        self.feat = feat
        self.miss_count = 0
        self.vote_history = deque(maxlen=10) # Store last 10 raw ReID results
        self.vote_history.append(person_id)
        self.display_id = person_id     # ID currently displayed
        self.consecutive_matches = 0    # Track how stable the current raw vote is

    def update(self, box, feat, raw_voting_id):
        self.box = box
        self.feat = feat
        self.miss_count = 0
        self.vote_history.append(raw_voting_id)
        
        # Temporal Consistency Logic
        # Calculate most frequent ID in history
        if len(self.vote_history) > 0:
            counts = {}
            for vid in self.vote_history:
                counts[vid] = counts.get(vid, 0) + 1
            
            # Find winner
            best_id, count = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0]
            
            # Logic: Only switch ID if the new ID is dominant in recent history
            # If current display_id is still in history with decent support, keep it
            # to prevent flickering.
            
            HISTORY_LEN = len(self.vote_history)
            THRESHOLD = max(3, HISTORY_LEN // 2 + 1) # Majority vote
            
            if best_id != self.display_id:
                if count >= THRESHOLD:
                    self.display_id = best_id
                    # print(f"ID SWITCH: {self.person_id} -> {self.display_id}")
            else:
                # If winner is same as current, just reinforce
                pass


def setup_transreid(config_file, weights_path, device='cuda'):
    if config_file != "":
        cfg.merge_from_file(config_file)
    
    cfg.MODEL.DEVICE_ID = "0"

    # Check for local weights
    local_weights = os.path.abspath("weights/jx_vit_base_p16_224-80ecf9dd.pth")
    if os.path.exists(local_weights):
        cfg.MODEL.PRETRAIN_PATH = local_weights
    else:
        cfg.MODEL.PRETRAIN_PATH = ""
        
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading TransReID weights from {weights_path}")
        model.load_param(weights_path)
    else:
        print(f"Warning: ReID Weights not found at {weights_path}")
        
    model.to(device)
    model.eval()
    return model

def get_transforms(cfg):
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    return transform

def extract_feature(model, img, transform, device):
    """Extract single image feature"""
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    cam_label = torch.tensor([0], device=device)
    with torch.no_grad():
        feat = model(img, cam_label=cam_label)
    return feat.cpu().numpy()

def extract_features_batch(model, imgs, transform, device):
    """
    Extract features for multiple images at once (GPU batch processing)
    Much faster than calling extract_feature() multiple times
    
    Args:
        model: ReID model
        imgs: List of PIL Images
        transform: Image transform
        device: GPU device
    
    Returns:
        numpy array of features [N, feat_dim]
    """
    if len(imgs) == 0:
        return np.array([])
    
    # Transform all images
    batch = torch.stack([transform(img) for img in imgs]).to(device)
    cam_labels = torch.zeros(len(imgs), dtype=torch.long, device=device)
    
    with torch.no_grad():
        feats = model(batch, cam_label=cam_labels)
    
    return feats.cpu().numpy()

# --- Face API Setup & Wrapper ---
def setup_face_api():
    """
    Initialize Face API models and targets with FAISS index.
    Adjusts paths since we are running from root.
    
    Returns:
        detector: SCRFD face detector
        recognizer: ArcFace recognizer
        face_targets: List of (embedding, name) tuples
        face_index: FAISS index for fast face matching
        face_id_to_name: Dict mapping FAISS ID to name
    """
    print("Setting up Face API...")
    
    # Adjust paths in Const to be relative to root
    # Original paths were ./weights/..., assuming run from API_Face/
    # We change them to ./API_Face/weights/...
    det_weight_path = os.path.abspath("./API_Face/weights/det_10g.onnx")
    rec_weight_path = os.path.abspath("./API_Face/weights/w600k_r50.onnx")
    faces_dir_path = os.path.abspath("./API_Face/faces/")
    
    if not os.path.exists(det_weight_path):
        print(f"❌ Error: Face detection weights not found at {det_weight_path}")
        raise FileNotFoundError(f"Missing weights: {det_weight_path}")
    if not os.path.exists(rec_weight_path):
        print(f"❌ Error: Face recognition weights not found at {rec_weight_path}")
        raise FileNotFoundError(f"Missing weights: {rec_weight_path}")
    
    print(f"✅ Found Face weights at: {det_weight_path}")
    
    # Update Const class with correct paths (for other code that might use it)
    Const.det_weight = det_weight_path
    Const.rec_weight = rec_weight_path
    Const.faces_dir = faces_dir_path

    # Load models with explicit paths
    # Try GPU first for Face API (requires onnxruntime-gpu)
    # Install: pip install onnxruntime-gpu
    detector, recognizer = load_model(
        det_weight=det_weight_path,
        rec_weight=rec_weight_path,
        conf_thres=Const.confidence_thresh,
        use_gpu=True  # Enable GPU for Face API - will auto-fallback to CPU if needed
    )
    
    # Load targets (known faces)
    print(f"Loading known faces from {faces_dir_path}...")
    face_targets = build_targets(detector, recognizer, faces_dir_path)
    print(f"✅ Loaded {len(face_targets)} known faces.")
    
    # Build FAISS index for fast face matching
    import faiss
    
    if len(face_targets) == 0:
        print("⚠️ No faces loaded, skipping FAISS index creation")
        return detector, recognizer, face_targets, None, {}
    
    print("Building FAISS index for face recognition...")
    
    # Extract embeddings and names
    embeddings = []
    face_id_to_name = {}
    
    for idx, (embedding, name) in enumerate(face_targets):
        embeddings.append(embedding)
        face_id_to_name[idx] = name
    
    # Stack all embeddings
    embeddings_np = np.vstack(embeddings)  # [N, embedding_dim]
    
    # Normalize embeddings (for cosine similarity)
    faiss.normalize_L2(embeddings_np)
    
    # Create FAISS index (Inner Product = Cosine Similarity after normalization)
    embedding_dim = embeddings_np.shape[1]
    face_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product
    face_index.add(embeddings_np)
    
    print(f"✅ FAISS face index built with {len(face_targets)} faces (dim={embedding_dim})")
    
    return detector, recognizer, face_targets, face_index, face_id_to_name

def rebuild_face_index(detector, recognizer, faces_dir):
    """
    Rebuild face targets and FAISS index.
    Called when faces directory changes.
    
    Returns:
        face_targets: List of (embedding, name) tuples
        face_index: New FAISS index
        face_id_to_name: New ID mapping
    """
    import faiss
    
    print("🔄 Rebuilding face targets and FAISS index...")
    
    # Rebuild targets
    face_targets = build_targets(detector, recognizer, faces_dir)
    print(f"   ✅ Loaded {len(face_targets)} known faces")
    
    if len(face_targets) == 0:
        print("   ⚠️ No faces loaded")
        return face_targets, None, {}
    
    # Extract embeddings and names
    embeddings = []
    face_id_to_name = {}
    
    for idx, (embedding, name) in enumerate(face_targets):
        embeddings.append(embedding)
        face_id_to_name[idx] = name
    
    # Stack and normalize
    embeddings_np = np.vstack(embeddings)
    faiss.normalize_L2(embeddings_np)
    
    # Create new FAISS index
    embedding_dim = embeddings_np.shape[1]
    face_index = faiss.IndexFlatIP(embedding_dim)
    face_index.add(embeddings_np)
    
    print(f"   ✅ FAISS index rebuilt (dim={embedding_dim})")
    
    return face_targets, face_index, face_id_to_name

def run_face_recognition(frame, detector, recognizer, face_index, face_id_to_name):
    """
    Pure function to detect and recognize faces using FAISS for fast matching.
    
    Args:
        frame: Input frame
        detector: SCRFD detector
        recognizer: ArcFace recognizer
        face_index: FAISS index for face embeddings
        face_id_to_name: Dict mapping FAISS ID to name
    
    Returns: 
        List of dicts {'box': [x1,y1,x2,y2], 'name': str, 'score': float}
    """
    results = []
    
    # Detect
    # max_num=0 means detect all faces
    bboxes, kpss = detector.detect(frame, max_num=0)
    
    if bboxes is None or len(bboxes) == 0:
        return results

    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        # bbox is [x1, y1, x2, y2, score]
        box = bbox[:4].astype(np.int32)
        det_score = bbox[4]
        
        # Recognize
        embedding = recognizer(frame, kps)
        
        max_similarity = 0
        best_match_name = "Unknown"
        
        # Use FAISS for fast matching (10-50x faster than linear search)
        if face_index is not None and len(face_id_to_name) > 0:
            import faiss
            
            # Normalize embedding for cosine similarity
            embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(embedding_normalized)
            
            # Search for top-1 match
            D, I = face_index.search(embedding_normalized, k=1)
            
            # D[0][0] is cosine similarity (inner product after normalization)
            max_similarity = float(D[0][0])
            
            if max_similarity > Const.similarity_thresh:
                face_id = int(I[0][0])
                best_match_name = face_id_to_name.get(face_id, "Unknown")
                
        results.append({
            'box': box,
            'name': best_match_name,
            'sim_score': max_similarity,
            'det_score': det_score
        })
        
    return results

# --- Main Logic ---
def run_demo(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- MCT DB LOGGING: Start Session ---
    if MCT_TRACKING_AVAILABLE:
        tracker = get_mct_tracker()
        session_id = tracker.start_session()
        print(f"🚀 MCT Session Started: {session_id}")

    # 1. Initialize YOLO (Person Detection)
    print("Loading YOLO11n...")
    yolo = YOLO('yolo11x.pt')
    yolo.to(device)

    # 2. Initialize TransReID
    print("Loading TransReID...")
    reid_model = setup_transreid(args.config_file, args.weights, device)
    reid_transform = get_transforms(cfg)

    # 3. Initialize Face API with FAISS index
    face_detector, face_recognizer, face_targets, face_index, face_id_to_name = setup_face_api()
    
    # 3.1 Setup Face Directory Monitoring
    faces_dir_path = os.path.abspath("./API_Face/faces/")
    face_update_queue = queue.Queue()
    face_resources_lock = threading.Lock()  # Lock for face_index and face_id_to_name
    
    # Start face directory monitor thread
    monitor_thread = threading.Thread(
        target=monitor_face_directory,
        args=(faces_dir_path, face_update_queue, 10),  # Check every 10 seconds
        daemon=True
    )
    monitor_thread.start()

    # 3.2 Initialize Map for Visualization
    print("Initializing Map Visualization...")
    map_dir = "image2map"
    
    # Initialize Floor 3 Map (Default)
    try:
        mapper3 = Map(map_image_path=os.path.join(map_dir, "F3", "3f.png"),
                     mm_per_pixel_x=23.8, mm_per_pixel_y=23.3)
        # Try loading ROIs from F3
        rois_f3 = os.path.join(map_dir, "F3", "rois.yaml")
        if os.path.exists(rois_f3):
            mapper3.load_rois_from_yaml(rois_f3)
        print("✅ Map Floor 3 initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Map Floor 3: {e}")
        mapper3 = None

    # Initialize Floor 1 Map (New)
    # Initialize Floor 1 Map (New)
    try:
        # Floor 1 map image
        mapper1 = Map(map_image_path=os.path.join(map_dir, "F1", "1f.png"),
                     mm_per_pixel_x=23.8, mm_per_pixel_y=23.3)
        # Try loading ROIs from F1
        rois_f1 = os.path.join(map_dir, "F1", "rois.yaml")
        if os.path.exists(rois_f1):
            mapper1.load_rois_from_yaml(rois_f1)
        print("✅ Map Floor 1 initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Map Floor 1: {e}")
        mapper1 = None
        
    # Define active cameras based on args or database
    # Format: {'id': 'camX', 'url': 'rtsp://...', 'floor': 1 or 3}
    active_cameras = []
    db_config = None
    
    # ==========================================================================
    # MODE 1: Load from Database (if --use_db flag is set)
    # ==========================================================================
    if getattr(args, 'use_db', False) and DB_CONFIG_AVAILABLE:
        print("\n" + "="*60)
        print("🗄️  Loading camera configuration from DATABASE")
        print("="*60)
        
        # Get floor filter from args (e.g., "3F,1F" -> ["3F", "1F"])
        floor_filter = None
        if hasattr(args, 'floors') and args.floors:
            floor_filter = [f.strip() for f in args.floors.split(',')]
            print(f"📌 Filtering floors: {floor_filter}")
        
        try:
            db_config = load_all_config(floor_filter=floor_filter)
            active_cameras = generate_active_cameras_list(db_config['cameras'])
            print(f"✅ Loaded {len(active_cameras)} cameras from database")
            
            # Print loaded cameras
            for cam in active_cameras:
                print(f"   - {cam['name']} ({cam['id']}) -> Floor {cam['floor']} [{cam['inout']}]")
            
        except Exception as e:
            print(f"❌ Failed to load from database: {e}")
            print("⚠️ Falling back to command-line arguments...")
            import traceback
            traceback.print_exc()
    
    # ==========================================================================
    # MODE 2: Load from Command-line Arguments (default/fallback)
    # ==========================================================================
    if not active_cameras:
        print("\n📋 Using command-line arguments for camera configuration")
        
        if hasattr(args, 'rtsp1') and args.rtsp1: 
            active_cameras.append({'id': 'cam1', 'url': args.rtsp1, 'floor': getattr(args, 'floor_cam1', 3)})
        if hasattr(args, 'rtsp2') and args.rtsp2: 
            active_cameras.append({'id': 'cam2', 'url': args.rtsp2, 'floor': getattr(args, 'floor_cam2', 3)})
        if hasattr(args, 'rtsp3') and args.rtsp3: 
            active_cameras.append({'id': 'cam3', 'url': args.rtsp3, 'floor': getattr(args, 'floor_cam3', 3)})
        if hasattr(args, 'rtsp4') and args.rtsp4: 
            active_cameras.append({'id': 'cam4', 'url': args.rtsp4, 'floor': getattr(args, 'floor_cam4', 3)})
        if hasattr(args, 'rtsp1T1') and args.rtsp1T1:
            active_cameras.append({
                'id': 'cam1T1',
                'url': args.rtsp1T1,
                'floor': getattr(args, 'floor_cam1T1', 1),
                'map_id': 'cam1'
            })
    
    # ==========================================================================
    # Configure Cameras to respective Maps
    # ==========================================================================
    cam_id_to_floor = {}  # Store for later use in main loop
    
    for cam in active_cameras:
        cam_id = cam['id']
        floor = cam['floor']
        cam_id_to_floor[cam_id] = floor
        
        # Determine floor folder
        floor_folder = f"F{floor}"  # e.g., F1, F3
        
        # Use camera IP as calibration folder name (e.g., image2map/F1/10.29.98.52/)
        # Fallback to map_id or cam_id if IP not available
        calib_folder = cam.get('ip') or cam.get('map_id', cam_id)
        intrinsic_path = os.path.join(map_dir, floor_folder, f"{calib_folder}/intrinsic.yaml")
        extrinsic_path = os.path.join(map_dir, floor_folder, f"{calib_folder}/extrinsic.yaml")
        
        # Check if calibration files exist
        if not os.path.exists(intrinsic_path) or not os.path.exists(extrinsic_path):
            print(f"⚠️ Calibration files not found for {cam_id} on Floor {floor}")
            print(f"   Expected: {intrinsic_path}")
            print(f"   Skipping camera-to-map registration (tracking will still work)")
            continue
        
        if floor == 1:
            if mapper1:
                mapper1.add_camera(camera_id=cam_id,
                                  intrinsic=intrinsic_path,
                                  extrinsic=extrinsic_path)
            else:
                print(f"⚠️ Warning: {cam_id} assigned to Floor 1 but Map Floor 1 not initialized.")
        elif floor == 3:
            if mapper3:
                mapper3.add_camera(camera_id=cam_id,
                                  intrinsic=intrinsic_path,
                                  extrinsic=extrinsic_path)
            else:
                print(f"⚠️ Warning: {cam_id} assigned to Floor 3 but Map Floor 3 not initialized.")
        else:
            print(f"⚠️ Warning: Unknown floor {floor} for camera {cam_id}. Skipping map config.")

    # 4. Open RTSP streams
    streams = []
    stream_to_cam_id = {}
    
    for idx, cam in enumerate(active_cameras):
        cam_id = cam['id']
        url = cam['url']
        print(f"Connecting to {cam_id} (Floor {cam['floor']}): {url}...")
        stream = ThreadedStream(url)
        if not stream.status:
            print(f"Failed to open RTSP stream: {url}")
        streams.append(stream)
        stream_to_cam_id[idx] = cam_id
    
    # Map stream index to full camera info for logging
    stream_to_cam_info = {idx: cam for idx, cam in enumerate(active_cameras)}

    # Initialize FAISS index
    import faiss
    
    index = None 
    MATCH_THRESH = 0.7  # Cosine similarity threshold
    MAX_VECTORS_PER_ID = 200  # Max feature vectors per person
    TOP_K_VOTING = 1  # Top-K nearest vectors for voting (k=1)
    # Adaptive voting: k=1→need 1 vote, k=2→need 1 vote, k=3→need 2 votes
    
    vector_id_to_person_id = {} 
    person_vector_ids = {} 
    
    # NEW: Mapping ReID Person ID -> Face Name (persistent)
    person_id_to_face_name = {}
    
    next_person_id = 0
    next_vector_id = 0 
    
    pending_tracks = {}
    confirmed_tracks = {} # {cam_idx: [ConfirmedTrack, ...]}
    CONFIRM_FRAMES = 2  # Reduced from 4 to 2 for faster ID assignment
    MAX_MISS_FRAMES = 1000 # Delete track if lost for N frames

    
    # Thread-safe locks for shared resources
    faiss_lock = threading.Lock()
    person_id_lock = threading.Lock()
    vector_id_lock = threading.Lock()
    
    def process_camera_frame(cam_idx, frame, shared_state):
        """
        Process single camera frame with Face Recognition + ReID
        Thread-safe function to be called in parallel
        
        Args:
            cam_idx: Camera index
            frame: Frame to process
            shared_state: Dict containing all shared resources and locks
            
        Returns:
            Processed frame with annotations
        """
        nonlocal index, next_person_id, next_vector_id
        
        # Unpack Shared State
        cam_id_to_floor = shared_state.get('cam_id_to_floor', {})
        stream_to_cam_id_map = shared_state.get('stream_to_cam_id', {})
        
        # Determine Floor and Logger
        cam_id_str = stream_to_cam_id_map.get(cam_idx, f"cam{cam_idx+1}")
        floor_id = cam_id_to_floor.get(cam_id_str, 3) # Default 3
        data_logger = get_floor_logger(floor_id)

        # --- STEP A: Run Face Recognition (with FAISS) ---
        # Use lock to safely read face_index and face_id_to_name
        # (they might be updated by reload thread)
        with face_resources_lock:
            current_face_index = face_index
            current_face_id_to_name = face_id_to_name
        
        face_results = run_face_recognition(frame, face_detector, face_recognizer, 
                                           current_face_index, current_face_id_to_name)
        
        # --- STEP B: Run Person Detection & ReID ---
        if cam_idx not in pending_tracks:
            pending_tracks[cam_idx] = []
        
        # Define camera name and log directory for logging
        # Use real camera name/IP from database
        cam_info = shared_state.get('stream_to_cam_info', {}).get(cam_idx, {})
        cam_name = cam_info.get('name') or cam_info.get('ip') or f"cam{cam_idx+1:02d}"
        # Sanitize cam_name for directory (replace spaces, special chars)
        cam_name_safe = cam_name.replace(' ', '_').replace('/', '_')
        log_dir = f"./logs/{cam_name_safe}"

        assigned_ids = set() 
        resolved_boxes = []  
        detections = [] 

        # Detect Persons
        yolo_results = yolo(frame, classes=[0], verbose=False, device=device, conf=0.5, iou=0.4)
        
        # Collect all person crops for batch processing
        person_crops = []
        person_boxes = []
        blur_scores = []
        
        for r in yolo_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0: continue
                
                # Quality Check (minimal CPU work)
                gray_crop = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                
                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                
                person_crops.append(person_pil)
                person_boxes.append((x1, y1, x2, y2))
                blur_scores.append(blur_score)
        
        # Batch extract ReID features (GPU accelerated)
        if len(person_crops) > 0:
            feats = extract_features_batch(reid_model, person_crops, reid_transform, device)
            
            # Initialize FAISS if needed (with lock)
            with faiss_lock:
                if index is None:
                    feature_dim = feats.shape[1]
                    print(f"Initializing FAISS index with dim {feature_dim}")
                    index = faiss.IndexIDMap(faiss.IndexFlatIP(feature_dim))
            
            # Normalize all features at once (GPU operation via numpy)
            faiss.normalize_L2(feats)
            
            # Build detections
            for i, (box, feat, blur_score) in enumerate(zip(person_boxes, feats, blur_scores)):
                is_sharp = blur_score > 100.0
                detections.append({'box': box, 'feat': feat.reshape(1, -1), 'is_sharp': is_sharp})

        # --- STEP B.1: Track Association (Temporal Consistency) ---
        if cam_idx not in confirmed_tracks:
            confirmed_tracks[cam_idx] = []
            
        active_confirmed = []
        unmatched_detections = []
        
        # 1. Associate Detections with Confirmed Tracks using IoU
        # Simple Greedy Matching
        matched_track_indices = set()
        matched_det_indices = set()
        
        # Sort tracks by size/confidence if needed, here just simple loop
        # We want to match existing tracks first
        
        for t_idx, track in enumerate(confirmed_tracks[cam_idx]):
            best_iou = 0.5 # Min IoU threshold
            best_d_idx = -1
            
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_indices: continue
                
                iou = compute_iou(track.box, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            
            if best_d_idx != -1:
                # Match Found
                matched_track_indices.add(t_idx)
                matched_det_indices.add(best_d_idx)
                
                det = detections[best_d_idx]
                
                # --- ReID Voting for Existing Track ---
                feat = det['feat']
                raw_voting_id = -1
                sim_score = 0.0
                
                # Perform ReID search to get "Raw Vote"
                with faiss_lock:
                    if index is not None and index.ntotal > 0:
                        k = min(TOP_K_VOTING, index.ntotal)
                        D, I = index.search(feat, k)
                        
                        votes = {}
                        for i in range(k):
                            similarity = float(D[0][i])
                            if similarity > MATCH_THRESH:
                                vec_id = int(I[0][i])
                                if vec_id in vector_id_to_person_id:
                                    pid = vector_id_to_person_id[vec_id]
                                    if pid not in votes:
                                        votes[pid] = {'count': 0, 'max_sim': 0.0}
                                    votes[pid]['count'] += 1
                                    votes[pid]['max_sim'] = max(votes[pid]['max_sim'], similarity)
                        
                        if votes:
                            sorted_votes = sorted(votes.items(), 
                                                key=lambda x: (x[1]['count'], x[1]['max_sim']), 
                                                reverse=True)
                            winner_pid, winner_data = sorted_votes[0]
                            required_votes = max(1, (k + 1) // 2)
                            
                            if winner_data['count'] >= required_votes:
                                raw_voting_id = winner_pid
                                sim_score = winner_data['max_sim']
                
                if raw_voting_id == -1:
                    raw_voting_id = track.display_id
                
                # Update Track with Temporal Consistency
                track.update(det['box'], feat, raw_voting_id)
                track.current_sim_score = sim_score # Store for drawing
                track.current_is_sharp = det['is_sharp']
                active_confirmed.append(track)

            else:
                # Track lost in this frame
                track.miss_count += 1
                if track.miss_count <= MAX_MISS_FRAMES:
                    active_confirmed.append(track)
        
        # --- CONFLICT RESOLUTION: Ensure Unique IDs in Active Tracks ---
        id_usage = {}
        for track in active_confirmed:
            if track.miss_count == 0: # Only check currently visible tracks
                pid = track.display_id
                if pid not in id_usage: id_usage[pid] = []
                id_usage[pid].append(track)
        
        for pid, tracks in id_usage.items():
            if len(tracks) > 1:
                # Duplicate ID detected!
                # Sort by stability (vote history count for this ID)
                tracks.sort(key=lambda t: t.vote_history.count(pid), reverse=True)
                
                # Winner is tracks[0], others must change
                for loser in tracks[1:]:
                    with person_id_lock:
                        new_id = next_person_id
                        next_person_id += 1
                    
                    # Force ID switch for loser
                    loser.person_id = new_id
                    loser.display_id = new_id
                    loser.vote_history.clear()
                    loser.vote_history.append(new_id)
                    # print(f"CONFLICT RESOLVED: Forced ID change to {new_id}")

        # --- FACE NAME CONFLICT RESOLUTION ---
        # Ensure one face name is not assigned to multiple tracks in the same frame
        track_face_matches = {} # track_idx -> (name, score)
        face_name_usage = {}    # name -> [(score, track_idx)]

        # 1. Collect all potential face matches for active tracks
        for idx, track in enumerate(active_confirmed):
            if track.miss_count > 0: continue
            
            best_face_name = None
            best_face_score = 0.0
            
            for face in face_results:
                if is_face_inside_body(face['box'], track.box):
                    if face['name'] != "Unknown":
                        # If multiple faces inside body (rare), pick highest score
                        if face['sim_score'] > best_face_score:
                            best_face_score = face['sim_score']
                            best_face_name = face['name']
            
            if best_face_name:
                track_face_matches[idx] = (best_face_name, best_face_score)
                if best_face_name not in face_name_usage:
                    face_name_usage[best_face_name] = []
                face_name_usage[best_face_name].append((best_face_score, idx))

        # 2. Resolve duplicates
        final_track_faces = {} # track_idx -> name
        
        for name, matches in face_name_usage.items():
            if len(matches) > 1:
                # Sort by score descending
                matches.sort(key=lambda x: x[0], reverse=True)
                # Winner takes the name
                winner_score, winner_idx = matches[0]
                final_track_faces[winner_idx] = name
            else:
                final_track_faces[matches[0][1]] = name

        # --- DRAWING & FINALIZING CONFIRMED TRACKS ---
        for idx, track in enumerate(active_confirmed):
            if track.miss_count > 0: continue
            
            # Add to assigned_ids to prevent re-use by new detections
            assigned_ids.add(track.display_id)
            resolved_boxes.append(track.box)
            
            # Update Feature Bank (if sharp and consistent)
            if track.display_id == track.vote_history[-1] and track.current_sim_score < 0.95 and track.current_is_sharp:
                 with vector_id_lock:
                    new_vid = next_vector_id
                    next_vector_id += 1
                
                 with faiss_lock:
                    index.add_with_ids(track.feat, np.array([new_vid], dtype=np.int64))
                    vector_id_to_person_id[new_vid] = track.display_id
                    if track.display_id not in person_vector_ids:
                        person_vector_ids[track.display_id] = deque()
                    person_vector_ids[track.display_id].append(new_vid)
                    
                    if len(person_vector_ids[track.display_id]) > MAX_VECTORS_PER_ID:
                        old_vid = person_vector_ids[track.display_id].popleft()
                        index.remove_ids(np.array([old_vid], dtype=np.int64))
                        if old_vid in vector_id_to_person_id:
                            del vector_id_to_person_id[old_vid]

            # Draw
            x1, y1, x2, y2 = track.box
            
            # Get Resolved Face Name
            matched_face_name = final_track_faces.get(idx)
            
            if matched_face_name:
                with person_id_lock:
                    person_id_to_face_name[track.display_id] = matched_face_name
            
            with person_id_lock:
                known_face_name = person_id_to_face_name.get(track.display_id)
            
            # Calculate and print bbox center point
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            mnv_info = matched_face_name if matched_face_name else (known_face_name if known_face_name else "Unknown")
            # Log to file instead of print
            data_logger.info(f"({cam_id_str}, {track.display_id}, {mnv_info}, {bbox_center})")
            
            color = (0, 255, 0)
            label = f"ID:{track.display_id} ({track.current_sim_score:.2f})"
            
            if matched_face_name:
                label = f"{matched_face_name} | {label}"
                color = (0, 255, 255)
                log_face_detection(matched_face_name, cam_name, log_dir, person_id=track.display_id)
            elif known_face_name:
                label = f"{known_face_name} | {label}"
                color = (0, 200, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Collect Unmatched Detections for ReID Search (New Candidates)
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                unmatched_detections.append(det)

        confirmed_tracks[cam_idx] = active_confirmed

        # ReID Phase 1: Identify Known People (For Unmatched Detections)
        unknown_detections = []
        
        for det in unmatched_detections:
            feat = det['feat']
            box = det['box']
            
            # 1. Try to match with Face Result FIRST
            matched_face_name = None
            matched_face_score = 0.0
            
            for face in face_results:
                if is_face_inside_body(face['box'], box):
                    if face['name'] != "Unknown":
                        matched_face_name = face['name']
                        matched_face_score = face['sim_score']
                        break
            
            # 2. Try to match with ReID (Body) - with lock + Adaptive Top-K Voting
            found_person_id = -1
            sim_score = 0.0
            
            with faiss_lock:
                if index is not None and index.ntotal > 0:
                    # Adaptive K: Search top K nearest vectors (adjust based on index size)
                    k = min(TOP_K_VOTING, index.ntotal)  # Don't search more than available
                    D, I = index.search(feat, k)
                    
                    # Collect votes: person_id -> (count, max_similarity)
                    votes = {}
                    for i in range(k):
                        similarity = float(D[0][i])
                        if similarity > MATCH_THRESH:
                            vec_id = int(I[0][i])
                            if vec_id in vector_id_to_person_id:
                                pid = vector_id_to_person_id[vec_id]
                                if pid not in votes:
                                    votes[pid] = {'count': 0, 'max_sim': 0.0}
                                votes[pid]['count'] += 1
                                votes[pid]['max_sim'] = max(votes[pid]['max_sim'], similarity)
                    
                    # Find winner by adaptive voting
                    if votes:
                        # Sort by vote count (descending), then by max similarity (descending)
                        sorted_votes = sorted(votes.items(), 
                                            key=lambda x: (x[1]['count'], x[1]['max_sim']), 
                                            reverse=True)
                        winner_pid, winner_data = sorted_votes[0]
                        
                        # Adaptive voting threshold based on k:
                        # k=1: require 1 vote (100%)
                        # k=2: require 1 vote (50%+)  
                        # k=3: require 2 votes (66%+)
                        required_votes = max(1, (k + 1) // 2)  # Ceiling of k/2
                        
                        if winner_data['count'] >= required_votes:
                            found_person_id = winner_pid
                            sim_score = winner_data['max_sim']
            
            if found_person_id != -1 and found_person_id in assigned_ids:
                # If this ID is already assigned to another detection in this frame (rare but possible),
                # we treat this as unknown (or could handle duplicate ID logic)
                found_person_id = -1
            
            if found_person_id != -1:
                # It's a known ReID person (Re-entry or First Detection)
                assigned_ids.add(found_person_id)
                resolved_boxes.append(box)
                
                # Create NEW Confirmed Track
                new_track = ConfirmedTrack(found_person_id, box, feat)
                confirmed_tracks[cam_idx].append(new_track)
                
                # Update mapping if face detected
                if matched_face_name:
                    with person_id_lock:
                        person_id_to_face_name[found_person_id] = matched_face_name
                
                # Update Vectors (with lock)
                if sim_score < 0.95 and det['is_sharp']:
                    with vector_id_lock:
                        new_vid = next_vector_id
                        next_vector_id += 1
                    
                    with faiss_lock:
                        index.add_with_ids(feat, np.array([new_vid], dtype=np.int64))
                        vector_id_to_person_id[new_vid] = found_person_id
                        if found_person_id not in person_vector_ids:
                            person_vector_ids[found_person_id] = deque()
                        person_vector_ids[found_person_id].append(new_vid)
                        
                        if len(person_vector_ids[found_person_id]) > MAX_VECTORS_PER_ID:
                            old_vid = person_vector_ids[found_person_id].popleft()
                            index.remove_ids(np.array([old_vid], dtype=np.int64))
                            if old_vid in vector_id_to_person_id:
                                del vector_id_to_person_id[old_vid]

                # --- DRAWING ---
                x1, y1, x2, y2 = box
                color = (0, 255, 0) # Green for known ReID
                
                # Label Logic - Check historical face mapping
                with person_id_lock:
                    known_face_name = person_id_to_face_name.get(found_person_id)
                
                label = f"ID:{found_person_id} ({sim_score:.2f})"
                if matched_face_name:
                    # Face detected now
                    label = f"{matched_face_name} | {label}"
                    color = (0, 255, 255) # Yellow/Cyan if face matched
                    
                    # Log face detection with ReID ID
                    log_face_detection(matched_face_name, cam_name, log_dir, person_id=found_person_id)
                elif known_face_name:
                    # Face not detected now, but known from history
                    label = f"{known_face_name} | {label}"
                    color = (0, 200, 0) # Slightly darker green
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
            else:
                # Unknown to ReID
                det['matched_face'] = matched_face_name
                unknown_detections.append(det)

        # ReID Phase 2: Process Unknowns
        current_pending = []
        
        active_tracks = []
        for track in pending_tracks[cam_idx]:
            is_resolved = False
            for rbox in resolved_boxes:
                if compute_iou(track.box, rbox) > 0.5:
                    is_resolved = True
                    break
            if not is_resolved:
                active_tracks.append(track)
        
        for det in unknown_detections:
            box = det['box']
            feat = det['feat']
            matched_face_name = det.get('matched_face')
            
            best_iou = 0
            best_track = None
            
            for track in active_tracks:
                iou = compute_iou(track.box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            
            if best_iou > 0.5:
                best_track.box = box
                best_track.feat = feat
                best_track.features.append(feat)  # Accumulate all features
                best_track.count += 1
                
                if best_track.count >= CONFIRM_FRAMES:
                    # New ReID Identity (with locks)
                    with person_id_lock:
                        new_id = next_person_id
                        next_person_id += 1
                    
                    # Add ALL accumulated features from pending track (batch add)
                    num_feats = len(best_track.features)
                    with vector_id_lock:
                        start_vid = next_vector_id
                        next_vector_id += num_feats
                        new_vids = list(range(start_vid, start_vid + num_feats))
                    
                    # Stack all features for batch insertion
                    all_feats = np.vstack(best_track.features)
                    
                    with faiss_lock:
                        index.add_with_ids(all_feats, np.array(new_vids, dtype=np.int64))
                        if new_id not in person_vector_ids:
                            person_vector_ids[new_id] = deque()
                        for vid in new_vids:
                            vector_id_to_person_id[vid] = new_id
                            person_vector_ids[new_id].append(vid)
                    
                    # Save face mapping if available
                    if matched_face_name:
                        with person_id_lock:
                            person_id_to_face_name[new_id] = matched_face_name
                    
                    # Draw
                    x1, y1, x2, y2 = box
                    color = (0, 0, 255) # Red for new
                    label = f"ID:{new_id} (New)"

                    # Calculate and print bbox center point
                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    mnv_info = matched_face_name if matched_face_name else "Unknown"
                    # Log to file instead of print
                    data_logger.info(f"({cam_id_str}, {new_id}, {mnv_info}, {bbox_center})")
                    
                    if matched_face_name:
                        label = f"{matched_face_name} | {label}"
                        color = (0, 255, 255)
                        
                        log_face_detection(matched_face_name, cam_name, log_dir, person_id=new_id)

                    # Create Confirmed Track for new ID
                    new_track = ConfirmedTrack(new_id, box, feat)
                    confirmed_tracks[cam_idx].append(new_track)


                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    active_tracks.remove(best_track) 
                else:
                    current_pending.append(best_track)
                    active_tracks.remove(best_track)
                    
                    # Draw Identifying
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Identifying... {best_track.count}/{CONFIRM_FRAMES}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                new_track = PendingTrack(box, feat)
                current_pending.append(new_track)
                
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Identifying... 1/{CONFIRM_FRAMES}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            pending_tracks[cam_idx] = current_pending

        # --- STEP C: Draw ALL Face Boxes (after ReID processing) ---
        for face in face_results:
            fx1, fy1, fx2, fy2 = face['box']
            face_name = face['name']
            face_sim = face['sim_score']
            
            if face_name == "Unknown":
                # Màu Đỏ cho Unknown face
                face_color = (0, 0, 255)  # Red (BGR)
                face_label = "Unknown"
            else:
                # Màu Xanh cho Known face
                face_color = (0, 255, 0)  # Green (BGR)
                face_label = f"{face_name} ({face_sim:.2f})"
            
            # Vẽ face bbox
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            
            # Vẽ label cho face
            label_y = fy1 - 5 if fy1 > 20 else fy2 + 20
            cv2.putText(frame, face_label, (fx1, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        
        return frame
    
    print("\n🚀 Starting Dual-Recognition System (Face + Body ReID) - Multi-threaded")
    print("Press 'q' to exit.")
    
    # Create thread pool for parallel processing
    max_workers = len(streams)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    shared_state = {
        'faiss_lock': faiss_lock,
        'person_id_lock': person_id_lock,
        'vector_id_lock': vector_id_lock,
        'cam_id_to_floor': cam_id_to_floor,
        'stream_to_cam_id': stream_to_cam_id,
        'stream_to_cam_info': stream_to_cam_info  # Full camera info for logging
    }
    
    # Start API Server (WebSocket)
    try:
        api_thread = threading.Thread(target=api_server.start_api_server, kwargs={'host': '0.0.0.0', 'port': 8068})
        api_thread.daemon = True
        api_thread.start()
        print("✅ WebSocket API Server started on port 8068")
    except Exception as e:
        print(f"❌ Failed to start API Server: {e}")

    while True:
        # Check for face directory updates
        if not face_update_queue.empty():
            try:
                signal = face_update_queue.get_nowait()
                if signal == 'reload_faces':
                    print("\n🔄 Reloading face embeddings...")
                    
                    # Rebuild face targets and FAISS index
                    new_targets, new_index, new_id_to_name = rebuild_face_index(
                        face_detector, face_recognizer, faces_dir_path
                    )
                    
                    # Thread-safe update
                    with face_resources_lock:
                        face_targets = new_targets
                        face_index = new_index
                        face_id_to_name = new_id_to_name
                    
                    print("✅ Face embeddings reloaded successfully!\n")
            except queue.Empty:
                pass
        
        # Read frames from all cameras
        frames = []
        for i, stream in enumerate(streams):
            ret, frame = stream.read()
            if not ret or frame is None:
                frames.append(None)
                continue
            frames.append(frame.copy())

        if all(f is None for f in frames):
            time.sleep(0.01)
            continue

        # Process cameras in parallel using ThreadPoolExecutor
        futures = {}
        for cam_idx, frame in enumerate(frames):
            if frame is None:
                continue
            # Submit task to thread pool
            future = executor.submit(process_camera_frame, cam_idx, frame, shared_state)
            futures[future] = cam_idx
        
        # Collect results (No imshow anymore)
        for future in as_completed(futures):
            cam_idx = futures[future]
            try:
                processed_frame = future.result()
                # HEADLESS MODE: No cv2.imshow needed
                # display_frame = cv2.resize(processed_frame, (960, 540)) 
                # cv2.imshow(f"Cam {cam_idx+1}", display_frame)
            except Exception as e:
                print(f"Error processing camera {cam_idx+1}: {e}")
                import traceback
                traceback.print_exc()

        # --- Update Live Maps & Broadcast WebSocket ---
        camera_points_f3 = {} # Floor 3
        camera_points_f1 = {} # Floor 1
        
        # Determine tracking points
        if mapper3 or mapper1:
            for cam_idx, tracks in confirmed_tracks.items():
                cam_idx_int = int(cam_idx) # Ensure int
                cam_id = stream_to_cam_id.get(cam_idx_int)
                if not cam_id: continue
                
                points = []
                for track in tracks:
                    if track.miss_count > 0: continue
                    
                    # Get Face Name
                    with person_id_lock:
                        face_name = person_id_to_face_name.get(track.display_id, "Unknown")
                    
                    # Get BBox Center
                    x1, y1, x2, y2 = track.box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # map.py expects: (pid, u, v)
                    points.append((track.display_id, center[0], center[1]))
                
                if points:
                    # Use stored floor info
                    floor = cam_id_to_floor.get(cam_id, 3) # Default to 3 if unknown
                    if floor == 1:
                        camera_points_f1[cam_id] = points
                    elif floor == 3:
                        camera_points_f3[cam_id] = points

        # Update Floor 3 Data
        if mapper3:
            try:
                # 1. Clear & Project
                mapper3.projected = []
                mapper3.image_to_map(camera_points_f3)
                
                # 2. Merge
                merged_points = mapper3.merge_points(distance_mm=130)
                
                # 3. Check ROIs
                roi_status = mapper3.find_points_in_roi(merged_points)
                
                # 4. Construct Payload
                payload = {
                    "points": merged_points,
                    "rois": roi_status,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 5. Broadcast via WebSocket
                api_server.send_update(3, payload)
                
                # --- MCT DB LOGGING: Position Tracking (Floor 3) ---
                if MCT_TRACKING_AVAILABLE:
                    tracker = get_mct_tracker()
                    for p in merged_points:
                        # merged_point structure from map.py:
                        # {'point_id': pid, 'map_px': (u, v), 'world_mm': (wx, wy), 'face_name': name, 'cameras': [...]}
                        
                        pid = p.get('point_id')
                        wx, wy = p.get('world_mm', (0, 0))
                        
                        # Resolve usr_id
                        usr_id = 'unknown'
                        face_name = p.get('face_name')
                        if face_name and face_name != 'Unknown':
                            usr_id = face_name
                        else:
                            # Try lookup in person_id_to_face_name
                            with person_id_lock:
                                known = person_id_to_face_name.get(pid)
                                if known: usr_id = known
                        
                        # Save Position
                        tracker.save_position(
                            local_track_id=pid,
                            usr_id=usr_id,
                            floor="3F",
                            x=float(wx),
                            y=float(wy),
                            # merged points are combined from multiple cameras, so camera_id is ambiguous
                            # we can leave it None or pick first
                            camera_id=p.get('cameras', [None])[0],
                            bbox_center=p.get('map_px') # Saving map pixel as center for now
                        )
                        
                        # Retroactively update usr_id for this track if identified
                        if usr_id != 'unknown':
                            tracker.update_track_usr_id(pid, usr_id)

                
            except Exception as e:
                print(f"❌ Map Floor 3 Update Error: {e}")
                traceback.print_exc() # Print full stack trace
             
        # Update Floor 1 Data
        if mapper1:
            try:
                # 1. Clear & Project
                mapper1.projected = []
                mapper1.image_to_map(camera_points_f1)
                
                # 2. Merge
                merged_points = mapper1.merge_points(distance_mm=130)
                
                # 3. Check ROIs
                roi_status = mapper1.find_points_in_roi(merged_points)
                
                # 4. Construct Payload
                payload = {
                    "points": merged_points,
                    "rois": roi_status,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 5. Broadcast via WebSocket
                api_server.send_update(1, payload)

                # --- MCT DB LOGGING: Position Tracking (Floor 1) ---
                if MCT_TRACKING_AVAILABLE:
                    tracker = get_mct_tracker()
                    for p in merged_points:
                        pid = p.get('point_id')
                        wx, wy = p.get('world_mm', (0, 0))
                        
                        # Resolve usr_id
                        usr_id = 'unknown'
                        face_name = p.get('face_name')
                        if face_name and face_name != 'Unknown':
                            usr_id = face_name
                        else:
                            with person_id_lock:
                                known = person_id_to_face_name.get(pid)
                                if known: usr_id = known
                        
                        # Save Position
                        tracker.save_position(
                            local_track_id=pid,
                            usr_id=usr_id,
                            floor="1F",
                            x=float(wx),
                            y=float(wy),
                            camera_id=p.get('cameras', [None])[0],
                            bbox_center=p.get('map_px')
                        )
                        
                        if usr_id != 'unknown':
                            tracker.update_track_usr_id(pid, usr_id)
                
            except Exception as e:
                print(f"❌ Map Floor 1 Update Error: {e}")
                traceback.print_exc()

        # Reduce CPU usage slightly since we are not throttled by waitKey(1)
        # But we want high FPS, so maybe sync with camera FPS?
        # A small sleep is good to valid busy loop if empty
        time.sleep(0.03) # Cap at approx 30 FPS to prevent OOM/CPU spike
        
        # HEADLESS: No 'q' key check possible
        # We can listen to a signal or just run forever until Ctrl+C
    
    # Cleanup (Unreachable in infinite loop usually, but good to have)
    if MCT_TRACKING_AVAILABLE:
        get_mct_tracker().end_session()
        
    executor.shutdown(wait=True)

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Camera Tracking System with Face + Body ReID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load cameras from database (recommended):
  python demo_mct.py --use_db
  
  # Load only specific floors from database:
  python demo_mct.py --use_db --floors "3F,1F"
  
  # Use manual RTSP URLs (legacy mode):
  python demo_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""
    )
    
    # TransReID Configuration
    parser.add_argument("--config_file", default="configs/Market/vit_transreid_stride.yml", 
                        help="path to TransReID config")
    parser.add_argument("--weights", default="weights/vit_transreid_market.pth", 
                        help="path to TransReID weights")
    
    # ==========================================================================
    # DATABASE MODE (NEW - Recommended)
    # ==========================================================================
    parser.add_argument("--use_db", action="store_true", 
                        help="Load camera configuration from PostgreSQL database")
    parser.add_argument("--floors", type=str, default=None,
                        help="Comma-separated floor names to load (e.g., '3F,1F'). Only used with --use_db")
    
    # Database connection (can be overridden, defaults are in db_config.py)
    parser.add_argument("--db_host", type=str, default=None,
                        help="Database host (default: 192.168.210.250)")
    parser.add_argument("--db_port", type=int, default=None,
                        help="Database port (default: 5432)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name (default: camera_ai_db)")
    
    # ==========================================================================
    # MANUAL MODE (Legacy - for backward compatibility)
    # ==========================================================================
    parser.add_argument("--rtsp1", default="rtsp://developer:Inf2026T1@10.29.98.60:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 1")
    parser.add_argument("--rtsp2", default="rtsp://developer:Inf2026T1@10.29.98.58:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 2")
    parser.add_argument("--rtsp3", default="rtsp://developer:Inf2026T1@10.29.98.57:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 3")
    parser.add_argument("--rtsp4", default="rtsp://developer:Inf2026T1@10.29.98.59:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 4")
    parser.add_argument("--rtsp1T1", default="rtsp://developer:Inf2026T1@10.29.98.52:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL for cam1T1")
    
    # Camera Floor Configuration (manual mode)
    parser.add_argument("--floor_cam1", type=int, default=3, help="Floor number for cam1 (default: 3)")
    parser.add_argument("--floor_cam2", type=int, default=3, help="Floor number for cam2 (default: 3)")
    parser.add_argument("--floor_cam3", type=int, default=3, help="Floor number for cam3 (default: 3)")
    parser.add_argument("--floor_cam4", type=int, default=3, help="Floor number for cam4 (default: 3)")
    parser.add_argument("--floor_cam1T1", type=int, default=1, help="Floor number for cam1T1 (default: 1)")
    
    args = parser.parse_args()
    
    # Set environment variables for database connection if provided
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = str(args.db_port)
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    
    run_demo(args)

