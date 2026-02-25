"""
FAISS indexer for face embeddings.
"""
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'API_Face'))

try:
    from API_Face.service.processing import build_targets
    FACE_API_AVAILABLE = True
except ImportError:
    FACE_API_AVAILABLE = False


def rebuild_face_index(detector, recognizer, faces_dir):
    """
    Rebuild face targets and FAISS index.
    Called when faces directory changes.
    
    Args:
        detector: SCRFD face detector
        recognizer: ArcFace recognizer
        faces_dir: Path to faces directory
    
    Returns:
        tuple: (face_targets, face_index, face_id_to_name)
            - face_targets: List of (embedding, name) tuples
            - face_index: New FAISS index
            - face_id_to_name: New ID mapping
    """
    import faiss
    
    print("🔄 Rebuilding face targets and FAISS index...")
    
    if not FACE_API_AVAILABLE:
        print("⚠️ Face API not available")
        return [], None, {}
    
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
