"""
MCT Engine — Main demo loop for Multi-Camera Tracking.
Orchestrates model initialization, camera streams, frame processing,
map updates, and WebSocket broadcasting.
"""
import os
import sys
import time
import queue
import threading
import traceback

import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.stream import ThreadedStream
from ..core.frame_processor import FrameProcessor
from ..core.camera_config import (
    initialize_maps,
    load_cameras_from_db,
    load_cameras_from_args,
    register_cameras_to_maps,
)
from ..core.map_manager import update_maps_and_broadcast
from ..core.display_manager import show_camera_frame, close_floor_display, close_all_displays
from ..reid.model import setup_transreid, get_transforms
from ..reid.reid_index import ReIDIndex
from ..face.detector import setup_face_api
from ..face.indexer import rebuild_face_index
from ..utils.file_utils import monitor_face_directory


def run_demo(args):
    """
    Main entry point for the Multi-Camera Tracking system.
    
    Args:
        args: Parsed argparse Namespace with config_file, weights,
              use_db, floors, rtsp URLs, etc.
    """
    import api_server
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- MCT DB LOGGING: Start Session ---
    mct_tracker = None
    try:
        from database.mct_tracking import get_mct_tracker
        mct_tracker = get_mct_tracker()
        session_id = mct_tracker.start_session()
        print(f"🚀 MCT Session Started: {session_id}")
    except ImportError:
        print("⚠️ MCT Tracking module not available")
    except Exception as e:
        print(f"⚠️ MCT Session start failed: {e}")

    # 1. Initialize YOLO (Person Detection)
    from ultralytics import YOLO
    print("Loading YOLO11x...")
    yolo = YOLO('yolo11x.pt')
    yolo.to(device)
    
    # Warmup: prevent multi-threading AttributeError
    _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo.predict(_dummy, verbose=False, device=device)
    print("✅ YOLO warmup complete")

    # 2. Initialize TransReID
    print("Loading TransReID...")
    reid_model = setup_transreid(args.config_file, args.weights, device)
    reid_transform = get_transforms()

    # 3. Initialize Face API with FAISS index
    face_detector, face_recognizer, face_targets, face_index, face_id_to_name = setup_face_api()
    
    # 3.1 Setup Face Directory Monitoring
    faces_dir_path = os.path.abspath("./API_Face/faces/")
    face_update_queue = queue.Queue()
    face_resources_lock = threading.Lock()
    
    monitor_thread = threading.Thread(
        target=monitor_face_directory,
        args=(faces_dir_path, face_update_queue, 10),
        daemon=True
    )
    monitor_thread.start()

    # 3.2 Initialize Maps for ALL Floors
    mappers = initialize_maps("image2map")
        
    # 4. Load Camera Configuration
    active_cameras = []
    
    if getattr(args, 'use_db', False):
        active_cameras = load_cameras_from_db(args)
    
    if not active_cameras:
        active_cameras = load_cameras_from_args(args)
    
    # 5. Register Cameras to Maps
    cam_id_to_floor = register_cameras_to_maps(active_cameras, mappers, "image2map")

    # 6. Open RTSP Streams
    streams = []
    stream_to_cam_id = {}
    connected_count = 0
    failed_count = 0
    
    for idx, cam in enumerate(active_cameras):
        cam_id = cam['id']  # IP address
        cam_name = cam.get('name', cam_id)
        url = cam['url']
        stream = ThreadedStream(url)
        if stream.status:
            print(f"   ✅ {cam_id} ({cam_name}, Floor {cam['floor']}) — Connected")
            connected_count += 1
        else:
            print(f"   ❌ {cam_id} ({cam_name}, Floor {cam['floor']}) — FAILED to connect")
            print(f"      URL: {url}")
            failed_count += 1
        streams.append(stream)
        stream_to_cam_id[idx] = cam_id
    
    print(f"\n📊 Stream Summary: {connected_count} connected, {failed_count} failed, {len(active_cameras)} total")
    
    stream_to_cam_info = {idx: cam for idx, cam in enumerate(active_cameras)}

    # 7. Initialize ReID Index & Frame Processor
    reid_index = ReIDIndex()

    processor = FrameProcessor(
        yolo=yolo,
        reid_model=reid_model,
        reid_transform=reid_transform,
        device=device,
        face_detector=face_detector,
        face_recognizer=face_recognizer,
        face_resources_lock=face_resources_lock,
        reid_index=reid_index,
        cam_id_to_floor=cam_id_to_floor,
        stream_to_cam_id=stream_to_cam_id,
        stream_to_cam_info=stream_to_cam_info,
    )
    processor.set_face_resources(face_index, face_id_to_name)

    # 8. Start Main Loop
    print("\n🚀 Starting Dual-Recognition System (Face + Body ReID) - Multi-threaded")
    print("ℹ️  All floors are DISABLED by default. Use API to enable:")
    print("   POST http://0.0.0.0:8068/api/floors/{id}/enable")
    print("Press Ctrl+C to exit.")
    
    max_workers = len(streams)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    # Initialize floor/camera status for API management
    api_server.init_floor_camera_status(active_cameras)
    
    # Start API Server
    try:
        api_thread = threading.Thread(
            target=api_server.start_api_server,
            kwargs={'host': '0.0.0.0', 'port': 8068}
        )
        api_thread.daemon = True
        api_thread.start()
        print("✅ WebSocket API Server started on port 8068")
        print("   📡 REST API: http://0.0.0.0:8068/api/status")
        print("   📡 WebSocket: ws://0.0.0.0:8068/ws/{floor_id}")
    except Exception as e:
        print(f"❌ Failed to start API Server: {e}")

    # Track previously enabled floors to detect changes
    prev_enabled_floors = set()
    
    # Build floor -> camera info mapping for display
    floor_to_cameras = {}
    for cam in active_cameras:
        floor = cam['floor']
        if floor not in floor_to_cameras:
            floor_to_cameras[floor] = []
        floor_to_cameras[floor].append(cam)
    
    # --- MAIN PROCESSING LOOP ---
    try:
        while True:
            # Check for face directory updates
            if not face_update_queue.empty():
                try:
                    signal = face_update_queue.get_nowait()
                    if signal == 'reload_faces':
                        print("\n🔄 Reloading face embeddings...")
                        new_targets, new_index, new_id_to_name = rebuild_face_index(
                            face_detector, face_recognizer, faces_dir_path
                        )
                        with face_resources_lock:
                            face_targets = new_targets
                            face_index = new_index
                            face_id_to_name = new_id_to_name
                        processor.set_face_resources(face_index, face_id_to_name)
                        print("✅ Face embeddings reloaded successfully!\n")
                except queue.Empty:
                    pass
            
            # Detect floor enable/disable changes and close displays for disabled floors
            current_enabled_floors = set()
            for floor_num in mappers:
                if api_server.is_floor_enabled(floor_num):
                    current_enabled_floors.add(floor_num)
            
            # Close displays for newly disabled floors
            newly_disabled = prev_enabled_floors - current_enabled_floors
            for floor_num in newly_disabled:
                floor_cams = floor_to_cameras.get(floor_num, [])
                cam_display_names = []
                for c in floor_cams:
                    cam_display_names.append(c.get('name', c['id']))
                close_floor_display(floor_num, cam_display_names)
                print(f"🖥️ Closed display windows for Floor {floor_num}")
            
            prev_enabled_floors = current_enabled_floors
            
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

            # Process cameras in parallel
            futures = {}
            for cam_idx, frame in enumerate(frames):
                if frame is None:
                    continue
                cam_id = stream_to_cam_id.get(cam_idx)
                if cam_id and not api_server.is_camera_enabled(cam_id):
                    continue
                future = executor.submit(processor.process_camera_frame, cam_idx, frame)
                futures[future] = cam_idx
            
            # Collect results and display camera frames
            for future in as_completed(futures):
                cam_idx = futures[future]
                try:
                    processed_frame = future.result()
                    # Show camera feed for enabled cameras
                    if processed_frame is not None:
                        cam_id = stream_to_cam_id.get(cam_idx)
                        if cam_id:
                            floor_num = cam_id_to_floor.get(cam_id)
                            if floor_num and api_server.is_floor_enabled(floor_num):
                                cam_info = stream_to_cam_info.get(cam_idx, {})
                                cam_name = cam_info.get('name', cam_id)
                                show_camera_frame(cam_id, floor_num, processed_frame, cam_name)
                except Exception as e:
                    print(f"Error processing camera {cam_idx+1}: {e}")
                    traceback.print_exc()

            # Update Maps & Broadcast WebSocket
            update_maps_and_broadcast(
                mappers, processor.confirmed_tracks,
                stream_to_cam_id, cam_id_to_floor, reid_index
            )

            # Process OpenCV window events
            cv2.waitKey(1)
            
            time.sleep(0.03)  # Cap at ~30 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Cleanup
        if mct_tracker:
            try:
                mct_tracker.end_session()
            except Exception:
                pass
        
        executor.shutdown(wait=True)
        
        for stream in streams:
            stream.stop()
        close_all_displays()
        print("✅ Cleanup complete.")
