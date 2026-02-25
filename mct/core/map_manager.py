"""
Map update and WebSocket broadcast logic for MCT system.
Handles projecting tracking points to floor maps and broadcasting via WebSocket.
"""
import traceback
from datetime import datetime
from ..core.display_manager import update_map_display


def update_maps_and_broadcast(mappers, confirmed_tracks, stream_to_cam_id,
                              cam_id_to_floor, reid_index):
    """
    Update floor maps with tracking data and broadcast via WebSocket.
    Also logs positions to MCT database.
    
    Args:
        mappers: Dict of {floor_num: Map}
        confirmed_tracks: Dict of {cam_idx: [ConfirmedTrack, ...]}
        stream_to_cam_id: Dict of {stream_idx: cam_id}
        cam_id_to_floor: Dict of {cam_id: floor_num}
        reid_index: ReIDIndex instance for face name lookup
    """
    import api_server
    
    if not mappers:
        return
    
    # Collect tracking points grouped by floor
    camera_points_by_floor = {floor_num: {} for floor_num in mappers}
    
    for cam_idx, tracks in confirmed_tracks.items():
        cam_idx_int = int(cam_idx)
        cam_id = stream_to_cam_id.get(cam_idx_int)
        if not cam_id:
            continue
        
        # Skip disabled cameras
        if not api_server.is_camera_enabled(cam_id):
            continue
        
        points = []
        for track in tracks:
            if track.miss_count > 0:
                continue
            
            face_name = reid_index.get_face_name(track.display_id) or "Unknown"
            
            x1, y1, x2, y2 = track.box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # map.py expects: (pid, u, v)
            points.append((track.display_id, center[0], center[1]))
        
        if points:
            floor = cam_id_to_floor.get(cam_id, 3)
            if floor in camera_points_by_floor:
                camera_points_by_floor[floor][cam_id] = points

    # Update each floor's map and broadcast
    for floor_num, mapper in mappers.items():
        # Skip disabled floors
        if not api_server.is_floor_enabled(floor_num):
            continue
        
        camera_points = camera_points_by_floor.get(floor_num, {})
        
        try:
            # 1. Clear & Project
            mapper.projected = []
            mapper.image_to_map(camera_points)
            
            # 2. Merge
            merged_points = mapper.merge_points(distance_mm=130)
            
            # 3. Check ROIs
            roi_status = mapper.find_points_in_roi(merged_points)
            
            # 4. Construct Payload
            payload = {
                "points": merged_points,
                "rois": roi_status,
                "timestamp": datetime.now().isoformat()
            }
            
            # 5. Broadcast via WebSocket
            api_server.send_update(floor_num, payload)
            
            # 6. Display map image on screen
            mapper._load_map()
            map_img = mapper.map_img_original.copy()
            map_img = mapper.draw_points_and_rois(map_img, merged_points)
            update_map_display(floor_num, map_img)
            
            # 7. MCT DB Logging
            _log_positions_to_db(floor_num, merged_points, reid_index)
            
        except Exception as e:
            print(f"❌ Map Floor {floor_num} Update Error: {e}")
            traceback.print_exc()


def _log_positions_to_db(floor_num, merged_points, reid_index):
    """
    Log merged tracking positions to MCT database.
    
    Args:
        floor_num: Floor number
        merged_points: List of merged point dicts from map
        reid_index: ReIDIndex instance for face name lookup
    """
    try:
        from database.mct_tracking import get_mct_tracker
    except ImportError:
        return
    
    try:
        tracker = get_mct_tracker()
        floor_name = f"{floor_num}F"
        
        for p in merged_points:
            pid = p.get('point_id')
            wx, wy = p.get('world_mm', (0, 0))
            
            # Resolve usr_id
            usr_id = 'unknown'
            face_name = p.get('face_name')
            if face_name and face_name != 'Unknown':
                usr_id = face_name
            else:
                known = reid_index.get_face_name(pid)
                if known:
                    usr_id = known
            
            # Save Position
            tracker.save_position(
                local_track_id=pid,
                usr_id=usr_id,
                floor=floor_name,
                x=float(wx),
                y=float(wy),
                camera_id=p.get('cameras', [None])[0],
                bbox_center=p.get('map_px')
            )
            
            # Retroactively update usr_id if identified
            if usr_id != 'unknown':
                tracker.update_track_usr_id(pid, usr_id)
    except Exception as e:
        print(f"Warning: Failed to log positions to DB: {e}")
