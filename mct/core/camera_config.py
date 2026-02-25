"""
Camera configuration loading and map initialization for MCT system.
Handles both database mode and CLI argument mode.
"""
import os


# Floor configuration: {floor_num: (folder_name, map_image_file)}
FLOOR_MAP_CONFIG = {
    1: ("F1", "1f.png"),
    2: ("F2", "2f.png"),
    3: ("F3", "3f.png"),
    4: ("F4", "4f.png"),
    5: ("F5", "5f.png"),
    6: ("F6", "6f.png"),
    7: ("F7", "7f.png"),
}


def initialize_maps(map_dir="image2map"):
    """
    Initialize Map objects for all available floors.
    
    Args:
        map_dir: Base directory for map data (default: 'image2map')
    
    Returns:
        dict: {floor_num: Map} for successfully loaded floors
    """
    # Import here to avoid circular imports (map.py is in image2map/)
    from map import Map
    
    print("Initializing Map Visualization for all floors...")
    mappers = {}
    
    for floor_num, (folder, img_file) in FLOOR_MAP_CONFIG.items():
        map_img_path = os.path.join(map_dir, folder, img_file)
        if not os.path.exists(map_img_path):
            print(f"⚠️ Map image not found for Floor {floor_num}: {map_img_path}")
            continue
        try:
            mapper = Map(
                map_image_path=map_img_path,
                mm_per_pixel_x=23.8,
                mm_per_pixel_y=23.3
            )
            # Load ROIs if available
            rois_path = os.path.join(map_dir, folder, "rois.yaml")
            if os.path.exists(rois_path):
                mapper.load_rois_from_yaml(rois_path)
            mappers[floor_num] = mapper
            print(f"✅ Map Floor {floor_num} initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize Map Floor {floor_num}: {e}")
    
    print(f"📊 Total mappers initialized: {len(mappers)} / {len(FLOOR_MAP_CONFIG)}")
    return mappers


def load_cameras_from_db(args):
    """
    Load camera configuration from PostgreSQL database.
    
    Args:
        args: Parsed argparse Namespace with use_db, floors, etc.
    
    Returns:
        list: Active camera dicts [{'id', 'url', 'floor', 'name', 'ip', 'inout'}, ...]
              or empty list on failure
    """
    try:
        from database.db_config import load_all_config, generate_active_cameras_list
    except ImportError:
        print("⚠️ Database config module not available")
        return []
    
    print("\n" + "="*60)
    print("🗄️  Loading camera configuration from DATABASE")
    print("="*60)
    
    floor_filter = None
    if hasattr(args, 'floors') and args.floors:
        floor_filter = [f.strip() for f in args.floors.split(',')]
        print(f"📌 Filtering floors: {floor_filter}")
    
    try:
        db_config = load_all_config(floor_filter=floor_filter)
        active_cameras = generate_active_cameras_list(db_config['cameras'])
        print(f"✅ Loaded {len(active_cameras)} cameras from database")
        
        for cam in active_cameras:
            print(f"   - {cam['name']} ({cam['id']}) -> Floor {cam['floor']} [{cam['inout']}]")
        
        return active_cameras
    except Exception as e:
        print(f"❌ Failed to load from database: {e}")
        print("⚠️ Falling back to command-line arguments...")
        import traceback
        traceback.print_exc()
        return []


def load_cameras_from_args(args):
    """
    Load camera configuration from command-line arguments (legacy mode).
    
    Args:
        args: Parsed argparse Namespace
    
    Returns:
        list: Active camera dicts
    """
    print("\n📋 Using command-line arguments for camera configuration")
    active_cameras = []
    
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
    
    return active_cameras


def register_cameras_to_maps(active_cameras, mappers, map_dir="image2map"):
    """
    Register cameras to their floor maps using calibration files.
    
    Args:
        active_cameras: List of camera dicts
        mappers: Dict of {floor_num: Map}
        map_dir: Base directory for calibration data
    
    Returns:
        dict: cam_id -> floor_num mapping
    """
    cam_id_to_floor = {}
    
    for cam in active_cameras:
        cam_id = cam['id']  # IP address (e.g., '10.29.98.52')
        floor = cam['floor']
        cam_id_to_floor[cam_id] = floor
        
        floor_folder = f"F{floor}"
        
        # cam_id is IP address, which matches calibration folder names
        # e.g., image2map/F3/10.29.98.58/intrinsic.yaml
        calib_folder = cam_id
        intrinsic_path = os.path.join(map_dir, floor_folder, f"{calib_folder}/intrinsic.yaml")
        extrinsic_path = os.path.join(map_dir, floor_folder, f"{calib_folder}/extrinsic.yaml")
        
        cam_name = cam.get('name', cam_id)
        
        if not os.path.exists(intrinsic_path) or not os.path.exists(extrinsic_path):
            print(f"⚠️ Calibration not found for {cam_id} ({cam_name}) on Floor {floor}")
            print(f"   Expected: {intrinsic_path}")
            print(f"   Skipping camera-to-map registration (tracking will still work)")
            continue
        
        if floor in mappers:
            mappers[floor].add_camera(
                camera_id=cam_id,
                intrinsic=intrinsic_path,
                extrinsic=extrinsic_path
            )
            print(f"   📷 Registered {cam_id} ({cam_name}) to Map Floor {floor}")
        else:
            print(f"⚠️ Warning: {cam_id} assigned to Floor {floor} but Map Floor {floor} not initialized.")
    
    return cam_id_to_floor
