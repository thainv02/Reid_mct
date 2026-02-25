#!/usr/bin/env python3
"""
Multi-Camera Tracking (MCT) System - Main Entry Point

This is the refactored entry point for the MCT system.
It uses modular components from the mct package.

Usage:
    # Load cameras from database (recommended):
    python main_mct.py --use_db
    
    # Load only specific floors from database:
    python main_mct.py --use_db --floors "3F,1F"
    
    # Use manual RTSP URLs (legacy mode):
    python main_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""

import os
import sys
import argparse

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# Demo Mode: Speed over Reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image2map'))

# Import MCT modules (new modular structure)
from mct.core.stream import ThreadedStream
from mct.core.tracker import PendingTrack, ConfirmedTrack
from mct.face.detector import setup_face_api, run_face_recognition
from mct.face.indexer import rebuild_face_index
from mct.reid.model import setup_transreid, get_transforms
from mct.reid.features import extract_feature, extract_features_batch
from mct.utils.time_utils import get_vn_time
from mct.utils.file_utils import load_json_file, save_json_file, monitor_face_directory
from mct.utils.geometry import compute_iou, is_face_inside_body
from mct.utils.logging_utils import get_floor_logger, log_face_detection, remove_expired_names

# Other imports
import api_server
from map import Map
from config import cfg

# Database imports
try:
    from database.db_config import load_all_config, generate_active_cameras_list
    DB_CONFIG_AVAILABLE = True
    print("✅ Database config module available")
except ImportError:
    DB_CONFIG_AVAILABLE = False
    print("⚠️ Database config module not available")

try:
    from database.mct_tracking import get_mct_tracker
    MCT_TRACKING_AVAILABLE = True
    print("✅ MCT Tracking module available")
except ImportError as e:
    MCT_TRACKING_AVAILABLE = False
    print(f"⚠️ MCT Tracking module not available: {e}")


def main():
    """Main entry point - delegates to demo_mct.run_demo for now."""
    parser = argparse.ArgumentParser(
        description="Multi-Camera Tracking System with Face + Body ReID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load cameras from database (recommended):
  python main_mct.py --use_db
  
  # Load only specific floors from database:
  python main_mct.py --use_db --floors "3F,1F"
  
  # Use manual RTSP URLs (legacy mode):
  python main_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""
    )
    
    # TransReID Configuration
    parser.add_argument("--config_file", default="configs/Market/vit_transreid_stride.yml", 
                        help="path to TransReID config")
    parser.add_argument("--weights", default="weights/transformer_120.pth", 
                        help="path to TransReID weights")
    
    # Database mode
    parser.add_argument("--use_db", action="store_true",
                        help="Load camera configuration from database instead of command-line args")
    parser.add_argument("--floors", type=str, default=None,
                        help="Comma-separated list of floors to load (e.g., '3F,1F'). Only used with --use_db")
    
    # Database connection (optional overrides)
    parser.add_argument("--db_host", type=str, default=None,
                        help="Database host (default: from environment/localhost)")
    parser.add_argument("--db_port", type=int, default=None,
                        help="Database port (default: from environment/5432)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name (default: camera_ai_db)")
    
    # Manual mode (legacy)
    parser.add_argument("--rtsp1", default="", help="RTSP URL 1")
    parser.add_argument("--rtsp2", default="", help="RTSP URL 2")
    parser.add_argument("--rtsp3", default="", help="RTSP URL 3")
    parser.add_argument("--rtsp4", default="", help="RTSP URL 4")
    parser.add_argument("--rtsp1T1", default="", help="RTSP URL for cam1T1 (Floor 1)")
    parser.add_argument("--floor_cam1", type=int, default=3, help="Floor number for cam1")
    parser.add_argument("--floor_cam2", type=int, default=3, help="Floor number for cam2")
    parser.add_argument("--floor_cam3", type=int, default=3, help="Floor number for cam3")
    parser.add_argument("--floor_cam4", type=int, default=3, help="Floor number for cam4")
    parser.add_argument("--floor_cam1T1", type=int, default=1, help="Floor number for cam1T1")
    
    args = parser.parse_args()
    
    # Set environment variables for database connection if provided
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = str(args.db_port)
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    
    # Import and run the main demo logic
    # (For now, we still use demo_mct.run_demo as the core processing loop is complex)
    print("="*60)
    print("🚀 MCT System - Modular Architecture")
    print("="*60)
    print("ℹ️  Using new modular structure from mct/ package")
    print("ℹ️  Components loaded:")
    print("   - mct.core.stream: ThreadedStream")
    print("   - mct.core.tracker: PendingTrack, ConfirmedTrack")
    print("   - mct.face.detector: setup_face_api, run_face_recognition")
    print("   - mct.face.indexer: rebuild_face_index")
    print("   - mct.reid.model: setup_transreid, get_transforms")
    print("   - mct.reid.features: extract_feature, extract_features_batch")
    print("   - mct.utils.*: Various utilities")
    print("="*60)
    
    # For backward compatibility, import and run demo_mct.run_demo
    from demo_mct import run_demo
    run_demo(args)


if __name__ == "__main__":
    main()
