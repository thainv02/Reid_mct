import asyncio
import json
import threading
import traceback # Added for debug
from typing import Dict, List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Global State for latest map data (if needed for initial connection)
# Format: { floor_id (int): { "points": [...], "rois": [...] } }
LATEST_MAP_DATA = {}

# =============================================================================
# Floor & Camera Status Management (Thread-safe)
# =============================================================================

# Status dicts: {id: bool} - True = enabled, False = disabled
_floor_status: Dict[int, bool] = {}      # {1: True, 2: True, ...}
_camera_status: Dict[str, bool] = {}     # {"cam45": True, "cam41": False, ...}
_status_lock = threading.Lock()

# Camera-to-floor mapping for lookup
_camera_floor_map: Dict[str, int] = {}   # {"cam45": 1, "cam41": 2, ...}
_all_cameras_info: List[dict] = []       # Full camera info list


def init_floor_camera_status(active_cameras: List[dict]):
    """
    Initialize floor and camera status from the active cameras list.
    Called once during startup from demo_mct.py.
    
    Args:
        active_cameras: List of camera dicts with 'id', 'floor', 'name', etc.
    """
    global _all_cameras_info
    with _status_lock:
        _all_cameras_info = list(active_cameras)
        for cam in active_cameras:
            cam_id = cam['id']
            floor = cam['floor']
            _camera_status[cam_id] = False
            _camera_floor_map[cam_id] = floor
            if floor not in _floor_status:
                _floor_status[floor] = False
        
        print(f"✅ Initialized status for {len(_floor_status)} floors, {len(_camera_status)} cameras (all DISABLED by default)")


def is_floor_enabled(floor_num: int) -> bool:
    """Check if a floor is enabled. Thread-safe."""
    with _status_lock:
        return _floor_status.get(floor_num, False)


def is_camera_enabled(cam_id: str) -> bool:
    """Check if a camera is enabled. Thread-safe."""
    with _status_lock:
        # Camera must be enabled AND its floor must be enabled
        cam_ok = _camera_status.get(cam_id, False)
        if not cam_ok:
            return False
        floor = _camera_floor_map.get(cam_id)
        if floor is not None:
            return _floor_status.get(floor, False)
        return False


def get_system_status() -> dict:
    """Get full system status. Thread-safe."""
    with _status_lock:
        floors_info = {}
        for floor_num, enabled in sorted(_floor_status.items()):
            floor_cameras = []
            for cam in _all_cameras_info:
                if cam['floor'] == floor_num:
                    cam_id = cam['id']
                    floor_cameras.append({
                        'id': cam_id,
                        'name': cam.get('name', cam_id),
                        'ip': cam.get('ip', ''),
                        'enabled': _camera_status.get(cam_id, False),
                    })
            floors_info[str(floor_num)] = {
                'floor': floor_num,
                'enabled': enabled,
                'cameras': floor_cameras,
                'camera_count': len(floor_cameras),
                'active_count': sum(1 for c in floor_cameras if c['enabled']),
            }
        return {
            'total_floors': len(_floor_status),
            'total_cameras': len(_camera_status),
            'active_floors': sum(1 for v in _floor_status.values() if v),
            'active_cameras': sum(1 for v in _camera_status.values() if v),
            'floors': floors_info,
        }


class ConnectionManager:
    def __init__(self):
        # Store active connections: { floor_id: Set[WebSocket] }
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, floor_id: int):
        await websocket.accept()
        if floor_id not in self.active_connections:
            self.active_connections[floor_id] = set()
        self.active_connections[floor_id].add(websocket)
        print(f"🔌 Client connected to Floor {floor_id}. Total: {len(self.active_connections[floor_id])}")
        
        # Optionally send the last known state immediately upon connection
        if floor_id in LATEST_MAP_DATA:
            try:
                await websocket.send_json(LATEST_MAP_DATA[floor_id])
            except:
                pass

    def disconnect(self, websocket: WebSocket, floor_id: int):
        if floor_id in self.active_connections:
            self.active_connections[floor_id].discard(websocket)
            print(f"🔌 Client disconnected from Floor {floor_id}")

    async def broadcast(self, floor_id: int, message: dict):
        """
        Send a JSON message to all clients connected to a specific floor.
        """
        if floor_id not in self.active_connections:
            return

        # Update global state (backup)
        LATEST_MAP_DATA[floor_id] = message

        # Broadcast to all connected clients
        # Copy set to avoid size change issues during iteration
        connections = self.active_connections[floor_id].copy()
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection, floor_id)
            except Exception as e:
                # Handle broken pipes or other errors by removing the connection
                print(f"⚠️ Error sending to client: {e}")
                self.disconnect(connection, floor_id)

# Initialize FastAPI and Manager
app = FastAPI(
    title="MCT System API",
    description="""
## Multi-Camera Tracking (MCT) System API

Hệ thống theo dõi đa camera với nhận diện khuôn mặt và ReID.

### Tính năng:
- **7 tầng** (F1-F7) với quản lý bật/tắt theo tầng
- **20+ cameras** với quản lý bật/tắt từng camera
- **WebSocket** broadcast dữ liệu tracking theo tầng
- **Face Recognition** + Body ReID

### WebSocket:
- Kết nối: `ws://host:8068/ws/{floor_id}`
- Nhận dữ liệu tracking realtime theo tầng
""",
    version="2.0.0",
    openapi_tags=[
        {"name": "System", "description": "Trạng thái hệ thống"},
        {"name": "Floors", "description": "Quản lý bật/tắt tầng"},
        {"name": "Cameras", "description": "Quản lý bật/tắt camera"},
        {"name": "WebSocket", "description": "WebSocket endpoints"},
    ]
)
manager = ConnectionManager()

@app.get("/", tags=["System"], summary="Health Check")
async def get_root():
    """Kiểm tra server đang chạy."""
    return {"status": "ok", "message": "MCT WebSocket Server is Running"}


# =============================================================================
# REST API: System Status & Floor/Camera Management
# =============================================================================

@app.get("/api/status", tags=["System"], summary="Trạng thái hệ thống")
async def api_get_status():
    """Xem trạng thái toàn bộ hệ thống: tầng, camera, số lượng active."""
    return get_system_status()


@app.post("/api/floors/{floor_id}/enable", tags=["Floors"], summary="Bật tầng")
async def api_enable_floor(floor_id: int):
    """Bật tầng (bắt đầu xử lý cameras trên tầng này). Tự động bật tất cả cameras trên tầng."""
    with _status_lock:
        if floor_id not in _floor_status:
            return {"error": f"Floor {floor_id} not found", "available": list(_floor_status.keys())}
        _floor_status[floor_id] = True
        # Also enable all cameras on this floor
        enabled_cams = []
        for cam in _all_cameras_info:
            if cam['floor'] == floor_id:
                _camera_status[cam['id']] = True
                enabled_cams.append(cam['id'])
    print(f"✅ API: Floor {floor_id} ENABLED (cameras: {enabled_cams})")
    return {"floor": floor_id, "enabled": True, "enabled_cameras": enabled_cams}


@app.post("/api/floors/{floor_id}/disable", tags=["Floors"], summary="Tắt tầng")
async def api_disable_floor(floor_id: int):
    """Tắt tầng (dừng xử lý tất cả cameras trên tầng). Tự động tắt tất cả cameras trên tầng."""
    with _status_lock:
        if floor_id not in _floor_status:
            return {"error": f"Floor {floor_id} not found", "available": list(_floor_status.keys())}
        _floor_status[floor_id] = False
        # Also disable all cameras on this floor
        disabled_cams = []
        for cam in _all_cameras_info:
            if cam['floor'] == floor_id:
                _camera_status[cam['id']] = False
                disabled_cams.append(cam['id'])
    print(f"🔴 API: Floor {floor_id} DISABLED (cameras: {disabled_cams})")
    return {"floor": floor_id, "enabled": False, "disabled_cameras": disabled_cams}


@app.post("/api/floors/{floor_id}/toggle", tags=["Floors"], summary="Bật/Tắt tầng")
async def api_toggle_floor(floor_id: int):
    """Đảo trạng thái tầng (bật→tắt, tắt→bật). Tự động bật/tắt tất cả cameras trên tầng."""
    with _status_lock:
        if floor_id not in _floor_status:
            return {"error": f"Floor {floor_id} not found", "available": list(_floor_status.keys())}
        _floor_status[floor_id] = not _floor_status[floor_id]
        new_state = _floor_status[floor_id]
        # Also toggle all cameras on this floor to match
        toggled_cams = []
        for cam in _all_cameras_info:
            if cam['floor'] == floor_id:
                _camera_status[cam['id']] = new_state
                toggled_cams.append(cam['id'])
    state_str = "ENABLED" if new_state else "DISABLED"
    print(f"🔄 API: Floor {floor_id} {state_str} (cameras: {toggled_cams})")
    return {"floor": floor_id, "enabled": new_state, "cameras": toggled_cams}


@app.post("/api/cameras/{cam_id}/enable", tags=["Cameras"], summary="Bật camera")
async def api_enable_camera(cam_id: str):
    """Bật một camera cụ thể (VD: cam45, cam36)."""
    with _status_lock:
        if cam_id not in _camera_status:
            return {"error": f"Camera {cam_id} not found", "available": list(_camera_status.keys())}
        _camera_status[cam_id] = True
    print(f"✅ API: Camera {cam_id} ENABLED")
    return {"camera": cam_id, "enabled": True}


@app.post("/api/cameras/{cam_id}/disable", tags=["Cameras"], summary="Tắt camera")
async def api_disable_camera(cam_id: str):
    """Tắt một camera cụ thể."""
    with _status_lock:
        if cam_id not in _camera_status:
            return {"error": f"Camera {cam_id} not found", "available": list(_camera_status.keys())}
        _camera_status[cam_id] = False
    print(f"🔴 API: Camera {cam_id} DISABLED")
    return {"camera": cam_id, "enabled": False}


@app.post("/api/cameras/{cam_id}/toggle", tags=["Cameras"], summary="Bật/Tắt camera")
async def api_toggle_camera(cam_id: str):
    """Đảo trạng thái camera (bật→tắt, tắt→bật)."""
    with _status_lock:
        if cam_id not in _camera_status:
            return {"error": f"Camera {cam_id} not found", "available": list(_camera_status.keys())}
        _camera_status[cam_id] = not _camera_status[cam_id]
        new_state = _camera_status[cam_id]
    state_str = "ENABLED" if new_state else "DISABLED"
    print(f"🔄 API: Camera {cam_id} {state_str}")
    return {"camera": cam_id, "enabled": new_state}


@app.on_event("startup")
async def log_routes():
    print("🛣️  Registered Routes:")
    for route in app.routes:
        print(f"   - {route.path} ({route.name})")

@app.websocket("/ws/{floor_id}")
async def websocket_endpoint(websocket: WebSocket, floor_id: int):
    await manager.connect(websocket, floor_id)
    try:
        while True:
            # Keep the connection open. Current logic is purely Push-based from server.
            # We can also listen for client messages (e.g. "ping") if needed.
            data = await websocket.receive_text()
            # If client sends something, we can ignore or respond.
            # print(f"Client {floor_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, floor_id)
    except Exception:
        manager.disconnect(websocket, floor_id)

def start_api_server(host="0.0.0.0", port=8068):
    """
    Function to start the server in a thread.
    """
    print(f"🚀 Starting WebSocket API Server at ws://{host}:{port}/ws/{{floor_id}}")
    uvicorn.run(app, host=host, port=port, log_level="info")

def broadcast_sync(floor_id: int, message: dict):
    """
    Helper to bridge between Sync code (demo_mct.py) and Async broadcast.
    Since uvicorn manages its own loop in a separate thread, 
    we need a way to schedule the async broadcast from the main thread.
    
    However, calling async from a different thread's loop is tricky.
    
    SIMPLIFIED APPROACH for Demo:
    We will rely on `run_coroutine_threadsafe` if we can access the loop,
    OR we can just rely on the fact that FastAPI is running in its own thread.
    
    Actually, a robust way in Python Sync->Async bridge is tricky.
    
    BETTER APPROACH:
    Use a shared Queue or just let the main thread loop run async? No, demo_mct is sync heavy.
    
    Let's try a safe persistent loop approach or just use `asyncio.run`? No.
    
    We will add a helper in the `app` that checks a Queue? 
    Or easier: make `manager.broadcast` purely async, and in `demo_mct.py`, we construct the loop?
    
    WAIT. uvicorn.run blocks. It runs an event loop.
    To interact with that loop from `demo_mct` (main thread), we need access to it.
    
    EASIEST SOLUTION FOR DEMO:
    Just run `uvicorn.run` in a daemon thread. 
    To send data, we can't easily jump into that thread's loop without reference.
    
    Alternative: `demo_mct` loop puts data into a thread-safe `queue`.
    The FastAPI app has a background task that reads this queue and broadcasts.
    """
    pass

# --- Queue-based Bridge Impl ---
import queue
message_queue = queue.Queue()

@app.on_event("startup")
async def startup_event():
    # Start a background task to consume the queue
    asyncio.create_task(queue_consumer())

async def queue_consumer():
    loop = asyncio.get_running_loop()
    while True:
        try:
            # Use run_in_executor to wait for queue without blocking the event loop
            # and without polling (sleep)
            floor_id, msg = await loop.run_in_executor(None, message_queue.get)
            await manager.broadcast(floor_id, msg)
        except Exception as e:
            print(f"❌ Queue consumer error: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

def send_update(floor_id: int, data: dict):
    """
    External function called by demo_mct.py
    """
    message_queue.put((floor_id, data))

if __name__ == "__main__":
    start_api_server()
