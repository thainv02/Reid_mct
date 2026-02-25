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
app = FastAPI()
manager = ConnectionManager()

@app.get("/")
async def get_root():
    return {"status": "ok", "message": "WebSocket Server is Running"}

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
