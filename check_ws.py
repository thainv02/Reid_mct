import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://127.0.0.1:8068/ws/3"
    print(f"🔄 Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri} successfully!")
            
            print("⏳ Waiting for message...")
            try:
                # Wait for a message (timeout 5s)
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"📩 Received: {message}")
            except asyncio.TimeoutError:
                print("⚠️ No message received in 10s (Server might be idle but connected)")
            except Exception as e:
                 print(f"⚠️ Error receiving: {e}")
                 
    except Exception as e:
        print(f"❌ Connection FAILED: {e}")

if __name__ == "__main__":
    # Install websockets if missing: pip install websockets
    try:
        asyncio.run(test_connection())
    except ImportError:
        print("Please run: pip install websockets")
