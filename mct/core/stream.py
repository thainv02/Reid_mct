"""
Threaded video stream handler for MCT.
"""
import cv2
import threading
import time


class ThreadedStream:
    """
    Thread-safe video stream handler.
    Runs frame capture in background thread for non-blocking reads.
    """
    
    def __init__(self, src):
        """
        Initialize threaded stream.
        
        Args:
            src: Video source (RTSP URL, file path, or camera index)
        """
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
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Background thread to continuously read frames."""
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
        """
        Read the latest frame.
        
        Returns:
            tuple: (status, frame)
        """
        with self.lock:
            return self.status, self.frame if self.frame is not None else None

    def stop(self):
        """Stop the stream and release resources."""
        self.stopped = True
        self.thread.join()
        self.capture.release()
