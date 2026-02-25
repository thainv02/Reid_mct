"""
Person tracking classes for MCT.
"""
from collections import deque


class PendingTrack:
    """
    Represents a track that is pending confirmation.
    Accumulates features until confirmed.
    """
    
    def __init__(self, box, feat):
        """
        Initialize pending track.
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            feat: Feature vector
        """
        self.box = box
        self.feat = feat
        self.count = 1
        self.features = [feat]  # Store all features for later batch add


class ConfirmedTrack:
    """
    Represents a confirmed track with stable ID.
    Uses voting mechanism for ID stability.
    """
    
    def __init__(self, person_id, box, feat):
        """
        Initialize confirmed track.
        
        Args:
            person_id: Stable person ID
            box: Bounding box [x1, y1, x2, y2]
            feat: Feature vector
        """
        self.person_id = person_id      # The stable, confirmed ID
        self.box = box
        self.feat = feat
        self.miss_count = 0
        self.vote_history = deque(maxlen=10)  # Store last 10 raw ReID results
        self.vote_history.append(person_id)
        self.display_id = person_id     # ID currently displayed
        self.consecutive_matches = 0    # Track how stable the current raw vote is

    def update(self, box, feat, raw_voting_id):
        """
        Update track with new detection.
        
        Args:
            box: New bounding box
            feat: New feature vector
            raw_voting_id: Raw ReID result for this frame
        """
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
            THRESHOLD = max(3, HISTORY_LEN // 2 + 1)  # Majority vote
            
            if best_id != self.display_id:
                if count >= THRESHOLD:
                    self.display_id = best_id
