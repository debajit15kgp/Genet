import numpy as np
import requests
import json
import time
from pensieve.constants import (
    A_DIM, BUFFER_NORM_FACTOR, DEFAULT_QUALITY, M_IN_K,
    S_INFO, S_LEN, TOTAL_VIDEO_CHUNK, VIDEO_BIT_RATE
)

class ServerEnvironment:
    """Environment that interacts with the ABR server for training.
    
    This environment maintains communication with a running ABR server instance,
    simulating real video streaming conditions during training.
    """
    def __init__(self, server_url, video_sizes, random_seed=42):
        """Initialize server environment.
        
        Args:
            server_url (str): URL of the ABR server (e.g. 'http://localhost:8333')
            video_sizes (dict): Mapping of bitrates to chunk sizes
            random_seed (int): Random seed for reproducibility
        """
        self.server_url = server_url
        self.video_sizes = video_sizes
        np.random.seed(random_seed)
        
        # State tracking
        self.chunk_idx = 0
        self.buffer_size = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((1, S_INFO, S_LEN))
        self.last_total_rebuf = 0
        
    def reset(self):
        """Reset environment to initial state."""
        self.chunk_idx = 0
        self.buffer_size = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((1, S_INFO, S_LEN))
        self.last_total_rebuf = 0
        return self.state

    def step(self, action):
        """Take action in environment by sending request to server.
        
        Args:
            action (int): Selected bitrate index
            
        Returns:
            next_state: Updated environment state
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information dictionary
        """
        # Prepare request data
        post_data = {
            'lastRequest': self.chunk_idx,
            'lastquality': self.last_bit_rate,
            'buffer': self.buffer_size,
            'RebufferTime': self.last_total_rebuf,
            'bandwidthEst': 0,  # Will be computed by server
            'lastChunkStartTime': time.time() * 1000,  # Convert to ms
            'lastChunkSize': self.video_sizes[self.last_bit_rate][self.chunk_idx] 
                           if self.chunk_idx < TOTAL_VIDEO_CHUNK else 0
        }

        # Send request to server
        response = requests.post(
            self.server_url,
            data=json.dumps(post_data),
            headers={'Content-Type': 'application/json'}
        )
        
        # Parse server response
        server_response = response.json()
        
        # Update state based on server response
        rebuffer_time = float(server_response['RebufferTime'] - self.last_total_rebuf)
        self.buffer_size = server_response['buffer']
        video_chunk_fetch_time = (server_response['lastChunkFinishTime'] - 
                                server_response['lastChunkStartTime'])
        video_chunk_size = server_response['lastChunkSize']
        
        # Calculate reward
        reward = server_response['reward']
        
        # Update state tensor (following server state update logic)
        self.state = np.roll(self.state, -1, axis=2)
        self.state[0, 0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE))
        self.state[0, 1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        self.state[0, 2, -1] = float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K
        self.state[0, 3, -1] = float(video_chunk_fetch_time) / M_IN_K / BUFFER_NORM_FACTOR
        
        # Next chunk sizes
        next_chunk_sizes = []
        for i in range(A_DIM):
            if self.chunk_idx + 1 < TOTAL_VIDEO_CHUNK:
                next_chunk_sizes.append(self.video_sizes[i][self.chunk_idx + 1])
            else:
                next_chunk_sizes.append(0)
        self.state[0, 4, :A_DIM] = np.array(next_chunk_sizes) / M_IN_K / M_IN_K
        
        # Chunks remaining
        self.state[0, 5, -1] = np.minimum(self.chunk_idx + 1, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

        # Update tracking variables
        self.chunk_idx += 1
        self.last_bit_rate = action
        self.last_total_rebuf = server_response['RebufferTime']
        
        # Check if done
        done = (self.chunk_idx >= TOTAL_VIDEO_CHUNK)
        
        info = {
            'video_chunk_size': video_chunk_size,
            'delay': video_chunk_fetch_time,
            'rebuf': rebuffer_time,
            'buffer_size': self.buffer_size,
        }
        
        return self.state, reward, done, info