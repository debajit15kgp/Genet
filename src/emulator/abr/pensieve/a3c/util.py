import numpy as np
import torch
from pensieve.a3c.models import create_mask, Seq2SeqWithEmbeddingmodClass

# Define bucket boundaries (same as original)
bucket_boundaries_ccbench = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.63, 2.12, 4.02, 8, 12],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5],
    3: [0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25]
}

def bucketize_with_offset(values_1d, boundaries, offset):
    """Helper function to bucketize values with an offset."""
    bins_local = np.searchsorted(boundaries, values_1d, side='right')
    return bins_local + offset

def read_tcp_stats(filepath, window_ms=800):
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        chunk_size = 50000
        size = f.tell()
        f.seek(max(0, size - chunk_size), 0)
        lines = f.read().decode().splitlines()

        if not lines:
            return np.zeros((1, 6))

        stats = []
        current_time = float(lines[-1].split(',')[0]) / 1e6
        cutoff_time = current_time - window_ms

        for line in reversed(lines):
            values = line.strip().split(',')
            timestamp = float(values[0]) / 1e6
            
            if timestamp < cutoff_time:
                break
                
            try:
                stats.append([float(x) for x in values])
            except ValueError:
                continue

        if not stats:
            return np.zeros((1, 6))
            
        return np.array(stats[::-1]) 

def process_tcp_stats(tcp_stats_array):
    if tcp_stats_array.size == 0:
        return np.zeros((1, 6))
    
    processed_data = tcp_stats_array.copy()
    N = len(processed_data)
    
    base_rtt_vals = processed_data[:, 0]
    rtt_min = base_rtt_vals.min()
    rtt_max = base_rtt_vals.max()
    
    if np.isclose(rtt_min, rtt_max):
        processed_data[:, 0] = 0.0
    else:
        processed_data[:, 0] = (base_rtt_vals - rtt_min) / (rtt_max - rtt_min)
    
    feature_ids = [1, 2, 3, 4, 5]
    bin_offsets = {}
    running_offset = 0
    
    for feat_idx in feature_ids:
        boundaries = bucket_boundaries_ccbench[feat_idx]
        bin_offsets[feat_idx] = running_offset
        running_offset += (len(boundaries) + 1)
    
    for feat_idx in feature_ids:
        col_vals = processed_data[:, feat_idx]
        boundaries = bucket_boundaries_ccbench[feat_idx]
        offset = bin_offsets[feat_idx]
        discrete_bins = bucketize_with_offset(col_vals, boundaries, offset)
        processed_data[:, feat_idx] = discrete_bins
    
    return processed_data

def transform_for_network(tcp_stats_array):
    processed_stats = process_tcp_stats(tcp_stats_array)
    
    if len(processed_stats.shape) == 2:
        processed_stats = np.expand_dims(processed_stats, 0)
        
    return processed_stats

def get_model_embeddings(inputs):
    inputs = torch.tensor(inputs)
    input_batch = inputs.unsqueeze(0)  # Shape becomes (1, 10, 6)
    
    enc_input = input_batch[:, :-prediction_len, :].to(device)  
    dec_input = (1.5 * torch.ones((1, prediction_len, input_batch.shape[2]))).to(device)
    
    src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=pad_idx, device=device)
    model = torch.load('models/RTT-Checkpoint-BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-809iter.p', map_location='cpu')

    probs, embeddings = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
    return embeddings