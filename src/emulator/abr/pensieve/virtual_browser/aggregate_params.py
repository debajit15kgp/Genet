import pandas as pd
import numpy as np
from collections import defaultdict
import time
from typing import Dict, List
import sys
import math

AGGREGATION_WINDOW_MS = 80 
WINDOW = 10
MSS = 1448 # bytes

class TCPStatsAggregator:
    def __init__(self):
        self.window_data = defaultdict(list)
        self.window_start_time = None
        
    def aggregate_window_data(self):
        if not self.window_data:
            return None
            
        stats = {}
        
        print(len(self.window_data['srtt_us']))
        rtts = [float(x) / 100000.0 / 8  for x in self.window_data['srtt_us']]
        stats['rtt_ms'] = np.mean(rtts) if rtts else 0
        
        rttvars = [float(x) / 1000.0 for x in self.window_data['rttvar']]
        stats['rttvar_ms'] = np.mean(rttvars) if rttvars else 0

        cwnd_rate = [float(x) for x in self.window_data['cwnd_rate']]
        stats['cwnd_rate'] = np.mean(cwnd_rate) if cwnd_rate else 0

        l_w_mbps = [float(x) * 8.0 for x in self.window_data['l_w_mbps']]
        stats['l_w_mbps'] = np.mean(l_w_mbps) if cwnd_rate else 0

        delivery_rate = [float(x) / 125000.0 / 100 for x in self.window_data['delivery_rate']]
        stats['delivery_rate'] = np.mean(delivery_rate) if cwnd_rate else 0
        
        stats['window_start'] = self.window_start_time
        stats['window_end'] = self.window_data['time_ms'][-1]
        stats['num_packets'] = len(rtts)
        
        return stats

def process_array_data(input_array: np.ndarray):
    aggregator = TCPStatsAggregator()
    metrics_data = []
    
    start_time_ms = input_array[0, 0] / 1000000
    last_time_ms = start_time_ms
    last_delivered = input_array[0, 3]
    lost_packets = 0
    
    aggregator.window_start_time = start_time_ms
    
    for row in input_array:
        current_time_ns = row[0]
        current_time_ms = current_time_ns / 1000000
        
        time_diff = current_time_ms - start_time_ms
        window_diff = current_time_ms - last_time_ms

        if time_diff >= AGGREGATION_WINDOW_MS:
            stats = aggregator.aggregate_window_data()
            if stats:
                metrics_data.append([
                    0.8,
                    stats['rtt_ms'],
                    stats['rttvar_ms'],
                    stats['delivery_rate'],
                    stats['l_w_mbps'],
                    stats['cwnd_rate']
                ])
            
            aggregator = TCPStatsAggregator()
            start_time_ms = current_time_ms
            aggregator.window_start_time = start_time_ms

        lost_packets += row[6]
        
        if window_diff >= WINDOW:
            aggregator.window_data['time_ms'].append(current_time_ms)
            aggregator.window_data['srtt_us'].append(row[1])
            aggregator.window_data['rttvar'].append(row[2])
            
            rate_interval = float(row[4])
            if rate_interval == 0:
                rate_interval = 1
            aggregator.window_data['delivery_rate'].append(row[3] * row[5] * 1000000 / rate_interval)
            
            aggregator.window_data['l_w_mbps'].append(lost_packets/window_diff)
            
            cwnd = row[6]
            if cwnd > 0:
                cwnd_rate = round(math.log2(cwnd) * 1000) / 1000
            else:
                cwnd_rate = math.log2(0.0001)
            aggregator.window_data['cwnd_rate'].append(cwnd_rate)
            
            last_time_ms = current_time_ms
            last_delivered = row[3]
            lost_packets = 0
    
    return np.array(metrics_data)

def process_log_file(log_path: str):
    aggregator = TCPStatsAggregator()
    current_lines = []
    metrics_data = []
    
    with open(log_path, "r") as f:
        first_line = f.readline()
        parts = first_line.strip().split(',')
        start_time_ns = float(parts[0])
        start_time_ms = start_time_ns / 1000000
        last_time_ms = start_time_ms
        last_delivered = float(parts[3])
        lost_packets = 0
        
        aggregator.window_start_time = start_time_ms
        # aggregator.window_data['time_ms'].append(start_time_ms)
        # aggregator.window_data['srtt_us'].append(parts[1])
        
        for line in f:
            if not line.strip():
                continue
                
            parts = line.strip().split(',')
            current_time_ns = float(parts[0])
            current_time_ms = current_time_ns / 1000000
            
            time_diff = current_time_ms - start_time_ms
            window_diff = current_time_ms - last_time_ms
            
            # print(current_time_ms, last_time_ms)

            if time_diff >= AGGREGATION_WINDOW_MS:
                stats = aggregator.aggregate_window_data()
                if stats:
                    metrics_data.append([
                        0.8,
                        stats['rtt_ms'],
                        stats['rttvar_ms'],
                        stats['delivery_rate'],
                        stats['l_w_mbps'],
                        stats['cwnd_rate']
                    ])

                    print(f"Window {stats['window_start']:.2f}ms - {stats['window_end']:.2f}ms "
                          f"({stats['num_packets']} packets):\n"
                          f"  Average RTT: {stats['rtt_ms']:.2f}ms\n"
                          f"  RTT Variance: {stats['rttvar_ms']:.2f}ms\n"
                          f"  Delivery Rate: {stats['delivery_rate']:.2f}\n"
                          f"  L W mbps: {stats['l_w_mbps']:.2f}\n"
                          f"  Cwnd rate: {stats['cwnd_rate']:.2f}\n")
                
                aggregator = TCPStatsAggregator()
                start_time_ms = current_time_ms
                aggregator.window_start_time = start_time_ms

            lost_packets += float(parts[6])
            
            if window_diff >= WINDOW:
                aggregator.window_data['time_ms'].append(current_time_ms)
                aggregator.window_data['srtt_us'].append(parts[1])
                aggregator.window_data['rttvar'].append(parts[2])
                rate_interval = float(parts[4])
                if rate_interval == 0:
                    rate_interval = 1
                aggregator.window_data['delivery_rate'].append(float(parts[3])*float(parts[5])*1000000 / rate_interval)
                aggregator.window_data['l_w_mbps'].append(lost_packets/window_diff)
                cwnd = float(parts[-3])
                if cwnd>0:
                    cwnd_rate = round(math.log2(cwnd)*1000)/1000
                else:
                    cwnd_rate = math.log2(0.0001)
                aggregator.window_data['cwnd_rate'].append(cwnd_rate)
                last_time_ms = current_time_ms
                last_delivered = float(parts[3])
                lost_packets = 0
    
    # Convert to numpy array and save
    metrics_array = np.array(metrics_data)
    np.save('tcp_metrics_reno_trace_file_1.npy', metrics_array)
                            

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tcp_aggregator.py <bpftrace_log_file>")
        sys.exit(1)
        
    log_path = sys.argv[1]
    process_log_file(log_path)