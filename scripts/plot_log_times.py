import sys
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_log_and_plot(filepath):
    print(f"Processing {filepath}...")
    
    # Regex for timestamp: 2026-01-27 07:56:08,725
    ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
    worker_pattern = re.compile(r' - (SpawnPoolWorker-\d+) - ')
    
    lines = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = ts_pattern.match(line)
                if match:
                    try:
                        dt = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                        lines.append({'ts': dt, 'line': line.strip()})
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return

    if not lines:
        print("No valid timestamped lines found.")
        return

    # Determine Mode
    # "If there isn't any 'System ready', then the log files is multi-processor."
    has_system_ready = any("System ready" in x['line'] for x in lines)
    
    results = {'IO': [], 'Solve': []}
    title = ""
    
    if has_system_ready:
        print("Detected Mode: Single Process / Solver Pair")
        mode = "single"
        
        # Logic:
        # All intervals between "System ready" and "Solver finished" are Solve time.
        # Others are IO times.
        # Total Time = End - Start
        # Solve Time = Sum(intervals)
        # IO Time = Total Time - Solve Time
        
        start_time = lines[0]['ts']
        end_time = lines[-1]['ts']
        total_duration = (end_time - start_time).total_seconds()
        
        solve_duration = 0.0
        current_solve_start = None
        
        # To robustly handle file structure where "Solver finished" might vary slightly
        # We look for "Solver finished" text.
        
        for entry in lines:
            if "System ready" in entry['line']:
                if current_solve_start is not None:
                    # Previous start didn't finish? Ignore or warning.
                    pass
                current_solve_start = entry['ts']
            elif "Solver finished" in entry['line']:
                if current_solve_start:
                    dur = (entry['ts'] - current_solve_start).total_seconds()
                    solve_duration += dur
                    current_solve_start = None
        
        io_duration = max(0, total_duration - solve_duration)
        
        results['IO'] = [io_duration]
        results['Solve'] = [solve_duration]
        title = "Time Distribution (Single Process)"
        
    else:
        print("Detected Mode: Multi-Processor / Solver Point")
        mode = "multi"
        
        # Logic:
        # Track every "SpawnPoolWorker-N" independently.
        # Solve interval: "PointLinearSystem:" to "Finished batch"
        
        workers = {}
        
        for entry in lines:
            w_match = worker_pattern.search(entry['line'])
            if w_match:
                w_name = w_match.group(1)
                if w_name not in workers:
                    workers[w_name] = {'lines': []}
                workers[w_name]['lines'].append(entry)
        
        print(f"Found {len(workers)} workers.")
        
        for w_name, data in workers.items():
            w_lines = data['lines']
            if not w_lines:
                continue
                
            w_start = w_lines[0]['ts']
            w_end = w_lines[-1]['ts']
            w_total = (w_end - w_start).total_seconds()
            
            w_solve = 0.0
            curr_start = None
            
            for entry in w_lines:
                # "PointLinearSystem:" starts solve
                if "PointLinearSystem:" in entry['line']:
                    curr_start = entry['ts']
                # "Finished batch" ends solve
                elif "Finished batch" in entry['line']:
                    if curr_start:
                        dur = (entry['ts'] - curr_start).total_seconds()
                        w_solve += dur
                        curr_start = None
            
            w_io = max(0, w_total - w_solve)
            
            results['IO'].append(w_io)
            results['Solve'].append(w_solve)
            
        title = f"Time Distribution (Mean of {len(workers)} Workers)"

    # Plotting
    if not results['IO'] and not results['Solve']:
        print("No data extracted to plot.")
        return

    io_mean = np.mean(results['IO']) if results['IO'] else 0
    solve_mean = np.mean(results['Solve']) if results['Solve'] else 0
    
    print(f"Mean IO: {io_mean:.2f}s")
    print(f"Mean Solve: {solve_mean:.2f}s")
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['IO Time', 'Solve Time'], [io_mean, solve_mean], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.1f}s', ha='center', va='bottom')
        
    output_path = filepath.replace('.log', '_time_dist.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_log_times.py <path_to_log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    parse_log_and_plot(log_file)
