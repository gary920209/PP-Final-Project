#!/usr/bin/env python3
import subprocess
import re
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
PAT_FILE = '/work/u8449362/data_density/density_pattern_len64_pattern.txt'
TEXT_FILE = '/work/u8449362/data_density/density_case_050_d3.35e-04_text.txt'

# Algorithm configurations
ALGORITHMS = {
    'BF': '/home/u8449362/PP-Final-Project/bf/bf_mpi',
    'BM': '/home/u8449362/PP-Final-Project/bm/bm_mpi',
    'RK': '/home/u8449362/PP-Final-Project/rk/rk_mpi'
}

# MPI rank configurations to test
RANK_CONFIGS = [1, 2, 4, 8, 16, 32]

def run_experiment(algorithm_name, executable, ranks, pattern_file, text_file, repeat=3):
    """Run a single experiment and return results"""
    results = []
    
    for _ in range(repeat):
        cmd = [
            'mpirun', '-np', str(ranks),
            executable,
            pattern_file,
            text_file
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Error running {algorithm_name} with {ranks} ranks:")
                print(result.stderr)
                return None
            
            # Parse output
            output = result.stdout
            matches = None
            exec_time = None
            
            match_pattern = r'Matches:\s*(\d+)'
            time_pattern = r'Time\(s\):\s*([\d.]+)'
            
            m = re.search(match_pattern, output)
            if m:
                matches = int(m.group(1))
            
            t = re.search(time_pattern, output)
            if t:
                exec_time = float(t.group(1))
            
            if matches is not None and exec_time is not None:
                results.append({
                    'matches': matches,
                    'time': exec_time
                })
        
        except subprocess.TimeoutExpired:
            print(f"Timeout for {algorithm_name} with {ranks} ranks")
            return None
        except Exception as e:
            print(f"Exception running {algorithm_name}: {e}")
            return None
    
    if not results:
        return None
    
    # Return average results
    avg_time = sum(r['time'] for r in results) / len(results)
    matches = results[0]['matches']  # Should be same for all runs
    
    return {
        'algorithm': algorithm_name,
        'ranks': ranks,
        'matches': matches,
        'time': avg_time,
        'min_time': min(r['time'] for r in results),
        'max_time': max(r['time'] for r in results)
    }

def main():
    print("=" * 60)
    print("MPI String Matching Algorithm Comparison")
    print("=" * 60)
    print(f"Pattern file: {PAT_FILE}")
    print(f"Text file: {TEXT_FILE}")
    print(f"Algorithms: {', '.join(ALGORITHMS.keys())}")
    print(f"Rank configurations: {RANK_CONFIGS}")
    print("=" * 60)
    
    all_results = []
    
    for algo_name, executable in ALGORITHMS.items():
        print(f"\nTesting {algo_name}...")
        
        # Check if executable exists
        if not Path(executable).exists():
            print(f"  Warning: {executable} not found. Skipping.")
            continue
        
        for ranks in RANK_CONFIGS:
            print(f"  Running with {ranks} ranks...", end=' ', flush=True)
            result = run_experiment(
                algo_name, executable, ranks,
                PAT_FILE, TEXT_FILE,
                repeat=3
            )
            
            if result:
                all_results.append(result)
                print(f"✓ Time: {result['time']:.6f}s, Matches: {result['matches']}")
            else:
                print("✗ Failed")
    
    if not all_results:
        print("\nNo results collected!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_file = 'mpi_comparison_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")
    
    # Calculate speedup (relative to 1 rank)
    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS (relative to 1 rank)")
    print("=" * 60)
    
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        base_time = algo_df[algo_df['ranks'] == 1]['time'].values
        
        if len(base_time) > 0:
            base_time = base_time[0]
            print(f"\n{algo}:")
            for _, row in algo_df.iterrows():
                speedup = base_time / row['time']
                efficiency = speedup / row['ranks'] * 100
                print(f"  {row['ranks']:2d} ranks: {speedup:6.2f}x speedup, {efficiency:5.1f}% efficiency")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Execution time vs ranks
    ax1 = axes[0]
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        ax1.plot(algo_df['ranks'], algo_df['time'], marker='o', label=algo, linewidth=2)
    
    ax1.set_xlabel('Number of MPI Ranks', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs MPI Ranks', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup vs ranks
    ax2 = axes[1]
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        base_time = algo_df[algo_df['ranks'] == 1]['time'].values
        
        if len(base_time) > 0:
            base_time = base_time[0]
            speedups = [base_time / t for t in algo_df['time']]
            ax2.plot(algo_df['ranks'], speedups, marker='o', label=algo, linewidth=2)
    
    # Add ideal speedup line
    ideal_ranks = RANK_CONFIGS
    ax2.plot(ideal_ranks, ideal_ranks, 'k--', label='Ideal', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Number of MPI Ranks', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Speedup vs MPI Ranks', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    
    plt.tight_layout()
    plot_file = 'mpi_comparison_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
