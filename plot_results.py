import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_bm_cuda_results(csv_path='bm/bm_cuda_results.csv', output_dir='plots'):
    """Plot Boyer-Moore CUDA experiment results"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmap: Time vs chunk_size and blocks
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Using reported_time_s
    pivot1 = df.pivot_table(values='reported_time_s', index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('BM CUDA: Reported Time (s)\nChunk Size vs Blocks')
    axes[0].set_xlabel('Chunk Size (MB)')
    axes[0].set_ylabel('Number of Blocks')
    
    # Using exec_time_s
    pivot2 = df.pivot_table(values='exec_time_s', index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('BM CUDA: Execution Time (s)\nChunk Size vs Blocks')
    axes[1].set_xlabel('Chunk Size (MB)')
    axes[1].set_ylabel('Number of Blocks')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bm_cuda_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/bm_cuda_heatmap.png")
    plt.close()
    
    # 2. Line plot: Effect of blocks on time for different chunk sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    for chunk_size in sorted(df['chunk_size_mb'].unique()):
        subset = df[df['chunk_size_mb'] == chunk_size]
        ax.plot(subset['blocks'], subset['reported_time_s'], marker='o', label=f'{chunk_size} MB')
    
    ax.set_xlabel('Number of Blocks')
    ax.set_ylabel('Time (s)')
    ax.set_title('BM CUDA: Performance vs Number of Blocks')
    ax.legend(title='Chunk Size')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/bm_cuda_blocks.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/bm_cuda_blocks.png")
    plt.close()
    
    # 3. Bar plot: Best configuration
    best_idx = df['reported_time_s'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest BM CUDA configuration:")
    print(f"  Chunk: {best['chunk_size_mb']} MB, Blocks: {best['blocks']}")
    print(f"  Time: {best['reported_time_s']:.4f} s")

def plot_rk_cuda_results(csv_path='rk/rk_test_results.csv', output_dir='plots'):
    """Plot Rabin-Karp CUDA experiment results"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for threads_per_block = 64 only
    df = df[df['threads_per_block'] == 64].copy()
    
    if len(df) == 0:
        print("No data found for threads_per_block=64")
        return
    
    # 1. Heatmap: Time vs chunk_size and blocks (similar to BM)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Using reported_time_s if available, otherwise exec_time_s
    time_col_reported = 'reported_time_s' if 'reported_time_s' in df.columns else 'exec_time_s'
    pivot1 = df.pivot_table(values=time_col_reported, index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('RK CUDA: Reported Time (s)\nChunk Size vs Blocks (threads_per_block=64)')
    axes[0].set_xlabel('Chunk Size (MB)')
    axes[0].set_ylabel('Number of Blocks')
    
    # Using exec_time_s
    pivot2 = df.pivot_table(values='exec_time_s', index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('RK CUDA: Execution Time (s)\nChunk Size vs Blocks (threads_per_block=64)')
    axes[1].set_xlabel('Chunk Size (MB)')
    axes[1].set_ylabel('Number of Blocks')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rk_cuda_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/rk_cuda_heatmap.png")
    plt.close()
    
    # 2. Line plot: Effect of blocks on time for different chunk sizes (similar to BM)
    fig, ax = plt.subplots(figsize=(10, 6))
    for chunk_size in sorted(df['chunk_size_mb'].unique()):
        subset = df[df['chunk_size_mb'] == chunk_size]
        ax.plot(subset['blocks'], subset[time_col_reported], marker='o', label=f'{chunk_size} MB')
    
    ax.set_xlabel('Number of Blocks')
    ax.set_ylabel('Time (s)')
    ax.set_title('RK CUDA: Performance vs Number of Blocks (threads_per_block=64)')
    ax.legend(title='Chunk Size')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/rk_cuda_blocks.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/rk_cuda_blocks.png")
    plt.close()
    
    # 3. Best configuration
    best_idx = df['exec_time_s'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest RK CUDA configuration (threads_per_block=64):")
    print(f"  Chunk: {best['chunk_size_mb']} MB, Blocks: {best['blocks']}")
    print(f"  Time: {best['exec_time_s']:.4f} s")

def plot_bf_cuda_results(csv_path='bf/bf_cuda_results.csv', output_dir='plots'):
    """Plot Brute Force CUDA experiment results"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    pivot1 = df.pivot_table(values='reported_time_s', index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0])
    axes[0].set_title('Brute Force CUDA: Reported Time (s)')
    axes[0].set_xlabel('Chunk Size (MB)')
    axes[0].set_ylabel('Number of Blocks')
    
    pivot2 = df.pivot_table(values='exec_time_s', index='blocks', columns='chunk_size_mb')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[1])
    axes[1].set_title('Brute Force CUDA: Execution Time (s)')
    axes[1].set_xlabel('Chunk Size (MB)')
    axes[1].set_ylabel('Number of Blocks')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bf_cuda_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/bf_cuda_heatmap.png")
    plt.close()
    
    # 2. Line plot: Effect of blocks on time for different chunk sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    for chunk_size in sorted(df['chunk_size_mb'].unique()):
        subset = df[df['chunk_size_mb'] == chunk_size]
        ax.plot(subset['blocks'], subset['reported_time_s'], marker='o', label=f'{chunk_size} MB')
    
    ax.set_xlabel('Number of Blocks')
    ax.set_ylabel('Time (s)')
    ax.set_title('BF CUDA: Performance vs Number of Blocks')
    ax.legend(title='Chunk Size')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/bf_cuda_blocks.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/bf_cuda_blocks.png")
    plt.close()
    
    # 3. Best configuration
    best_idx = df['reported_time_s'].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest BF CUDA configuration:")
    print(f"  Chunk: {best['chunk_size_mb']} MB, Blocks: {best['blocks']}")
    print(f"  Time: {best['reported_time_s']:.4f} s")

def plot_algorithm_comparison(bm_csv='experiment_results.csv', 
                              bf_csv='experiment_results_bf.csv',
                              rk_csv='experiment_results_rk.csv',
                              output_dir='plots'):
    """Compare different algorithms (BM, BF, RK) across implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    dfs = {}
    for name, path in [('BM', bm_csv), ('BF', bf_csv), ('RK', rk_csv)]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Algorithm'] = name
            dfs[name] = df
    
    if not dfs:
        print("No experiment result files found for comparison")
        return
    
    combined = pd.concat(dfs.values(), ignore_index=True)
    
    # Filter numeric time values only
    combined = combined[pd.to_numeric(combined['Time(s)'], errors='coerce').notna()]
    combined['Time(s)'] = pd.to_numeric(combined['Time(s)'])
    
    # Plot by method
    methods = combined['Method'].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 6))
    if len(methods) == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        subset = combined[combined['Method'] == method]
        
        if len(subset) > 0:
            pivot = subset.pivot_table(values='Time(s)', index='Dataset', columns='Algorithm', aggfunc='mean')
            pivot.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{method} Performance')
            axes[idx].set_ylabel('Time (s)')
            axes[idx].set_xlabel('Dataset')
            axes[idx].legend(title='Algorithm')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/algorithm_comparison.png")
    plt.close()
    
    # Summary statistics
    print("\n=== Algorithm Performance Summary ===")
    summary = combined.groupby(['Algorithm', 'Method'])['Time(s)'].agg(['mean', 'min', 'max', 'std'])
    print(summary)

def main():
    print("Generating plots from experiment results...\n")
    
    # Plot individual CUDA experiments
    plot_bm_cuda_results(csv_path='bm/bm_cuda_results.csv')
    plot_rk_cuda_results(csv_path='rk/rk_test_results.csv')  # Explicitly specify the path
    plot_bf_cuda_results(csv_path='bf/bf_cuda_results.csv')
    
    # Compare algorithms
    plot_algorithm_comparison()
    
    print("\n=== All plots generated successfully ===")

if __name__ == '__main__':
    main()
