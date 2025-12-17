import os
import subprocess
import time
import pandas as pd

# Constants
CU_FILE = 'rk_cuda.cu'
EXE_FILE = 'rk_cuda'

# User-provided: Set your pattern file and text file here
PAT_FILE = '/work/u8449362/data_density/density_pattern_len64_pattern.txt'
TEXT_FILE = '/work/u8449362/data_density/density_case_050_d3.35e-04_text.txt'

# Define test configurations (different states/conditions)
# You can modify these lists to test more/different values
chunk_sizes_mb = [64, 128, 256, 512]  # Chunk sizes in MB
block_counts = [32, 64, 128, 256, 512] # Number of blocks
threads_per_block = [64, 128, 256, 512] # Threads per block

chunk_sizes_bytes = [mb * 1024 * 1024 for mb in chunk_sizes_mb]

# Function to compile the CUDA code (if not already compiled)
def compile_cuda():
    if os.path.exists(EXE_FILE):
        print(f"{EXE_FILE} already exists. Skipping compilation.")
        return True
    if not os.path.exists(CU_FILE):
        print(f"Error: {CU_FILE} not found.")
        return False
    cmd = ['nvcc', '-O3', '-std=c++17', CU_FILE, '-o', EXE_FILE]
    try:
        subprocess.check_call(cmd)
        print(f"Compiled {EXE_FILE} successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        return False
    except FileNotFoundError:
        print("nvcc not found. Ensure CUDA is installed and nvcc is in PATH.")
        return False

# Function to run the executable with given parameters and capture output
def run_test(pat_file, text_file, chunk_bytes, blocks, threads):
    cmd = [f'./{EXE_FILE}', pat_file, text_file, str(chunk_bytes), str(blocks), str(threads)]
    start_time = time.time()
    try:
        output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT, timeout=600)
        exec_time = time.time() - start_time
        lines = output.splitlines()
        matches = 0
        for line in lines:
            if 'Matches:' in line:
                matches = int(line.split(':')[1].strip())
                break
        return {
            'matches': matches,
            'exec_time_s': exec_time,
            'success': True
        }
    except subprocess.CalledProcessError as e:
        print(f"Run failed for chunk={chunk_bytes}, blocks={blocks}, threads={threads}: {e}")
        return {'success': False}
    except subprocess.TimeoutExpired:
        print(f"Timeout for chunk={chunk_bytes}, blocks={blocks}, threads={threads}")
        return {'success': False}
    except Exception as e:
        print(f"Error: {e}")
        return {'success': False}

# Main testing function
def main():
    # Compile if needed
    if not compile_cuda():
        return

    # Run tests and collect results for different configurations
    results = []
    for chunk_bytes in chunk_sizes_bytes:
        for blocks in block_counts:
            for threads in threads_per_block:
                print(f"Testing: chunk={chunk_bytes//(1024*1024)}MB, blocks={blocks}, threads={threads}")
                result = run_test(PAT_FILE, TEXT_FILE, chunk_bytes, blocks, threads)
                if result['success']:
                    results.append({
                        'chunk_size_mb': chunk_bytes // (1024 * 1024),
                        'blocks': blocks,
                        'threads_per_block': threads,
                        'total_threads': blocks * threads,
                        'matches': result['matches'],
                        'exec_time_s': result['exec_time_s']
                    })

    # Save results to CSV and print summary
    if results:
        df = pd.DataFrame(results)
        df.to_csv('rk_test_results.csv', index=False)
        print("\nRabin-Karp CUDA Test Results Summary:")
        print(df)
        print(f"\nBest configuration (by time):")
        best = df.loc[df['exec_time_s'].idxmin()]
        print(best)
    else:
        print("No successful tests.")

if __name__ == '__main__':
    main()
