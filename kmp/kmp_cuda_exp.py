import os
import subprocess
import time
import pandas as pd

CU_FILE = 'kmp_cuda.cu'
EXE_FILE = 'kmp_cuda'

PAT_FILE = '../data/data_density/density_pattern_len64_pattern.txt'
TEXT_FILE = '../data/data_density/density_case_050_d3.35e-04_text.txt'

chunk_sizes_mb = [64, 128, 256, 512]
block_counts = [32, 64, 128, 256, 512]

chunk_sizes_bytes = [mb * 1024 * 1024 for mb in chunk_sizes_mb]

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

def run_test(pat_file, text_file, chunk_bytes, blocks):
    cmd = [f'./{EXE_FILE}', pat_file, text_file, str(chunk_bytes), str(blocks)]
    start_time = time.time()
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        exec_time = time.time() - start_time
        lines = output.splitlines()
        matches = int(lines[0].split(': ')[1]) if lines else 0
        reported_time = float(lines[1].split(': ')[1]) if len(lines) > 1 else exec_time
        return {
            'matches': matches,
            'reported_time_s': reported_time,
            'exec_time_s': exec_time,
            'success': True
        }
    except subprocess.CalledProcessError as e:
        print(f"Run failed for chunk={chunk_bytes}, blocks={blocks}: {e}")
        return {'success': False}

def main():
    if not compile_cuda():
        return

    results = []
    for chunk_bytes in chunk_sizes_bytes:
        for blocks in block_counts:
            print(f"Testing: chunk={chunk_bytes//(1024*1024)}MB, blocks={blocks}")
            result = run_test(PAT_FILE, TEXT_FILE, chunk_bytes, blocks)
            if result['success']:
                results.append({
                    'chunk_size_mb': chunk_bytes // (1024 * 1024),
                    'blocks': blocks,
                    'matches': result['matches'],
                    'reported_time_s': result['reported_time_s'],
                    'exec_time_s': result['exec_time_s']
                })

    if results:
        df = pd.DataFrame(results)
        df.to_csv('kmp_cuda_results.csv', index=False)
        print("\nKMP CUDA Test Results Summary:")
        print(df)
    else:
        print("No successful tests.")

if __name__ == '__main__':
    main()

