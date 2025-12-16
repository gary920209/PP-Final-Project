import os
import subprocess
import csv
import glob

# Configuration
SOURCE_DIR = "/home/u8449362/PP-Final-Project"
DATA_DIR = "/work/u8449362/data_density"
OUTPUT_CSV = os.path.join(SOURCE_DIR, "experiment_results.csv")

# Compile commands
COMPILE_CMDS = [
    (f"g++ -O3 -std=c++17 {os.path.join(SOURCE_DIR, 'bm_seq.cpp')} -o {os.path.join(SOURCE_DIR, 'bm_seq')}", "bm_seq"),
    (f"mpicxx -O3 -std=c++17 {os.path.join(SOURCE_DIR, 'bm_mpi.cpp')} -o {os.path.join(SOURCE_DIR, 'bm_mpi')}", "bm_mpi"),
    (f"nvcc -O3 -std=c++17 {os.path.join(SOURCE_DIR, 'bm_cuda.cu')} -o {os.path.join(SOURCE_DIR, 'bm_cuda')}", "bm_cuda")
]

def compile_code():
    print("Compiling...")
    for cmd, name in COMPILE_CMDS:
        print(f"  {name}...")
        ret = subprocess.run(cmd, shell=True, capture_output=True)
        if ret.returncode != 0:
            print(f"Error compiling {name}: {ret.stderr.decode()}")
            exit(1)

def parse_output(output):
    matches = 0
    time_sec = 0.0
    for line in output.splitlines():
        if "Matches:" in line:
            try:
                matches = int(line.split(":")[1].strip())
            except:
                pass
        if "Time(s):" in line:
            try:
                # Handle "0.123456 (max rank time)"
                parts = line.split(":")[1].strip().split()
                time_sec = float(parts[0])
            except:
                pass
    return matches, time_sec

def find_datasets():
    datasets = []
    # Walk through data directory
    for root, dirs, files in os.walk(DATA_DIR):
        # Heuristic: Find files containing 'pat' (pattern) and pair with 'txt' (text)
        patterns = sorted([f for f in files if 'pat' in f.lower()])
        
        # Identify potential text files (files with 'text' in name, or just not patterns)
        potential_texts = sorted([f for f in files if 'text' in f.lower() and f not in patterns])

        # Special Case: 1 Pattern, Multiple Texts -> Run pattern against ALL texts
        if len(patterns) == 1 and len(potential_texts) > 0:
            pat_path = os.path.join(root, patterns[0])
            for t in potential_texts:
                datasets.append((pat_path, os.path.join(root, t)))
            continue

        # Default Case: Try to match patterns to texts 1-to-1
        for p in patterns:
            pat_path = os.path.join(root, p)
            
            # Try to find corresponding text file
            # 1. Replace 'pat' with 'txt'
            candidates = [
                p.replace('pat', 'txt'),
                p.replace('pattern', 'text'),
                p.replace('Pattern', 'Text')
            ]
            
            txt_path = None
            for c in candidates:
                if c in files and c != p:
                    txt_path = os.path.join(root, c)
                    break
            
            # 2. If not found, look for a generic text file in the directory
            # (Only if we haven't already handled the 1-pattern-many-texts case above)
            if not txt_path and len(potential_texts) == 1:
                txt_path = os.path.join(root, potential_texts[0])

            # 3. If still not found, look for any other file if it's not a pattern
            if not txt_path:
                others = [f for f in files if f != p and f not in patterns]
                if len(others) == 1:
                    txt_path = os.path.join(root, others[0])
            
            if txt_path:
                datasets.append((pat_path, txt_path))
            else:
                print(f"Skipping pattern {p}: No matching text file found.")
    
    # Sort datasets by text filename to ensure order (000 -> 099)
    # (Primary sort by pattern, secondary by text)
    datasets.sort(key=lambda x: (os.path.basename(x[0]), os.path.basename(x[1])))
    
    return datasets

def run_experiment():
    datasets = find_datasets()
    print(f"Found {len(datasets)} datasets.")
    
    results = []

    for pat_file, txt_file in datasets:
        case_name = f"{os.path.basename(pat_file)} vs {os.path.basename(txt_file)}"
        print(f"Running {case_name}...")
        
        # 1. Sequential
        cmd = [os.path.join(SOURCE_DIR, 'bm_seq'), pat_file, txt_file]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode == 0:
                m, t = parse_output(res.stdout)
                results.append([case_name, "Sequential", t, m])
                print(f"  Seq: {t:.4f}s")
            else:
                print(f"  Seq Error: {res.stderr}")
                results.append([case_name, "Sequential", "Error", 0])
        except Exception as e:
            print(f"  Seq Exception: {e}")
            results.append([case_name, "Sequential", "Timeout/Error", 0])

        # 2. MPI (4 ranks)
        cmd = ["mpirun", "-np", "4", "--mca", "mpi_cuda_support", "0", os.path.join(SOURCE_DIR, 'bm_mpi'), pat_file, txt_file]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode == 0:
                m, t = parse_output(res.stdout)
                results.append([case_name, "MPI (4)", t, m])
                print(f"  MPI: {t:.4f}s")
            else:
                print(f"  MPI Error: {res.stderr}")
                results.append([case_name, "MPI (4)", "Error", 0])
        except Exception as e:
            print(f"  MPI Exception: {e}")
            results.append([case_name, "MPI (4)", "Timeout/Error", 0])

        # 3. CUDA
        cmd = [os.path.join(SOURCE_DIR, 'bm_cuda'), pat_file, txt_file]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode == 0:
                m, t = parse_output(res.stdout)
                results.append([case_name, "CUDA", t, m])
                print(f"  CUDA: {t:.4f}s")
            else:
                print(f"  CUDA Error: {res.stderr}")
                results.append([case_name, "CUDA", "Error", 0])
        except Exception as e:
            print(f"  CUDA Exception: {e}")
            results.append([case_name, "CUDA", "Timeout/Error", 0])

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Method", "Time(s)", "Matches"])
        writer.writerows(results)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    compile_code()
    run_experiment()
