// bm_cuda.cu
// nvcc -O3 -std=c++17 bm_cuda.cu -o bm_cuda
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

static std::string read_all_trim(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << path << "\n"; std::exit(1); }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

// -------- BM tables on host (copied to device) --------
static void build_bad_char(const std::string& pat, int bc[256]) {
    for (int i=0;i<256;i++) bc[i] = -1;
    for (int i=0;i<(int)pat.size();i++) bc[(unsigned char)pat[i]] = i;
}

static void preprocess_strong_suffix(const std::string& pat, std::vector<int>& shift, std::vector<int>& bpos) {
    int m = (int)pat.size();
    int i = m, j = m + 1;
    bpos[i] = j;
    while (i > 0) {
        while (j <= m && pat[i - 1] != pat[j - 1]) {
            if (shift[j] == 0) shift[j] = j - i;
            j = bpos[j];
        }
        i--; j--;
        bpos[i] = j;
    }
}
static void preprocess_case2(std::vector<int>& shift, const std::vector<int>& bpos) {
    int m = (int)bpos.size() - 1;
    int j = bpos[0];
    for (int i = 0; i <= m; i++) {
        if (shift[i] == 0) shift[i] = j;
        if (i == j) j = bpos[j];
    }
}

// Device constant memory for small tables
__constant__ int d_bc[256];
__constant__ int d_shift[8192+1]; // adjust if you need >8192 pattern
__constant__ char d_pat[8192];    // adjust if you need >8192 pattern
__device__ __forceinline__ int d_max(int a,int b){return a>b?a:b;}

// Each block processes one segment (coarse-grain BM).
__global__ void bm_segment_kernel(
    const char* __restrict__ text,
    long long text_len,
    int m,
    long long seg_start,
    long long seg_end,   // inclusive, within [0, text_len-1]
    long long* out_count // one per block
) {
    // 1 thread per block does BM sequentially on its segment
    if (threadIdx.x != 0) return;

    long long count = 0;
    long long n = text_len;

    // segment bounds in [0, n-1]
    long long start = seg_start;
    long long end = seg_end;

    // We search positions whose start is in [start, end] but pattern must fit.
    long long last_start = min(end, n - (long long)m);
    long long s = start;

    while (s <= last_start) {
        int j = m - 1;
        while (j >= 0 && d_pat[j] == text[s + j]) j--;

        if (j < 0) {
            count++;
            s += d_shift[0];
        } else {
            int bc_shift = j - d_bc[(unsigned char)text[s + j]];
            int gs_shift = d_shift[j + 1];
            long long adv = (long long)d_max(1, d_max(bc_shift, gs_shift));
            s += adv;
        }
    }

    out_count[blockIdx.x] = count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <pattern_file> <text_file> [chunk_bytes] [blocks]\n";
        return 1;
    }

    std::string pat = read_all_trim(argv[1]);
    int m = (int)pat.size();
    if (m <= 0) { std::cerr << "Empty pattern.\n"; return 1; }
    if (m > 8192) {
        std::cerr << "Pattern too long for this build (max 8192). Increase constants in code.\n";
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    size_t chunk_bytes = (argc >= 4) ? (size_t)std::stoull(argv[3]) : (size_t)512ull*1024ull*1024ull; // default 512MB
    int blocks = (argc >= 5) ? std::stoi(argv[4]) : 120; // tune to your GPU
    int threads = 64; // unused mostly; only thread0 runs per block

    // Build BM tables
    int bc[256];
    build_bad_char(pat, bc);
    std::vector<int> shift(m + 1, 0), bpos(m + 1, 0);
    preprocess_strong_suffix(pat, shift, bpos);
    preprocess_case2(shift, bpos);

    // Copy to device constant
    cudaMemcpyToSymbol(d_bc, bc, sizeof(bc));
    cudaMemcpyToSymbol(d_pat, pat.data(), m);
    cudaMemcpyToSymbol(d_shift, shift.data(), (m+1)*sizeof(int));

    // Open text file for streaming
    FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::cerr << "Failed to open text file.\n"; return 1; }

    // Allocate host buffer with overlap space
    // We keep (m-1) bytes from the end of previous chunk to not miss boundary matches.
    size_t overlap = (size_t)(m - 1);
    std::vector<char> hbuf(chunk_bytes + overlap);

    // Device buffers
    char* d_text = nullptr;
    cudaMalloc(&d_text, chunk_bytes + overlap);

    long long* d_counts = nullptr;
    cudaMalloc(&d_counts, blocks * sizeof(long long));
    std::vector<long long> h_counts(blocks);

    long long total_matches = 0;
    long long global_offset = 0;     // global index of hbuf[0] in the whole file
    size_t carry = 0;                // number of overlap bytes carried to next chunk

    while (true) {
        // Move overlap from previous iteration to start of buffer
        if (carry > 0) std::memmove(hbuf.data(), hbuf.data() + (chunk_bytes), carry);

        // Read next chunk into hbuf[carry ... carry+chunk_bytes-1]
        size_t r = std::fread(hbuf.data() + carry, 1, chunk_bytes, f);
        if (r == 0) break;

        size_t cur_len = carry + r;

        // Trim trailing newlines in the very last chunk if present (optional)
        while (cur_len > 0 && (hbuf[cur_len-1] == '\n' || hbuf[cur_len-1] == '\r')) cur_len--;

        // Copy to device
        cudaMemcpy(d_text, hbuf.data(), cur_len, cudaMemcpyHostToDevice);

        // Partition this chunk into segments for blocks
        // Each block gets an owned range of starting positions; add overlap between segments to avoid missing.
        long long n = (long long)cur_len;
        long long owned = (n > m) ? (n - m + 1) : 0;
        if (owned <= 0) {
            // Prepare carry and continue
            carry = std::min(overlap, cur_len);
            // copy tail to the end area for next memmove trick
            std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);
            global_offset += (long long)r;
            continue;
        }

        long long per = (owned + blocks - 1) / blocks;

        // Launch: each block sequential BM on its segment [seg_start, seg_end]
        // Note: seg_start/seg_end are indices within the chunk buffer.
        dim3 grid(blocks), block(threads);

        // We store counts per block then sum on host.
        // Ensure seg_end includes overlap (m-1) so BM inside a segment can see full pattern.
        // But we only count matches whose start is within the owned start range of that block implicitly by segment bounds.
        // Here: each block owns [i*per, min((i+1)*per-1, owned-1)] as starting positions.
        // Convert to segment text index range: [start, end + (m-1)].
        // Kernel itself only checks starts within [start, end] and pattern fit.
        std::vector<long long> seg_starts(blocks), seg_ends(blocks);
        for (int i = 0; i < blocks; i++) {
            long long s0 = (long long)i * per;
            long long s1 = std::min(owned - 1, (long long)(i + 1) * per - 1);
            if (s0 > s1) { seg_starts[i] = 1; seg_ends[i] = 0; } // empty
            else { seg_starts[i] = s0; seg_ends[i] = s1; }
        }

        // Copy segment ranges via kernel parameters (we’ll launch one kernel per block range using blockIdx)
        // Simpler: launch one kernel and compute inside by blockIdx from per.
        // For simplicity and speed, we recompute inside kernel by per, owned, etc.
        // So we call a second kernel that computes its own segment.
        // (To keep code short, we re-launch a specialized kernel by embedding per/owned.)
        // Instead, easiest: call bm_segment_kernel in a loop? Too slow.
        // We do one kernel launch with per and owned parameters by using lambda-like approach: not possible in CUDA C easily.
        // So: we accept a mild overhead and launch 1 kernel per block range using cudaLaunchCooperativeKernel? too complex.
        //
        // Practical compromise: run 1 kernel launch per chunk but inside kernel derive its segment:
        // We'll re-use bm_segment_kernel by calling it blocks times? Not good.
        //
        // -> We'll do the derivation in bm_segment_kernel by passing per and owned and let each block compute its start/end.
        //
        // So we need a second kernel. We'll include it now:
        //
        // (See below: bm_chunk_kernel)
        ;

        // Define and launch bm_chunk_kernel
        // (We can't define kernels inside main in CUDA. So we emulate by using a separate kernel below.)
        // To keep this file single, bm_chunk_kernel is defined after main, but we must forward declare.
        extern __global__ void bm_chunk_kernel(const char*, long long, int, long long, long long, int, long long*);
        bm_chunk_kernel<<<grid, block>>>(d_text, n, m, (long long)owned, (long long)per, blocks, d_counts);
        cudaMemcpy(h_counts.data(), d_counts, blocks * sizeof(long long), cudaMemcpyDeviceToHost);

        long long chunk_matches = 0;
        for (int i = 0; i < blocks; i++) chunk_matches += h_counts[i];

        total_matches += chunk_matches;

        // Prepare carry overlap for next iteration: keep last (m-1) bytes from this chunk
        carry = std::min(overlap, cur_len);
        std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);

        // Advance file/global offset by bytes read (r), not by trimmed bytes
        global_offset += (long long)r;
    }

    std::fclose(f);
    cudaFree(d_text);
    cudaFree(d_counts);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Matches: " << total_matches << "\n";
    std::cout << "Time(s): " << std::fixed << std::setprecision(6) << sec << "\n";
    return 0;
}

// Kernel that derives per-block segment and runs BM on that segment.
// Each block owns start positions [b*per, min((b+1)*per-1, owned-1)].
__global__ void bm_chunk_kernel(
    const char* __restrict__ text,
    long long text_len,
    int m,
    long long owned_starts,
    long long per,
    int blocks,
    long long* out_count
) {
    if (blockIdx.x >= blocks) return;
    if (threadIdx.x != 0) return;

    long long b = (long long)blockIdx.x;
    long long start = b * per;
    long long end = min(owned_starts - 1, (b + 1) * per - 1);
    if (start > end) { out_count[blockIdx.x] = 0; return; }

    long long s = start;
    long long last_start = end; // restrict starts to owned range; pattern fit ensured since start <= owned-1

    long long count = 0;
    while (s <= last_start) {
        int j = m - 1;
        while (j >= 0 && d_pat[j] == text[s + j]) j--;
        if (j < 0) {
            count++;
            s += d_shift[0];
        } else {
            int bc_shift = j - d_bc[(unsigned char)text[s + j]];
            int gs_shift = d_shift[j + 1];
            long long adv = (long long)d_max(1, d_max(bc_shift, gs_shift));
            s += adv;
        }
    }
    out_count[blockIdx.x] = count;
}
