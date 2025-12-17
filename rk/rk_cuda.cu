// rk_cuda.cu
// nvcc -O3 -std=c++17 rk_cuda.cu -o rk_cuda
// Usage: ./rk_cuda pattern.txt text.txt [chunk_bytes] [blocks] [threads_per_block]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using u64 = unsigned long long;
using u128 = __uint128_t;

static std::string read_all_trim(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << path << "\n"; std::exit(1); }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

__device__ inline unsigned char dev_map_char(char c) {
    switch (c) { case 'A': return 1; case 'C': return 2; case 'G': return 3; case 'T': return 4; default: return (unsigned char)c; }
}

__constant__ char d_pat[8192];
__constant__ int d_m;
__constant__ u64 d_ph;
__constant__ u64 d_powB;
__constant__ u64 d_B;

// Device verifies pattern match by checking characters equal
__device__ bool dev_verify_match(const char* text, long long pos, int m) {
    for (int k=0;k<m;k++) {
        if (text[pos + k] != d_pat[k]) return false;
    }
    return true;
}

// kernel: each thread handles multiple starting positions via stride
__global__ void rk_kernel(const char* text, long long chunk_len, long long owned, long long offset_global, long long* out_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    long long cnt = 0;
    int m = d_m;
    u64 B = d_B;
    u64 powB = d_powB;
    // For each starting position assigned to this thread
    for (long long pos = tid; pos < owned; pos += total) {
        // compute hash for window starting at pos
        // compute initial hash by iterating m chars
        u64 h = 0;
        long long base = pos;
        for (int k=0;k<m;k++) {
            h = (u64)((u128)h * B + (u64)dev_map_char(text[base + k]));
        }
        if (h == d_ph) {
            if (dev_verify_match(text, pos, m)) cnt++;
        }
        // Now roll inside this thread for subsequent positions with stride `total` would skip; we don't roll across positions that are not contiguous.
        // For correctness we only compute per-position hash directly; could be optimized to rolling for contiguous positions but complexity increases.
        // Note: this means per-thread cost ~ m * (#positions_assigned). This is acceptable if m is small/medium and owned is large.
    }
    out_counts[tid] = cnt;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <pattern_file> <text_file> [chunk_bytes=256MB] [blocks=120] [threads_per_block=128]\n"; return 1; }
    std::string pat = read_all_trim(argv[1]);
    int m = (int)pat.size();
    if (m <= 0) { std::cerr << "Empty pattern\n"; return 1; }
    if (m > 8192) { std::cerr << "pattern > 8192 not supported in this build\n"; return 1; }

    size_t chunk_bytes = (argc >= 4) ? (size_t)std::stoull(argv[3]) : (size_t)256ull*1024ull*1024ull;
    int blocks = (argc >= 5) ? std::stoi(argv[4]) : 120;
    int threads = (argc >= 6) ? std::stoi(argv[5]) : 128;
    int total_threads = blocks * threads;

    // RK params
    const u64 B = 1315423911ULL;
    u64 powB = 1;
    for (int i=0;i<m-1;i++) powB = (u64)((u128)powB * B);
    u64 ph = 0;
    for (int i=0;i<m;i++) ph = (u64)((u128)ph * B + (u64)((pat[i]=='A')?1: (pat[i]=='C')?2: (pat[i]=='G')?3: (pat[i]=='T')?4 : (unsigned char)pat[i]));

    // Copy pattern and constants to device constant memory
    cudaMemcpyToSymbol(d_pat, pat.data(), m);
    cudaMemcpyToSymbol(d_m, &m, sizeof(int));
    cudaMemcpyToSymbol(d_ph, &ph, sizeof(u64));
    cudaMemcpyToSymbol(d_powB, &powB, sizeof(u64));
    cudaMemcpyToSymbol(d_B, &B, sizeof(u64));

    FILE* f = fopen(argv[2], "rb");
    if (!f) { std::cerr << "Failed to open text file\n"; return 1; }

    size_t overlap = (size_t)(m - 1);
    std::vector<char> hbuf(chunk_bytes + overlap);
    char* d_text = nullptr;
    cudaMalloc(&d_text, chunk_bytes + overlap);
    long long* d_counts = nullptr;
    cudaMalloc(&d_counts, total_threads * sizeof(long long));
    std::vector<long long> h_counts(total_threads);

    long long total_matches = 0;
    size_t carry = 0;
    while (true) {
        if (carry > 0) std::memmove(hbuf.data(), hbuf.data() + chunk_bytes, carry);
        size_t r = fread(hbuf.data() + carry, 1, chunk_bytes, f);
        if (r == 0) break;
        size_t cur_len = carry + r;
        // trim trailing newline if last chunk
        while (cur_len > 0 && (hbuf[cur_len-1] == '\n' || hbuf[cur_len-1] == '\r')) cur_len--;

        if (cur_len < (size_t)m) {
            // prepare carry and continue
            carry = std::min(overlap, cur_len);
            std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);
            continue;
        }

        // Copy to device
        cudaMemcpy(d_text, hbuf.data(), cur_len, cudaMemcpyHostToDevice);

        long long owned = (long long)cur_len - m + 1; // number of start positions in this chunk
        long long offset_global = 0; // we don't track global offset in this simplified CUDA version

        // Launch kernel: each thread calculates hashes for its strided starts
        dim3 grid(blocks), block(threads);
        size_t counts_bytes = total_threads * sizeof(long long);
        cudaMemset(d_counts, 0, counts_bytes);

        rk_kernel<<<grid, block>>>(d_text, (long long)cur_len, owned, offset_global, d_counts);
        cudaMemcpy(h_counts.data(), d_counts, counts_bytes, cudaMemcpyDeviceToHost);

        long long chunk_sum = 0;
        for (int i=0;i<total_threads;i++) chunk_sum += h_counts[i];
        total_matches += chunk_sum;

        // prepare carry
        carry = std::min(overlap, cur_len);
        std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);
    }

    fclose(f);
    cudaFree(d_text);
    cudaFree(d_counts);

    std::cout << "Matches: " << total_matches << "\n";
    return 0;
}
