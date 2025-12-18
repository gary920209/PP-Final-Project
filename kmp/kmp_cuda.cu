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

static void build_failure_function(const std::string& pat, std::vector<int>& failure) {
    int m = (int)pat.size();
    failure[0] = 0;
    int j = 0;
    
    for (int i = 1; i < m; i++) {
        while (j > 0 && pat[i] != pat[j]) {
            j = failure[j - 1];
        }
        if (pat[i] == pat[j]) {
            j++;
        }
        failure[i] = j;
    }
}

__constant__ char d_pat[8192];
__constant__ int d_failure[8192];

__global__ void kmp_segment_kernel(
    const char* __restrict__ text,
    long long text_len,
    int m,
    long long seg_start,
    long long seg_end,
    long long* out_count
) {
    if (threadIdx.x != 0) return;

    long long count = 0;
    long long n = text_len;

    long long start = seg_start;
    long long end = seg_end;

    long long last_start = min(end, n - (long long)m);
    if (start > last_start) {
        out_count[blockIdx.x] = 0;
        return;
    }

    int j = 0;
    for (long long i = start; i <= last_start + (long long)m - 1 && i < n; i++) {
        while (j > 0 && text[i] != d_pat[j]) {
            j = d_failure[j - 1];
        }
        if (text[i] == d_pat[j]) {
            j++;
        }
        if (j == m) {
            long long match_start = i - (long long)m + 1;
            if (match_start >= start && match_start <= last_start) {
                count++;
            }
            j = d_failure[j - 1];
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

    size_t chunk_bytes = (argc >= 4) ? (size_t)std::stoull(argv[3]) : (size_t)512ull*1024ull*1024ull;
    int blocks = (argc >= 5) ? std::stoi(argv[4]) : 120;
    int threads = 64;

    std::vector<int> failure(m);
    build_failure_function(pat, failure);

    cudaMemcpyToSymbol(d_pat, pat.data(), m);
    cudaMemcpyToSymbol(d_failure, failure.data(), m * sizeof(int));

    FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::cerr << "Failed to open text file.\n"; return 1; }

    size_t overlap = (size_t)(m - 1);
    std::vector<char> hbuf(chunk_bytes + overlap);

    char* d_text = nullptr;
    cudaMalloc(&d_text, chunk_bytes + overlap);

    long long* d_counts = nullptr;
    cudaMalloc(&d_counts, blocks * sizeof(long long));
    std::vector<long long> h_counts(blocks);

    long long total_matches = 0;
    long long global_offset = 0;
    size_t carry = 0;

    while (true) {
        if (carry > 0) std::memmove(hbuf.data(), hbuf.data() + (chunk_bytes), carry);

        size_t r = std::fread(hbuf.data() + carry, 1, chunk_bytes, f);
        if (r == 0) break;

        size_t cur_len = carry + r;

        while (cur_len > 0 && (hbuf[cur_len-1] == '\n' || hbuf[cur_len-1] == '\r')) cur_len--;

        cudaMemcpy(d_text, hbuf.data(), cur_len, cudaMemcpyHostToDevice);

        long long n = (long long)cur_len;
        long long owned = (n > m) ? (n - m + 1) : 0;
        if (owned <= 0) {
            carry = std::min(overlap, cur_len);
            std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);
            global_offset += (long long)r;
            continue;
        }

        long long per = (owned + blocks - 1) / blocks;

        dim3 grid(blocks), block(threads);

        std::vector<long long> seg_starts(blocks), seg_ends(blocks);
        for (int i = 0; i < blocks; i++) {
            long long s0 = (long long)i * per;
            long long s1 = std::min(owned - 1, (long long)(i + 1) * per - 1);
            if (s0 > s1) { seg_starts[i] = 1; seg_ends[i] = 0; }
            else { seg_starts[i] = s0; seg_ends[i] = s1; }
        }

        extern __global__ void kmp_chunk_kernel(const char*, long long, int, long long, long long, int, long long*);
        kmp_chunk_kernel<<<grid, block>>>(d_text, n, m, (long long)owned, (long long)per, blocks, d_counts);
        cudaMemcpy(h_counts.data(), d_counts, blocks * sizeof(long long), cudaMemcpyDeviceToHost);

        long long chunk_matches = 0;
        for (int i = 0; i < blocks; i++) chunk_matches += h_counts[i];

        total_matches += chunk_matches;

        carry = std::min(overlap, cur_len);
        std::memcpy(hbuf.data() + chunk_bytes, hbuf.data() + (cur_len - carry), carry);

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

__global__ void kmp_chunk_kernel(
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

    long long last_start = end;

    long long count = 0;
    int j = 0;
    
    for (long long i = start; i <= last_start + (long long)m - 1 && i < text_len; i++) {
        while (j > 0 && text[i] != d_pat[j]) {
            j = d_failure[j - 1];
        }
        if (text[i] == d_pat[j]) {
            j++;
        }
        if (j == m) {
            long long match_start = i - (long long)m + 1;
            if (match_start >= start && match_start <= last_start) {
                count++;
            }
            j = d_failure[j - 1];
        }
    }
    
    out_count[blockIdx.x] = count;
}

