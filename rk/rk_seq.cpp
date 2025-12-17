// rk_seq.cpp
// g++ -O3 -std=c++17 rk_seq.cpp -o rk_seq
// Usage: ./rk_seq pattern.txt text.txt

#include <bits/stdc++.h>
using namespace std;
using u64 = unsigned long long;
using u128 = __uint128_t;

static string read_all_trim(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Failed to open " << path << "\n"; exit(1); }
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static inline unsigned char map_char(char c) {
    // map A,C,G,T -> 1..4 (non-zero)
    switch (c) {
        case 'A': return 1;
        case 'C': return 2;
        case 'G': return 3;
        case 'T': return 4;
        default: return (unsigned char)c;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { cerr << "Usage: " << argv[0] << " <pattern_file> <text_file>\n"; return 1; }
    string pat = read_all_trim(argv[1]);
    string txt = read_all_trim(argv[2]);

    int m = (int)pat.size();
    long long n = (long long)txt.size();
    if (m == 0 || n < m) {
        cout << "Matches: 0\nTime(s): 0.0\n";
        return 0;
    }

    // RK params: base, operate modulo 2^64 using u64 overflow
    const u64 B = 1315423911ULL; // chosen base
    u64 powB = 1;
    for (int i = 0; i < m-1; ++i) powB = (u64)((u128)powB * B);

    // compute pattern hash
    u64 ph = 0;
    for (int i = 0; i < m; ++i) {
        ph = (u64)((u128)ph * B + (u64)map_char(pat[i]));
    }

    // first window hash
    u64 h = 0;
    for (int i = 0; i < m; ++i) {
        h = (u64)((u128)h * B + (u64)map_char(txt[i]));
    }

    auto t0 = chrono::high_resolution_clock::now();
    long long count = 0;
    if (h == ph) {
        if (memcmp(txt.data(), pat.data(), m) == 0) count++;
    }

    for (long long i = 1; i <= n - m; ++i) {
        // remove leading char: h = (h - txt[i-1]*B^{m-1})*B + txt[i+m-1]
        u64 left = (u64)map_char(txt[i-1]);
        // compute h_minus = h - left * powB
        u64 h_minus = (u64)((u128)h - (u128)left * (u128)powB);
        h = (u64)((u128)h_minus * B + (u64)map_char(txt[i + m - 1]));
        if (h == ph) {
            if (memcmp(txt.data() + i, pat.data(), m) == 0) count++;
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double sec = chrono::duration<double>(t1 - t0).count();
    cout << "Matches: " << count << "\n";
    cout << "Time(s): " << fixed << setprecision(6) << sec << "\n";
    return 0;
}
