// bm_seq.cpp
// g++ -O3 -std=c++17 bm_seq.cpp -o bm_seq
#include <bits/stdc++.h>
using namespace std;

static string read_all_trim(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Failed to open " << path << "\n"; exit(1); }
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static void build_bad_char(const string& pat, array<int,256>& bc) {
    bc.fill(-1);
    for (int i = 0; i < (int)pat.size(); i++) bc[(unsigned char)pat[i]] = i;
}

// Standard good-suffix preprocessing (BM)
static void preprocess_strong_suffix(const string& pat, vector<int>& shift, vector<int>& bpos) {
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

static void preprocess_case2(vector<int>& shift, const vector<int>& bpos) {
    int m = (int)bpos.size() - 1;
    int j = bpos[0];
    for (int i = 0; i <= m; i++) {
        if (shift[i] == 0) shift[i] = j;
        if (i == j) j = bpos[j];
    }
}

static long long bm_search_count(const string& text, const string& pat) {
    int n = (int)text.size();
    int m = (int)pat.size();
    if (m == 0 || n < m) return 0;

    array<int,256> bc;
    build_bad_char(pat, bc);

    vector<int> shift(m + 1, 0), bpos(m + 1, 0);
    preprocess_strong_suffix(pat, shift, bpos);
    preprocess_case2(shift, bpos);

    long long count = 0;
    int s = 0; // shift of pattern w.r.t text
    while (s <= n - m) {
        int j = m - 1;
        while (j >= 0 && pat[j] == text[s + j]) j--;

        if (j < 0) {
            count++;
            s += shift[0];
        } else {
            int bc_shift = j - bc[(unsigned char)text[s + j]];
            int gs_shift = shift[j + 1];
            s += max(1, max(bc_shift, gs_shift));
        }
    }
    return count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <pattern_file> <text_file>\n";
        return 1;
    }
    string pat = read_all_trim(argv[1]);
    string txt = read_all_trim(argv[2]); // for huge texts, this is not feasible; use MPI/CUDA streaming versions.

    auto t0 = chrono::high_resolution_clock::now();
    long long matches = bm_search_count(txt, pat);
    auto t1 = chrono::high_resolution_clock::now();

    double sec = chrono::duration<double>(t1 - t0).count();
    cout << "Matches: " << matches << "\n";
    cout << "Time(s): " << fixed << setprecision(6) << sec << "\n";
    return 0;
}
