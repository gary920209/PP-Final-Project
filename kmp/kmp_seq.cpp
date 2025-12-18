// kmp_seq.cpp
// g++ -O3 -std=c++17 kmp_seq.cpp -o kmp_seq
#include <bits/stdc++.h>
using namespace std;

static string read_all_trim(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Failed to open " << path << "\n"; exit(1); }
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static void build_failure_function(const string& pat, vector<int>& failure) {
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

static long long kmp_search_count(const string& text, const string& pat) {
    int n = (int)text.size();
    int m = (int)pat.size();
    if (m == 0 || n < m) return 0;

    vector<int> failure(m);
    build_failure_function(pat, failure);

    long long count = 0;
    int j = 0;
    for (int i = 0; i < n; i++) {
        while (j > 0 && text[i] != pat[j]) {
            j = failure[j - 1];
        }
        if (text[i] == pat[j]) {
            j++;
        }
        if (j == m) {
            count++;
            j = failure[j - 1];
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
    string txt = read_all_trim(argv[2]);

    auto t0 = chrono::high_resolution_clock::now();
    long long matches = kmp_search_count(txt, pat);
    auto t1 = chrono::high_resolution_clock::now();

    double sec = chrono::duration<double>(t1 - t0).count();
    cout << "Matches: " << matches << "\n";
    cout << "Time(s): " << fixed << setprecision(6) << sec << "\n";
    return 0;
}

