// rk_mpi.cpp
// mpicxx -O3 -std=c++17 rk_mpi.cpp -o rk_mpi
// Usage: mpirun -np <ranks> ./rk_mpi pattern.txt text.txt

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;
using u64 = unsigned long long;
using u128 = __uint128_t;

static string read_all_trim_root_bcast(const string& path, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    string s;
    if (rank == 0) {
        ifstream in(path, ios::binary);
        if (!in) { cerr << "Failed to open " << path << "\n"; MPI_Abort(comm,1); }
        s.assign((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    }
    int len = (int)s.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (rank != 0) s.resize(len);
    MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm);
    return s;
}

static inline unsigned char map_char(char c) {
    switch (c) { case 'A': return 1; case 'C': return 2; case 'G': return 3; case 'T': return 4; default: return (unsigned char)c; }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);

    if (argc < 3) {
        if (rank==0) cerr << "Usage: " << argv[0] << " <pattern> <text>\n";
        MPI_Finalize(); return 1;
    }

    string pat = read_all_trim_root_bcast(argv[1], comm);
    int m = (int)pat.size();
    if (m == 0) { if (rank==0) cerr<<"Empty pattern\n"; MPI_Finalize(); return 1; }

    // open file via MPI I/O
    MPI_File fh;
    if (MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank==0) cerr << "Failed to open text file\n";
        MPI_Abort(comm,1);
    }

    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);
    long long N = (long long)file_size;
    // compute per-rank owned region
    long long base = N / size;
    long long rem = N % size;
    long long my_len = base + (rank < rem ? 1 : 0);
    long long my_start = rank * base + min<long long>(rank, rem);
    long long my_end = my_start + my_len - 1;

    long long left_ov = (rank==0) ? 0 : (m - 1);
    long long right_ov = (rank==size-1) ? 0 : (m - 1);

    long long read_start = max(0LL, my_start - left_ov);
    long long read_end = min(N - 1, my_end + right_ov);
    long long read_len = read_end - read_start + 1;

    string local;
    local.resize(read_len);
    MPI_Status st;
    MPI_File_read_at_all(fh, (MPI_Offset)read_start, local.data(), (int)read_len, MPI_CHAR, &st);
    MPI_File_close(&fh);
    while (!local.empty() && (local.back()=='\n' || local.back()=='\r')) local.pop_back();

    // RK parameters
    const u64 B = 1315423911ULL;
    u64 powB = 1;
    for (int i=0;i<m-1;i++) powB = (u64)((u128)powB * B);
    u64 ph = 0;
    for (int i=0;i<m;i++) ph = (u64)((u128)ph * B + (u64)map_char(pat[i]));

    long long local_n = (long long)local.size();
    long long owned_last_start = min(local_n - m, my_end - read_start - 0);
    // But safer: only count matches whose global start in [my_start, my_end]
    long long global_offset = read_start;

    // compute first window hash for local buffer (if enough)
    double t0 = MPI_Wtime();
    long long local_count = 0;
    if (local_n >= m) {
        u64 h = 0;
        for (int i=0;i<m;i++) h = (u64)((u128)h * B + (u64)map_char(local[i]));
        long long global_pos = global_offset + 0;
        if (global_pos >= my_start && global_pos <= my_end) {
            if (h == ph && memcmp(local.data(), pat.data(), m) == 0) local_count++;
        }
        for (long long i=1;i<=local_n - m;i++) {
            u64 left = (u64)map_char(local[i-1]);
            u64 h_minus = (u64)((u128)h - (u128)left * (u128)powB);
            h = (u64)((u128)h_minus * B + (u64)map_char(local[i + m - 1]));
            long long gpos = global_offset + i;
            if (gpos >= my_start && gpos <= my_end) {
                if (h == ph && memcmp(local.data() + i, pat.data(), m) == 0) local_count++;
            }
        }
    }
    double t1 = MPI_Wtime();
    long long total = 0;
    MPI_Reduce(&local_count, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank==0) {
        cout << "Matches: " << total << "\n";
        cout << "Time(s): " << fixed << setprecision(6) << max_time << "\n";
        cout << "Ranks: " << size << "\n";
    }

    MPI_Finalize();
    return 0;
}
