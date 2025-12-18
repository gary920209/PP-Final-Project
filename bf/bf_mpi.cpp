// bf_mpi.cpp
// mpicxx -O3 -std=c++17 bf_mpi.cpp -o bf_mpi
#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

static string read_all_trim_root_bcast(const string& path, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    string s;
    if (rank == 0) {
        ifstream in(path, ios::binary);
        if (!in) { cerr << "Failed to open " << path << "\n"; MPI_Abort(comm, 1); }
        s.assign((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    }
    int len = (int)s.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (rank != 0) s.resize(len);
    MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm);
    return s;
}

// Count matches whose starting position falls within [owner_start, owner_end] (inclusive) in GLOBAL text index.
static long long bf_search_count_owned(
    const string& local_text,
    long long global_local_start,
    const string& pat,
    long long owner_start,
    long long owner_end
) {
    int n = (int)local_text.size();
    int m = (int)pat.size();
    if (m == 0 || n < m) return 0;

    long long count = 0;
    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && local_text[i + j] == pat[j]) j++;
        
        if (j == m) {
            long long match_pos_global = global_local_start + i;
            if (match_pos_global >= owner_start && match_pos_global <= owner_end) count++;
        }
    }
    return count;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 3) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " <pattern_file> <text_file>\n";
        MPI_Finalize();
        return 1;
    }

    string pat = read_all_trim_root_bcast(argv[1], comm);
    int m = (int)pat.size();
    if (m == 0) {
        if (rank == 0) cerr << "Empty pattern.\n";
        MPI_Finalize();
        return 1;
    }

    // Parallel file read with overlap
    MPI_File fh;
    if (MPI_File_open(comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) cerr << "Failed to open text file.\n";
        MPI_Abort(comm, 1);
    }

    MPI_Offset file_size;
    MPI_File_get_size(fh, &file_size);

    long long N = (long long)file_size;

    long long base = N / size;
    long long rem  = N % size;
    long long my_len = base + (rank < rem ? 1 : 0);
    long long my_start = rank * base + min<long long>(rank, rem);
    long long my_end = my_start + my_len - 1;

    // overlap (m-1) on both sides
    long long left_ov  = (rank == 0) ? 0 : (m - 1);
    long long right_ov = (rank == size - 1) ? 0 : (m - 1);

    long long read_start = max(0LL, my_start - left_ov);
    long long read_end   = min(N - 1, my_end + right_ov);
    long long read_len   = read_end - read_start + 1;

    string local(read_len, '\0');
    MPI_Status st;
    MPI_File_read_at_all(fh, (MPI_Offset)read_start, local.data(), (int)read_len, MPI_CHAR, &st);
    MPI_File_close(&fh);

    while (!local.empty() && (local.back()=='\n' || local.back()=='\r')) local.pop_back();

    double t0 = MPI_Wtime();
    long long local_count = bf_search_count_owned(local, read_start, pat, my_start, my_end);
    double t1 = MPI_Wtime();

    long long total = 0;
    MPI_Reduce(&local_count, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        cout << "Matches: " << total << "\n";
        cout << "Time(s): " << fixed << setprecision(6) << max_time << "\n";
        cout << "Ranks: " << size << "\n";
    }

    MPI_Finalize();
    return 0;
}
