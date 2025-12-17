#define _POSIX_C_SOURCE 200809L
#include "dna_alignment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/**
 * Generate a large random DNA sequence
 */
char* generate_dna_sequence(size_t length) {
    const char bases[] = "ACGT";
    char* sequence = (char*)malloc((length + 1) * sizeof(char));
    if (!sequence) return NULL;
    
    // Seed with current time for randomness
    srand((unsigned int)time(NULL));
    
    for (size_t i = 0; i < length; i++) {
        sequence[i] = bases[rand() % 4];
    }
    sequence[length] = '\0';
    
    return sequence;
}

/**
 * Generate random DNA patterns
 */
int generate_patterns(size_t num_patterns, size_t pattern_len, 
                      char*** patterns, size_t** pattern_lengths) {
    const char bases[] = "ACGT";
    *patterns = (char**)malloc(num_patterns * sizeof(char*));
    *pattern_lengths = (size_t*)malloc(num_patterns * sizeof(size_t));
    
    if (!*patterns || !*pattern_lengths) {
        return -1;
    }
    
    for (size_t i = 0; i < num_patterns; i++) {
        (*patterns)[i] = (char*)malloc((pattern_len + 1) * sizeof(char));
        if (!(*patterns)[i]) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                free((*patterns)[j]);
            }
            free(*patterns);
            free(*pattern_lengths);
            return -1;
        }
        
        for (size_t j = 0; j < pattern_len; j++) {
            (*patterns)[i][j] = bases[rand() % 4];
        }
        (*patterns)[i][pattern_len] = '\0';
        (*pattern_lengths)[i] = pattern_len;
    }
    
    return 0;
}

/**
 * Read DNA sequence from file or generate test data
 */
char* read_sequence(const char* filename, size_t* len, size_t default_len) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        // Generate large test sequence
        printf("Generating DNA sequence of length %zu...\n", default_len);
        char* seq = generate_dna_sequence(default_len);
        *len = default_len;
        return seq;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* sequence = (char*)malloc((file_size + 1) * sizeof(char));
    size_t read_size = fread(sequence, 1, file_size, file);
    sequence[read_size] = '\0';
    
    // Remove newlines and whitespace
    size_t j = 0;
    for (size_t i = 0; i < read_size; i++) {
        if (sequence[i] != '\n' && sequence[i] != '\r' && sequence[i] != ' ') {
            sequence[j++] = sequence[i];
        }
    }
    sequence[j] = '\0';
    *len = j;
    
    fclose(file);
    return sequence;
}

/**
 * Read patterns from file or generate test patterns
 */
int read_patterns(const char* filename, char*** patterns, size_t** pattern_lengths, 
                  size_t* num_patterns, size_t default_num, size_t default_pattern_len) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        // Generate test patterns
        printf("Generating %zu patterns of length %zu...\n", default_num, default_pattern_len);
        return generate_patterns(default_num, default_pattern_len, patterns, pattern_lengths);
    }
    
    // Read patterns from file (one per line)
    size_t capacity = 10;
    *patterns = (char**)malloc(capacity * sizeof(char*));
    *pattern_lengths = (size_t*)malloc(capacity * sizeof(size_t));
    *num_patterns = 0;
    
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }
        
        if (len > 0) {
            if (*num_patterns >= capacity) {
                capacity *= 2;
                *patterns = (char**)realloc(*patterns, capacity * sizeof(char*));
                *pattern_lengths = (size_t*)realloc(*pattern_lengths, capacity * sizeof(size_t));
            }
            
            (*patterns)[*num_patterns] = strdup(line);
            (*pattern_lengths)[*num_patterns] = len;
            (*num_patterns)++;
        }
    }
    
    fclose(file);
    return 0;
}

/**
 * Get high-resolution time
 */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/**
 * Print alignment results
 */
void print_results(const char* algorithm_name, AlignmentResult* result, double elapsed_time, int iterations) {
    printf("\n=== %s Algorithm ===\n", algorithm_name);
    printf("Number of matches: %zu\n", result->num_matches);
    printf("Checksum: %lu\n", (unsigned long)result->checksum);
    printf("Total time (%d iterations): %.6f seconds\n", iterations, elapsed_time);
    printf("Average time per iteration: %.6f seconds\n", elapsed_time / iterations);
    printf("Throughput: %.2f MB/s\n", 
           (result->num_matches > 0 ? (double)result->num_matches / elapsed_time : 0.0) / 1000000.0);
}

int main(int argc, char* argv[]) {
    printf("DNA Sequence Alignment - Sequential Implementation\n");
    printf("==================================================\n\n");
    
    // Parse command line arguments
    size_t sequence_len = 1000000;  // Default: 1M characters
    size_t num_patterns = 100;      // Default: 100 patterns
    size_t pattern_len = 20;        // Default: 20 characters per pattern
    int iterations = 5;             // Default: 5 iterations for timing
    
    if (argc > 1) {
        sequence_len = (size_t)atol(argv[1]);
    }
    if (argc > 2) {
        num_patterns = (size_t)atol(argv[2]);
    }
    if (argc > 3) {
        pattern_len = (size_t)atol(argv[3]);
    }
    if (argc > 4) {
        iterations = atoi(argv[4]);
    }
    
    printf("Configuration:\n");
    printf("  Sequence length: %zu characters (~%.2f MB)\n", sequence_len, sequence_len / 1000000.0);
    printf("  Number of patterns: %zu\n", num_patterns);
    printf("  Pattern length: %zu characters\n", pattern_len);
    printf("  Timing iterations: %d\n\n", iterations);
    
    // Read or generate input data
    const char* sequence_file = (argc > 5) ? argv[5] : NULL;
    const char* pattern_file = (argc > 6) ? argv[6] : NULL;
    
    char* sequence = read_sequence(sequence_file, &sequence_len, sequence_len);
    if (!sequence) {
        fprintf(stderr, "Error: Failed to allocate sequence memory\n");
        return 1;
    }
    
    char** patterns;
    size_t* pattern_lengths;
    if (read_patterns(pattern_file, &patterns, &pattern_lengths, &num_patterns, 
                      num_patterns, pattern_len) != 0) {
        fprintf(stderr, "Error: Failed to allocate pattern memory\n");
        free(sequence);
        return 1;
    }
    
    // Convert patterns to const char** for alignment function
    const char** pattern_ptrs = (const char**)patterns;
    
    // Test all algorithms
    AlgorithmType algorithms[] = {
        ALG_BRUTE_FORCE,
        ALG_KMP,
        ALG_BOYER_MOORE,
        ALG_RABIN_KARP
    };
    
    const char* algorithm_names[] = {
        "Brute Force",
        "KMP",
        "Boyer-Moore",
        "Rabin-Karp"
    };
    
    AlignmentResult* results[4] = {NULL, NULL, NULL, NULL};
    double avg_times[4] = {0.0, 0.0, 0.0, 0.0};
    
    // Run each algorithm multiple times for accurate timing
    for (int i = 0; i < 4; i++) {
        // First run (warm-up, don't count)
        AlignmentResult* warmup = align_sequences(sequence, sequence_len, pattern_ptrs, 
                                                  num_patterns, pattern_lengths, algorithms[i]);
        if (warmup) {
            free_alignment_result(warmup);
        }
        
        // Timing runs
        double start_time = get_time();
        for (int iter = 0; iter < iterations; iter++) {
            if (results[i]) {
                free_alignment_result(results[i]);
            }
            results[i] = align_sequences(sequence, sequence_len, pattern_ptrs, 
                                        num_patterns, pattern_lengths, algorithms[i]);
        }
        double end_time = get_time();
        double total_time = end_time - start_time;
        
        if (results[i]) {
            avg_times[i] = total_time / iterations;
            print_results(algorithm_names[i], results[i], total_time, iterations);
        } else {
            printf("\n=== %s Algorithm ===\n", algorithm_names[i]);
            printf("ERROR: Algorithm failed\n");
            avg_times[i] = -1.0;
        }
    }
    
    // Performance summary table
    printf("\n=== Performance Summary ===\n");
    printf("%-15s %15s %15s %15s\n", "Algorithm", "Avg Time (s)", "Speedup", "Relative");
    printf("---------------------------------------------------------------\n");
    
    // Find fastest time for speedup calculation
    double fastest_time = -1.0;
    for (int i = 0; i < 4; i++) {
        if (avg_times[i] > 0 && (fastest_time < 0 || avg_times[i] < fastest_time)) {
            fastest_time = avg_times[i];
        }
    }
    
    for (int i = 0; i < 4; i++) {
        if (avg_times[i] > 0) {
            double speedup = fastest_time / avg_times[i];
            printf("%-15s %15.6f %15.2fx %15.2f%%\n", 
                   algorithm_names[i], avg_times[i], speedup, 
                   (avg_times[i] / avg_times[0]) * 100.0);
        } else {
            printf("%-15s %15s %15s %15s\n", algorithm_names[i], "FAILED", "-", "-");
        }
    }
    
    // Verify correctness: compare all results with brute-force (baseline)
    printf("\n=== Correctness Verification ===\n");
    if (results[0]) {
        for (int i = 1; i < 4; i++) {
            if (results[i]) {
                int match = compare_results(results[0], results[i]);
                printf("%s vs Brute-Force: %s\n", algorithm_names[i], 
                       match ? "PASS" : "FAIL");
                
                if (!match) {
                    printf("  Brute-Force: matches=%zu, checksum=%lu\n",
                           results[0]->num_matches, (unsigned long)results[0]->checksum);
                    printf("  %s: matches=%zu, checksum=%lu\n",
                           algorithm_names[i], results[i]->num_matches, 
                           (unsigned long)results[i]->checksum);
                }
            }
        }
    }
    
    // Cleanup
    free(sequence);
    for (size_t i = 0; i < num_patterns; i++) {
        free(patterns[i]);
    }
    free(patterns);
    free(pattern_lengths);
    
    for (int i = 0; i < 4; i++) {
        if (results[i]) {
            free_alignment_result(results[i]);
        }
    }
    
    return 0;
}

