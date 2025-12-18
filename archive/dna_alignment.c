#include "dna_alignment.h"
#include <stdlib.h>
#include <string.h>

// Forward declarations
extern AlignmentResult* brute_force_align(
    const char* sequence, size_t sequence_len,
    const char** patterns, size_t num_patterns, const size_t* pattern_lengths
);

extern AlignmentResult* kmp_align(
    const char* sequence, size_t sequence_len,
    const char** patterns, size_t num_patterns, const size_t* pattern_lengths
);

extern AlignmentResult* boyer_moore_align(
    const char* sequence, size_t sequence_len,
    const char** patterns, size_t num_patterns, const size_t* pattern_lengths
);

extern AlignmentResult* rabin_karp_align(
    const char* sequence, size_t sequence_len,
    const char** patterns, size_t num_patterns, const size_t* pattern_lengths
);

/**
 * Perform DNA sequence alignment using specified algorithm
 */
AlignmentResult* align_sequences(
    const char* sequence,
    size_t sequence_len,
    const char** patterns,
    size_t num_patterns,
    const size_t* pattern_lengths,
    AlgorithmType algorithm
) {
    switch (algorithm) {
        case ALG_BRUTE_FORCE:
            return brute_force_align(sequence, sequence_len, patterns, num_patterns, pattern_lengths);
        case ALG_KMP:
            return kmp_align(sequence, sequence_len, patterns, num_patterns, pattern_lengths);
        case ALG_BOYER_MOORE:
            return boyer_moore_align(sequence, sequence_len, patterns, num_patterns, pattern_lengths);
        case ALG_RABIN_KARP:
            return rabin_karp_align(sequence, sequence_len, patterns, num_patterns, pattern_lengths);
        default:
            return NULL;
    }
}

/**
 * Free alignment result memory
 */
void free_alignment_result(AlignmentResult* result) {
    if (result) {
        if (result->coverage) {
            free(result->coverage);
        }
        free(result);
    }
}

/**
 * Compare two alignment results for correctness verification
 */
int compare_results(const AlignmentResult* result1, const AlignmentResult* result2) {
    if (!result1 || !result2) {
        return 0;
    }
    
    // Compare number of matches
    if (result1->num_matches != result2->num_matches) {
        return 0;
    }
    
    // Compare checksums
    if (result1->checksum != result2->checksum) {
        return 0;
    }
    
    // Note: Coverage arrays should also match, but we'd need sequence_len to compare
    // This is a simplified comparison
    
    return 1;
}

