#include "dna_alignment.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Brute-force DNA sequence matching
 * Simple pattern matching by checking every position
 */
AlignmentResult* brute_force_align(
    const char* sequence,
    size_t sequence_len,
    const char** patterns,
    size_t num_patterns,
    const size_t* pattern_lengths
) {
    AlignmentResult* result = (AlignmentResult*)malloc(sizeof(AlignmentResult));
    if (!result) {
        return NULL;
    }
    
    // Initialize coverage array
    result->coverage = (int*)calloc(sequence_len, sizeof(int));
    if (!result->coverage) {
        free(result);
        return NULL;
    }
    
    result->num_matches = 0;
    result->checksum = 0;
    
    // For each pattern
    for (size_t p = 0; p < num_patterns; p++) {
        const char* pattern = patterns[p];
        size_t pattern_len = pattern_lengths[p];
        
        // Check every possible starting position in sequence
        for (size_t i = 0; i <= sequence_len - pattern_len; i++) {
            int match = 1;
            
            // Compare pattern with sequence at position i
            for (size_t j = 0; j < pattern_len; j++) {
                if (sequence[i + j] != pattern[j]) {
                    match = 0;
                    break;
                }
            }
            
            if (match) {
                // Pattern found at position i
                result->num_matches++;
                
                // Update coverage for each position in the match
                for (size_t j = 0; j < pattern_len; j++) {
                    result->coverage[i + j]++;
                }
                
                // Update checksum: add pattern index and position
                result->checksum += (uint64_t)p * 1000000 + (uint64_t)i;
            }
        }
    }
    
    return result;
}

