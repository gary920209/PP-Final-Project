#include "dna_alignment.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Build the failure function (prefix table) for KMP algorithm
 */
static void build_failure_function(const char* pattern, size_t pattern_len, int* failure) {
    failure[0] = 0;
    int j = 0;
    
    for (size_t i = 1; i < pattern_len; i++) {
        while (j > 0 && pattern[i] != pattern[j]) {
            j = failure[j - 1];
        }
        if (pattern[i] == pattern[j]) {
            j++;
        }
        failure[i] = j;
    }
}

/**
 * Knuth-Morris-Pratt (KMP) algorithm for DNA sequence matching
 * Uses prefix table to avoid unnecessary comparisons
 */
AlignmentResult* kmp_align(
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
    
    // Allocate failure function array (max pattern length)
    size_t max_pattern_len = 0;
    for (size_t p = 0; p < num_patterns; p++) {
        if (pattern_lengths[p] > max_pattern_len) {
            max_pattern_len = pattern_lengths[p];
        }
    }
    
    int* failure = (int*)malloc(max_pattern_len * sizeof(int));
    if (!failure) {
        free(result->coverage);
        free(result);
        return NULL;
    }
    
    // For each pattern
    for (size_t p = 0; p < num_patterns; p++) {
        const char* pattern = patterns[p];
        size_t pattern_len = pattern_lengths[p];
        
        // Build failure function for this pattern
        build_failure_function(pattern, pattern_len, failure);
        
        // KMP search
        size_t j = 0;  // Position in pattern
        for (size_t i = 0; i < sequence_len; i++) {
            // While mismatch, use failure function to skip
            while (j > 0 && sequence[i] != pattern[j]) {
                j = failure[j - 1];
            }
            
            // If characters match, advance in pattern
            if (sequence[i] == pattern[j]) {
                j++;
            }
            
            // If entire pattern matched
            if (j == pattern_len) {
                size_t match_pos = i - pattern_len + 1;
                result->num_matches++;
                
                // Update coverage
                for (size_t k = 0; k < pattern_len; k++) {
                    result->coverage[match_pos + k]++;
                }
                
                // Update checksum
                result->checksum += (uint64_t)p * 1000000 + (uint64_t)match_pos;
                
                // Continue searching: use failure function to find next potential match
                j = failure[j - 1];
            }
        }
    }
    
    free(failure);
    return result;
}

