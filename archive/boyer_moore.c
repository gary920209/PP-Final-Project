#include "dna_alignment.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define ALPHABET_SIZE 4  // DNA: A, C, G, T

/**
 * Build bad character table for Boyer-Moore algorithm
 * Maps each character to its rightmost position in pattern
 */
static void build_bad_char_table(const char* pattern, size_t pattern_len, int* bad_char) {
    // Initialize all characters to -1 (only need 256 for ASCII)
    for (int i = 0; i < 256; i++) {
        bad_char[i] = -1;
    }
    
    // Fill with rightmost position of each character
    for (size_t i = 0; i < pattern_len; i++) {
        bad_char[(int)pattern[i]] = (int)i;
    }
}

/**
 * Build good suffix table for Boyer-Moore algorithm
 * Simplified version using only basic heuristics
 */
static void build_good_suffix_table(const char* pattern, size_t pattern_len, int* good_suffix) {
    // Initialize all to pattern length (default shift)
    for (size_t i = 0; i <= pattern_len; i++) {
        good_suffix[i] = (int)pattern_len;
    }
    
    // For full match: find longest suffix that is also a prefix
    for (size_t len = pattern_len - 1; len > 0; len--) {
        int match = 1;
        for (size_t i = 0; i < len; i++) {
            if (pattern[i] != pattern[pattern_len - len + i]) {
                match = 0;
                break;
            }
        }
        if (match) {
            good_suffix[0] = (int)(pattern_len - len);
            break;
        }
    }
}

/**
 * Boyer-Moore algorithm for DNA sequence matching
 * Uses bad character and good suffix heuristics for efficient skipping
 */
AlignmentResult* boyer_moore_align(
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
    
    // Allocate tables (max pattern length)
    size_t max_pattern_len = 0;
    for (size_t p = 0; p < num_patterns; p++) {
        if (pattern_lengths[p] > max_pattern_len) {
            max_pattern_len = pattern_lengths[p];
        }
    }
    
    int* bad_char = (int*)malloc(256 * sizeof(int));
    int* good_suffix = (int*)malloc((max_pattern_len + 1) * sizeof(int));
    
    if (!bad_char || !good_suffix) {
        free(bad_char);
        free(good_suffix);
        free(result->coverage);
        free(result);
        return NULL;
    }
    
    // For each pattern
    for (size_t p = 0; p < num_patterns; p++) {
        const char* pattern = patterns[p];
        size_t pattern_len = pattern_lengths[p];
        
        if (pattern_len == 0) continue;
        
        // Build tables for this pattern
        build_bad_char_table(pattern, pattern_len, bad_char);
        build_good_suffix_table(pattern, pattern_len, good_suffix);
        
        // Boyer-Moore search
        size_t i = 0;  // Position in sequence
        while (i <= sequence_len - pattern_len) {
            int j = (int)pattern_len - 1;  // Position in pattern (start from end)
            
            // Match from right to left
            while (j >= 0 && pattern[j] == sequence[i + j]) {
                j--;
            }
            
            // If pattern matched
            if (j < 0) {
                result->num_matches++;
                
                // Update coverage
                for (size_t k = 0; k < pattern_len; k++) {
                    result->coverage[i + k]++;
                }
                
                // Update checksum
                result->checksum += (uint64_t)p * 1000000 + (uint64_t)i;
                
                // Shift: use good suffix rule (or shift by 1 if good_suffix[0] is 0)
                int shift = good_suffix[0];
                if (shift == 0) shift = 1;
                i += shift;
            } else {
                // Mismatch occurred at position j
                // Calculate shift using bad character rule
                int bad_char_pos = bad_char[(int)sequence[i + j]];
                int bad_char_shift = (bad_char_pos < 0) ? j + 1 : j - bad_char_pos;
                if (bad_char_shift < 1) {
                    bad_char_shift = 1;
                }
                
                // Use the bad character shift (simplified - not using full good suffix)
                i += bad_char_shift;
            }
        }
    }
    
    free(bad_char);
    free(good_suffix);
    return result;
}

