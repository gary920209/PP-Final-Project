#include "dna_alignment.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#define BASE 256
#define MOD 1000000007  // Large prime for modulo

/**
 * Compute hash value for a string
 */
static uint64_t compute_hash(const char* str, size_t len) {
    uint64_t hash = 0;
    for (size_t i = 0; i < len; i++) {
        hash = (hash * BASE + (unsigned char)str[i]) % MOD;
    }
    return hash;
}

/**
 * Compute rolling hash: update hash when sliding window moves
 */
static uint64_t update_hash(uint64_t old_hash, char old_char, char new_char, size_t pattern_len, uint64_t base_power) {
    (void)pattern_len;  // Unused parameter, kept for consistency
    // Remove old character contribution
    uint64_t hash = (old_hash + MOD - ((unsigned char)old_char * base_power) % MOD) % MOD;
    // Add new character
    hash = (hash * BASE + (unsigned char)new_char) % MOD;
    return hash;
}

/**
 * Compute BASE^(pattern_len-1) mod MOD for rolling hash
 */
static uint64_t compute_base_power(size_t pattern_len) {
    uint64_t power = 1;
    for (size_t i = 1; i < pattern_len; i++) {
        power = (power * BASE) % MOD;
    }
    return power;
}

/**
 * Rabin-Karp algorithm for DNA sequence matching
 * Uses rolling hash for efficient pattern matching
 */
AlignmentResult* rabin_karp_align(
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
        
        if (pattern_len == 0 || pattern_len > sequence_len) continue;
        
        // Compute pattern hash
        uint64_t pattern_hash = compute_hash(pattern, pattern_len);
        
        // Compute base power for rolling hash
        uint64_t base_power = compute_base_power(pattern_len);
        
        // Compute initial window hash
        uint64_t window_hash = compute_hash(sequence, pattern_len);
        
        // Check first window
        if (window_hash == pattern_hash) {
            // Verify actual match (hash collision check)
            int match = 1;
            for (size_t j = 0; j < pattern_len; j++) {
                if (sequence[j] != pattern[j]) {
                    match = 0;
                    break;
                }
            }
            
            if (match) {
                result->num_matches++;
                for (size_t k = 0; k < pattern_len; k++) {
                    result->coverage[k]++;
                }
                result->checksum += (uint64_t)p * 1000000;
            }
        }
        
        // Slide window through sequence
        for (size_t i = 1; i <= sequence_len - pattern_len; i++) {
            // Update rolling hash
            window_hash = update_hash(window_hash, sequence[i - 1], sequence[i + pattern_len - 1], 
                                     pattern_len, base_power);
            
            // If hashes match, verify actual match
            if (window_hash == pattern_hash) {
                int match = 1;
                for (size_t j = 0; j < pattern_len; j++) {
                    if (sequence[i + j] != pattern[j]) {
                        match = 0;
                        break;
                    }
                }
                
                if (match) {
                    result->num_matches++;
                    for (size_t k = 0; k < pattern_len; k++) {
                        result->coverage[i + k]++;
                    }
                    result->checksum += (uint64_t)p * 1000000 + (uint64_t)i;
                }
            }
        }
    }
    
    return result;
}

