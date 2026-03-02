#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "dfa.h"
#include "dfa_old.h"

// 1. Define the function pointer for consistency
typedef double* (*AlgoFunc)(double*, long, long*, int, double);

// Helper to compare double arrays with a small epsilon for precision errors
bool compare_arrays(double* a, double* b, long size) {
    double epsilon = 1e-9;
    for (long i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

void compare_algorithms(AlgoFunc func_v1, AlgoFunc func_v2, 
                        double* original_arr, long size, 
                        long* indices, int idx_size, double mode) {
    
    // Create copies so each function starts with the same data
    double* arr1 = malloc(size * sizeof(double));
    double* arr2 = malloc(size * sizeof(double));
    memcpy(arr1, original_arr, size * sizeof(double));
    memcpy(arr2, original_arr, size * sizeof(double));

    struct timespec start, end;
    double time_v1, time_v2;

    // --- Test Iteration 1 ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    double* result_1 = func_v1(arr1, size, indices, idx_size, mode);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_v1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Result 1 finished\n");

    // --- Test Iteration 2 ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    double* result_2 = func_v2(arr2, size, indices, idx_size, mode);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_v2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Result 2 finished\n");

    // --- Validation and Results ---
    printf("Results for mode %f:\n", mode);
    printf("Iteration 1 Time: %f seconds\n", time_v1);
    printf("Iteration 2 Time: %f seconds\n", time_v2);
    
    if (compare_arrays(result_1, result_2, idx_size)) {
        printf("Verification: SUCCESS (Outputs match)\n");
        printf("Speedup: %.2fx\n", time_v1 / time_v2);
    } else {
        printf("Verification: FAILED (Outputs differ!)\n");
        double epsilon = 1e-9;
        for (long i = 0; i < idx_size; i++) {
            if (fabs(result_1[i] - result_2[i]) > epsilon) printf("In index: %ld, first_result: %f, second result: %f\n", i, result_1[i], result_2[i]);
        }
    }


    free(arr1);
    free(arr2);
}

// Helper to generate a random double between a min and max
double rand_double(double min, double max) {
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int comp(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

void run_test() {
    // 1. Setup sizes
    long array_size = 1000000; // 1 million elements
    int index_size = 10;      // 500 random indices
    double mode = 0.5;         // Mode > 0

    // 2. Allocate memory
    double* data = malloc(array_size * sizeof(double));
    long* indices = malloc(index_size * sizeof(long));

    if (!data || !indices) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    // 3. Seed the random number generator
    srand((unsigned int)time(NULL));

    // 4. Fill the double array with random values (e.g., -100.0 to 100.0)
    for (long i = 0; i < array_size; i++) {
        data[i] = rand_double(0.0, 1);
    }

    // 5. Fill the long array with values smaller than array_size
    for (int i = 0; i < index_size; i++) {
        indices[i] = rand() % array_size;
    }

    qsort(indices, index_size, sizeof(long), comp);

    printf("Random data generated. Starting comparison...\n");

    compare_algorithms(dfa_old, dfa, data, array_size, indices, index_size, mode);

    // Cleanup
    free(data);
    free(indices);
}

int main() {
    run_test();
    return 0;
}