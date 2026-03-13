#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "fEI.h"
#include "fEI_old.h"

// 1. Define the function pointer for consistency
typedef double* (*AlgoFunc)(double*, long, long, double);

// Helper to compare double arrays with a small epsilon for precision errors
bool compare_arrays(double* a, double* b, long size) {
    double epsilon = 1e-9;
    for (long i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

void compare_algorithms(AlgoFunc func_v1, AlgoFunc func_v2, 
                        double* original_arr, long size, long boxsize, double mode) {
    
    // Create copies so each function starts with the same data
    double* arr1 = malloc(size * sizeof(double));
    double* arr2 = malloc(size * sizeof(double));
    memcpy(arr1, original_arr, size * sizeof(double));
    memcpy(arr2, original_arr, size * sizeof(double));

    struct timespec start, end;
    double time_v1, time_v2;

    // --- Test Iteration 1 ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    double* result_1 = func_v1(arr1, size, boxsize, mode);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_v1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Result 1 finished\n");

    // --- Test Iteration 2 ---
    clock_gettime(CLOCK_MONOTONIC, &start);
    double* result_2 = func_v2(arr2, size, boxsize, mode);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_v2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Result 2 finished\n");

    // --- Validation and Results ---
    printf("Results for mode %f:\n", mode);
    printf("Iteration 1 Time: %f seconds\n", time_v1);
    printf("Iteration 2 Time: %f seconds\n", time_v2);
    
    if (compare_arrays(result_1, result_2, boxsize)) {
        printf("Verification: SUCCESS (Outputs match)\n");
        printf("Speedup: %.2fx\n", time_v1 / time_v2);
    } else {
        printf("Verification: FAILED (Outputs differ!)\n");
        double epsilon = 1e-9;
        for (long i = 0; i < boxsize; i++) {
            if (fabs(result_1[i] - result_2[i]) > epsilon) printf("In index: %ld, first_result: %f, second result: %f\n", i, result_1[i], result_2[i]);
        }
    }


    free(arr1);
    free(arr2);
}

void validate_single_window(double* seq, long npts, long boxsize, double overlap, long window_j)
{
    double mean_sig = 0;
    for (long i = 0; i < npts; i++)
        mean_sig += seq[i];
    mean_sig /= npts;

    // --- OLD ---
    double* cumsum_old = malloc(npts * sizeof(double));
    cumsum_old[0] = seq[0] - mean_sig;
    for (long i = 1; i < npts; i++)
        cumsum_old[i] = cumsum_old[i-1] + seq[i] - mean_sig;

    double mean_amp_old = 0;
    for (long i = 0; i < boxsize; i++)
        mean_amp_old += seq[window_j + i];
    mean_amp_old /= boxsize;

    double* crt_window = malloc(boxsize * sizeof(double));
    double* x_old      = malloc(boxsize * sizeof(double));
    for (long i = 0; i < boxsize; i++) {
        crt_window[i] = cumsum_old[window_j + i];
        x_old[i]      = i + 1;
    }

    double old_S_y = 0, old_S_y2 = 0, old_S_xy = 0;
    for (long i = 0; i < boxsize; i++) {
        old_S_y  += crt_window[i] / mean_amp_old;
        old_S_y2 += (crt_window[i] / mean_amp_old) * (crt_window[i] / mean_amp_old);
        old_S_xy += x_old[i] * crt_window[i] / mean_amp_old;
    }

    // old bestfit — using long n and fixed S_x cast
    double old_S_x  = (double)boxsize * (boxsize + 1) / 2.0;
    double old_S_x2 = (double)boxsize * (boxsize + 1) * (2 * boxsize + 1) / 6.0;
    double old_m = (boxsize * old_S_xy - old_S_x * old_S_y)
                 / (boxsize * old_S_x2 - old_S_x * old_S_x);
    double old_c = (old_S_y - old_m * old_S_x) / (double)boxsize;

    double old_sse = 0;
    for (long i = 0; i < boxsize; i++) {
        double err = (crt_window[i] / mean_amp_old) - (old_m * x_old[i] + old_c);
        old_sse += err * err;
    }
    double old_mse = sqrt(old_sse / boxsize);

    // --- NEW ---
    double* sum_mem            = malloc((npts+1) * sizeof(double));
    double* cumsum_mem         = malloc((npts+1) * sizeof(double));
    double* square_cumsum_mem  = malloc((npts+1) * sizeof(double));
    double* product_cumsum_mem = malloc((npts+1) * sizeof(double));
    sum_mem[0] = cumsum_mem[0] = square_cumsum_mem[0] = product_cumsum_mem[0] = 0.0;

    double cumsum_curr = 0.0;
    for (long i = 1; i <= npts; i++) {
        cumsum_curr           += seq[i-1] - mean_sig;
        sum_mem[i]            = sum_mem[i-1]            + seq[i-1];
        cumsum_mem[i]         = cumsum_mem[i-1]         + cumsum_curr;
        square_cumsum_mem[i]  = square_cumsum_mem[i-1]  + cumsum_curr * cumsum_curr;
        product_cumsum_mem[i] = product_cumsum_mem[i-1] + i * cumsum_curr;
    }

    long j   = window_j;
    double S_a  = (sum_mem[j + boxsize] - sum_mem[j]) / (double)boxsize;
    double raw  = cumsum_mem[j + boxsize] - cumsum_mem[j];
    double S_y  = raw / S_a;
    double S_y2 = (square_cumsum_mem[j + boxsize]  - square_cumsum_mem[j])  / (S_a * S_a);
    double S_xy = ((product_cumsum_mem[j + boxsize] - product_cumsum_mem[j]) - j * raw) / S_a;

    // fixed S_x with proper cast and long n
    double S_x  = (double)boxsize * (boxsize + 1) / 2.0;
    double S_x2 = (double)boxsize * (boxsize + 1) * (2 * boxsize + 1) / 6.0;
    double denom = (double)boxsize * S_x2 - S_x * S_x;
    double new_m = ((double)boxsize * S_xy - S_x * S_y) / denom;
    double new_c = (S_y - new_m * S_x) / (double)boxsize;
    double new_sse = S_y2 - 2*(new_m*S_xy + S_y*new_c)
                   + new_m*new_m*S_x2 + 2*new_c*new_m*S_x + (double)boxsize*new_c*new_c;
    double new_mse = sqrt(new_sse / boxsize);

    // --- Report ---
    printf("=== Window j=%ld ===\n", window_j);
    printf("mean_amp:  old=%.9f  new=%.9f  %s\n", mean_amp_old, S_a,
        fabs(mean_amp_old - S_a) < 1e-9 ? "OK" : "MISMATCH");
    printf("S_y:       old=%.9f  new=%.9f  %s\n", old_S_y, S_y,
        fabs(old_S_y - S_y) < 1e-9 ? "OK" : "MISMATCH");
    printf("S_y2:      old=%.9f  new=%.9f  %s\n", old_S_y2, S_y2,
        fabs(old_S_y2 - S_y2) < 1e-9 ? "OK" : "MISMATCH");
    printf("S_xy:      old=%.9f  new=%.9f  %s\n", old_S_xy, S_xy,
        fabs(old_S_xy - S_xy) < 1e-9 ? "OK" : "MISMATCH");
    printf("m:         old=%.9f  new=%.9f  %s\n", old_m, new_m,
        fabs(old_m - new_m) < 1e-9 ? "OK" : "MISMATCH");
    printf("c:         old=%.9f  new=%.9f  %s\n", old_c, new_c,
        fabs(old_c - new_c) < 1e-9 ? "OK" : "MISMATCH");
    printf("mse:       old=%.9f  new=%.9f  %s\n", old_mse, new_mse,
        fabs(old_mse - new_mse) < 1e-9 ? "OK" : "MISMATCH");

    free(cumsum_old);
    free(crt_window);
    free(x_old);
    free(sum_mem);
    free(cumsum_mem);
    free(square_cumsum_mem);
    free(product_cumsum_mem);
}

// Helper to generate a random double between a min and max
double rand_double(double min, double max) {
    double range = max - min;
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

long rand_long(long min, long max) {
    return min + rand() % (max - min + 1);
}

void run_test() {
    // 1. Setup sizes
    long array_size = 1000000; // 1 million elements
    long boxsize;
    double mode = 0.5;         // Mode > 0

    // 2. Allocate memory
    double* data = malloc(array_size * sizeof(double));

    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    // 3. Seed the random number generator
    srand((unsigned int)time(NULL));

    // 4. Fill the double array with random values (e.g., -100.0 to 100.0)
    for (long i = 0; i < array_size; i++) {
        data[i] = rand_double(0.0, 1.0);
    }
    boxsize = rand_long(2, array_size / 2);

    printf("Random data generated. Starting comparison...\n");

    //compare_algorithms(fEI_old, fEI, data, array_size, boxsize, mode);

    validate_single_window(data, array_size, boxsize, mode, 0);
    validate_single_window(data, array_size, boxsize, mode, (array_size / boxsize / 2) * boxsize);
    validate_single_window(data, array_size, boxsize, mode, 16487); // first failing index from output
    // Cleanup
    free(data);
}

int main() {
    run_test();
    return 0;
}