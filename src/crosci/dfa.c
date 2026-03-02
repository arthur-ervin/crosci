// This file is part of crosci, licensed under the Academic Public License.
// See LICENSE.txt for more details.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Function prototypes. */
double* dfa(double* seq, long npts, long* rs, int nr, double overlap_perc);

typedef struct
{
    double m;
    double c;
} BestFitResult;

// function to find the sum of a given array
double sum(double* arr, int n)
{
    double sum = 0;
    int i;
    for (i = 0; i < n; i++)
        sum += arr[i];
    return sum;
}

// function to find the product of two given arrays
double sumOfProduct(double* arr1, double* arr2, int n)
{
    double sum = 0;
    int i;
    for (i = 0; i < n; i++)
        sum += arr1[i] * arr2[i];
    return sum;
}

// function to find the square of a given array
double sumOfSquare(double* arr, int n)
{
    double sum = 0;
    int i;
    for (i = 0; i < n; i++)
        sum += arr[i] * arr[i];
    return sum;
}

// function to calculate the best fit
BestFitResult bestFit(double* x, double* y, double* sum_mem, double* square_sum_mem, double* product_sum_mem, int n)
{
    BestFitResult result;
    double sum_x = (n*(n+1))/2;
    double sum_y = sum_mem[n];
    double sum_x_sq = square_sum_mem[n];
    double sum_xy = product_sum_mem[n];

    result.m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x);
    result.c = (sum_y - result.m * sum_x) / n;
    return result;
}

// function to calculate the sum of squared errors
double sumOfSquaredErrors(double* x, double* y, int n, double m, double c)
{
    double sum = 0;
    int i;
    for (i = 0; i < n; i++)
    {
        double error = y[i] - (m * x[i] + c);
        sum += error * error;
    }
    return sum;
}

int main()
{
    return 0;
}

/* Detrended fluctuation analysis
    seq:	input data array
    npts:	number of input points
    rs:		array of box sizes (uniformly distributed on log scale)
    nr:		number of entries in rs[] and mse[]
    sw:		mode (0: non-overlapping windows, 1: sliding window)
   This function returns the mean squared fluctuations in mse[].
*/
double* dfa(double* seq, long npts, long* rs, int nr, double overlap_perc)
{
    long i, boxsize, inc, j;

    // write cumulative sum
    for (i = 1; i < npts; i++)
    {
        seq[i] = seq[i - 1] + seq[i];
    }

    long largest_window_size = rs[nr - 1];

    double* mse = (double*)malloc(nr * sizeof(double));
    double* x = (double*)malloc(largest_window_size * sizeof(double));

    for (i = 0; i < largest_window_size; i++)
    {
        x[i] = i + 1;
    }

    double* sum_mem = malloc((nr+1) * sizeof(double));
    double* square_sum_mem = malloc((nr+1) * sizeof(double));
    double* product_sum_mem = malloc((nr+1) * sizeof(double));

    for(int i = 1; i <= npts; i++){
        sum_mem[i] = seq[i] + sum_mem[i+1];
        square_sum_mem[i] = seq[i]*seq[i] + square_sum_mem[i+1];
        product_sum_mem[i] += product_sum_mem[i+1];
        for(int j = 1; j<=i; j++){
            product_sum_mem[i] += seq[j];
        }
    }

    int num_W = 0;
    double local_mse = 0.0;
    BestFitResult bestFitResult;

    for (i = 0; i < nr; i++)
    {
        boxsize = rs[i];
        if (overlap_perc > 0)
        {
            inc = floor(boxsize * (1 - overlap_perc));
        }
        else
        {
            inc = boxsize;
        }

        num_W = 0;
        local_mse = 0.0;

#pragma omp parallel for reduction(+ : local_mse, num_W) private(bestFitResult)
        for (j = 0; j < npts - boxsize; j += inc)
        {
            bestFitResult = bestFit(x, seq + j, sum_mem, square_sum_mem, product_sum_mem, boxsize);
            local_mse += sqrt(sumOfSquaredErrors(x, seq + j, boxsize, bestFitResult.m, bestFitResult.c) / boxsize);
            num_W++;
        }
        mse[i] = local_mse / num_W;
    }

    // cleanup
    free(x);

    return mse;
}
