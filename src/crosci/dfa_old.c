// This file is part of crosci, licensed under the Academic Public License.
// See LICENSE.txt for more details.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
BestFitResult bestFit_old(double* x, double* y, int n)
{
    BestFitResult result;
    double sum_x = sum(x, n);
    double sum_y = sum(y, n);
    double sum_x_sq = sumOfSquare(x, n);
    double sum_xy = sumOfProduct(x, y, n);

    result.m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x);
    result.c = (sum_y - result.m * sum_x) / n;
    return result;
}

// function to calculate the sum of squared errors
double sumOfSquaredErrors_old(double* x, double* y, int n, double m, double c)
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


/* Detrended fluctuation analysis
    seq:	input data array
    npts:	number of input points
    rs:		array of box sizes (uniformly distributed on log scale)
    nr:		number of entries in rs[] and mse[]
    sw:		mode (0: non-overlapping windows, 1: sliding window)
   This function returns the mean squared fluctuations in mse[].
*/
double* dfa_old(double* seq, long npts, long* rs, int nr, double overlap_perc)
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
        
        for (j = 0; j < npts - boxsize; j += inc)
        {
            bestFitResult = bestFit_old(x, seq + j, boxsize);
            local_mse += sqrt(sumOfSquaredErrors_old(x, seq + j, boxsize, bestFitResult.m, bestFitResult.c) / boxsize);
            num_W++;
        }
        printf("\n");
        mse[i] = local_mse / num_W;
    }

    // cleanup
    free(x);

    return mse;
}