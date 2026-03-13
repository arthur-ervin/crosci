// This file is part of crosci, licensed under the Academic Public License.
// See LICENSE.txt for more details.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Function prototypes. */
double* fEI(double* seq, long npts, long boxsize, double overlap);

typedef struct
{
    double m;
    double c;
} BestFitResult;

double* cumsum; /* cumulative sum of signal */

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

/* fE/I
    seq:	    input data array
    npts:	    number of input points
    boxsize:    box size
    overlap:	overlap (number between 0 and 1)
   This function returns the mean squared fluctuations in mse[].
*/

double* fEI_old(double* seq, long npts, long boxsize, double overlap)
{
    long i, inc, j;
    double mean_sig;

    if (overlap > 0)
    {
        inc = floor(boxsize * (1 - overlap));
    }
    else
    {
        inc = boxsize;
    }

    // count number of windows
    int num_W = 0;
    for (i = 0; i < npts - boxsize; i += inc)
    {
        num_W++;
    }

    double* x = (double*)malloc(boxsize * sizeof(double));
    double* cumsum = (double*)malloc(npts * sizeof(double));
    double* mse = (double*)malloc(num_W * 2 * sizeof(double));

    for (i = 0; i < boxsize; i++)
    {
        x[i] = i + 1;
    }

    mean_sig = 0;
    for (i = 0; i < npts; i++)
    {
        mean_sig += seq[i];
    }
    mean_sig /= (double)npts;

    cumsum[0] = seq[0] - mean_sig;
    // write cumulative sum
    for (i = 1; i < npts; i++)
    {
        cumsum[i] = cumsum[i - 1] + seq[i] - mean_sig;
    }

    //    #pragma omp parallel for private(j) schedule(dynamic)
    for (j = 0; j < npts - boxsize; j += inc)
    {
        double mean_amp = 0;
        double* crt_window = (double*)malloc(boxsize * sizeof(double));
        for (int i = 0; i < boxsize; i++)
        {
            mean_amp += seq[j + i];
        }
        mean_amp = mean_amp / boxsize;

        for (int i = 0; i < boxsize; i++)
        {
            crt_window[i] = cumsum[j + i] / mean_amp;
        }

        int crt_win = j / inc;
        BestFitResult bestFitResult = bestFit_old(x, crt_window, boxsize);
        mse[crt_win] = sqrt(sumOfSquaredErrors_old(x, crt_window, boxsize, bestFitResult.m, bestFitResult.c) / boxsize);
        mse[num_W + crt_win] = mean_amp;
        free(crt_window);
    }

    // cleanup all except the vector to return
    free(x);
    free(cumsum);
    return mse;
}
