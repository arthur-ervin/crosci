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

// function to calculate the best fit
BestFitResult bestFit(double S_y, double S_xy, long n)
{
    BestFitResult result;
    double S_x = (double)n*(n+1)/2.0;
    double S_x2 = (double)n * (n + 1) * (2 * n + 1) / 6.0;

    double denom = (n * S_x2 - S_x * S_x);
    result.m = (n * S_xy - S_x * S_y) / denom;
    result.c = (S_y - result.m * S_x) / (double)n;

    return result;
}

// function to calculate the sum of squared errors
double sumOfSquaredErrors(double S_y, double S_y2, double S_xy, long n, double m, double c)
{
    double S_x = (double)n * (n + 1) / 2.0;
    double S_x2 = (double)n * (n + 1) * (2 * n + 1) / 6.0;

    double error = S_y2 - 2*(m*S_xy + S_y*c);
    error += m*m*S_x2 + 2*c*m*S_x + (double)n*c*c;
    return error;
}

/* fE/I
    seq:	    input data array
    npts:	    number of input points
    boxsize:    box size
    overlap:	overlap (number between 0 and 1)
   This function returns the mean squared fluctuations in mse[].
*/

double* fEI(double* seq, long npts, long boxsize, double overlap)
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

    double* mse = (double*)malloc(num_W * 2 * sizeof(double));

    //Arrays are 1-indexed for easier initialization and DP implementation.
    double* cumsum_mem = (double*)malloc((npts+1) * sizeof(double));
    double* sum_mem = malloc((npts+1) * sizeof(double));
    double* square_cumsum_mem = malloc((npts+1) * sizeof(double));
    double* product_cumsum_mem = malloc((npts+1) * sizeof(double));
    sum_mem[0], square_cumsum_mem[0], product_cumsum_mem[0], cumsum_mem[0] = 0.0;

    mean_sig = 0.0;
    for (i = 0; i < npts; i++)
    {
        mean_sig += seq[i];
    }
    mean_sig /= (double)npts;

    // TODO: Duplicated code segment here. 
    // Make a utility class and put initialization and best fit and other stuff there
    double cumsum_curr = 0.0;
    for(int i = 1; i <= npts; i++){
        cumsum_curr += seq[i-1] - mean_sig;
        sum_mem[i] = seq[i-1] + sum_mem[i-1];
        square_cumsum_mem[i] = cumsum_curr*cumsum_curr + square_cumsum_mem[i-1];
        product_cumsum_mem[i] = product_cumsum_mem[i-1] + i*cumsum_curr;
        cumsum_mem[i] = (cumsum_mem[i-1] + cumsum_curr); 
    }

    //    #pragma omp parallel for private(j) schedule(dynamic)
    for (j = 0; j < npts - boxsize; j += inc)
    {
        int crt_win = j / inc;

        double S_a = (sum_mem[j + boxsize] - sum_mem[j]) / (double)boxsize;
        double S_y = (cumsum_mem[j + boxsize] - cumsum_mem[j]) / S_a;
        double S_y2 = (square_cumsum_mem[j + boxsize] - square_cumsum_mem[j]) / (S_a*S_a);
        double S_xy = (product_cumsum_mem[j + boxsize] - product_cumsum_mem[j]) / S_a;
        S_xy -= (double)j * (cumsum_mem[j + boxsize] - cumsum_mem[j]) / S_a;

        BestFitResult bestFitResult = bestFit(S_y, 
                S_xy, 
                boxsize);
        mse[crt_win] = sqrt(sumOfSquaredErrors(S_y, 
                S_y2, 
                S_xy, 
                boxsize, 
                bestFitResult.m, 
                bestFitResult.c) / boxsize);
        mse[num_W + crt_win] = S_a;
    }

    free(cumsum_mem);
    free(sum_mem);
    free(square_cumsum_mem);
    free(product_cumsum_mem);
    return mse;
}
