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

// function to calculate the best fit
BestFitResult bestFit(double S_y, double S_xy, int n)
{
    BestFitResult result;
    double S_x = (n*(n+1))/2.0;
    double S_x2 = (double)n * (n + 1) * (2 * n + 1) / 6.0;

    double denom = (n * S_x2 - S_x * S_x);
    result.m = (n * S_xy - S_x * S_y) / denom;
    result.c = (S_y - result.m * S_x) / n;

    return result;
}

// function to calculate the sum of squared errors
double sumOfSquaredErrors(double S_y, double S_y2, double S_xy, int n, double m, double c)
{
    double S_x = (double)n * (n + 1) / 2.0;
    double S_x2 = (double)n * (n + 1) * (2 * n + 1) / 6.0;

    double error = S_y2 - 2*(m*S_xy + S_y*c);
    error += m*m*S_x2 + 2*c*m*S_x + (double)n*c*c;
    return error;
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

    double* sum_mem = malloc((npts+1) * sizeof(double));
    double* square_sum_mem = malloc((npts+1) * sizeof(double));
    double* product_sum_mem = malloc((npts+1) * sizeof(double));

    sum_mem[0] = 0;
    square_sum_mem[0] = 0;
    product_sum_mem[0] = 0;

    for(int i = 1; i <= npts; i++){
        sum_mem[i] = seq[i-1] + sum_mem[i-1];
        square_sum_mem[i] = seq[i-1]*seq[i-1] + square_sum_mem[i-1];
        product_sum_mem[i] = product_sum_mem[i-1] + i*seq[i-1];
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
            double S_y = sum_mem[j+boxsize] - sum_mem[j];
            double S_y2 = square_sum_mem[j+boxsize] - square_sum_mem[j];
            double S_xy_global = product_sum_mem[j+boxsize] - product_sum_mem[j];
            double S_xy = S_xy_global - ((double)j * S_y);
            
            bestFitResult = bestFit(S_y, 
                S_xy, 
                boxsize);
            local_mse += sqrt(sumOfSquaredErrors(S_y, 
                S_y2, 
                S_xy, 
                boxsize, 
                bestFitResult.m, 
                bestFitResult.c) / boxsize);
            num_W++;
        }
        mse[i] = local_mse / num_W;
    }

    // cleanup
    free(sum_mem);
    free(square_sum_mem);
    free(product_sum_mem);

    return mse;
}
