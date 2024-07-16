// include standard headers
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

// main program
int main( int argc , char ** argv ) {

	// variables for timing
	struct timeval before , after;
	long long int usec = 0;

	// check number command line arguments
	if ( argc != 2 ) {
		// incorrect number of command line arguments
		fprintf(
		    stderr ,
		    "need one argument: number of data values\n"
		);
		// exit with error status
		return( 1 );
	}

	// read command line argument:
	// number of data values
	int N = atoi( argv[ 1 ] );
	// sanity check: echo argument value
	fprintf(
		stderr ,
		"N = %d\n" ,
		N
	);

	// allocate memory for values array
	int *values = malloc( N * sizeof(int) );
	if ( values == NULL ) {
		// report error
		fprintf(
			stderr ,
			"malloc failed for values array (%zu bytes)\n" ,
			N * sizeof(int)
		);
		// exit with error status
		return( 1 );
	}

	// allocate memory for result array
	int *result = calloc( N , sizeof(int) );
	if ( result == NULL ) {
		// report error
		fprintf(
			stderr ,
			"malloc failed for result array (%zu bytes)\n" ,
			N * sizeof(int)
		);
		// free allocated memory
		free( values );
		// exit with error status
		return( 1 );
	}

	// sanity check: echo value of RAND_MAX
	fprintf(
		stderr ,
		"RAND_MAX = %d\n" ,
		RAND_MAX
	);
	// fill values array with ordered values equally spaced in [0,RAND_MAX]
	for ( int i = 0 ; i < N ; i++ ) {
		values[i] = (int)( (double)RAND_MAX * (double)i / (double)(N-1) );
	}

	// iterate over selectivities 0% .. 100% n steps of 5%
	for ( int p = 0 ; p <= 100 ; p += 5 ) {
		// result counter
		int j = 0;
		// boundary value for given selectivity
		int w = (int)((long long int)RAND_MAX * p / 100);
		// report selectivity (%) and respective boundary value
		printf(
			"%3d%% %10d " ,
			p , w
		);

		// record start time
		gettimeofday( &before , NULL );
		// loop over values and calculate result
		for ( int i = 0 ; i < N ; i++ ) {
			// "branched" implementation
			if ( values[i] < w ) {
				result[j] = values[i];
				j++;
			}
		}
		// record end time
		gettimeofday( &after , NULL );

		// calculate execution time
		usec = (long long int)(after.tv_sec - before.tv_sec) * 1000000
		                   + (after.tv_usec - before.tv_usec);
		// report number of result values, execution time and random sample result value
		int k = (int)(((size_t)rand()) * j / RAND_MAX);
		printf(
			" %10d %9lld usec %10d %10d\n" , 
			j , usec , k , result[k]
		);
	}

	// free allocated memory
	free( values );
	free( result );

	// exit without error status
	return( 0 );
}
