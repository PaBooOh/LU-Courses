// include standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>

// TODO: YOUR "predicated" out-of-place sort implementation
void sort_predicated_outofplace( int N , int *input_array , int *result_array ) {

	// TODO:
	// replace this with your
	// "predicated" (branch-free) out-of-place sort implementation

	for ( int i = 0 , j = N - 1 ; i < N ; i++ , j-- ) {
		result_array[ i ] = input_array[ j ];
	}

	return;
}

// main program
int main( int argc , char ** argv ) {

	// variables for timing
	struct timeval before , after;
	long long int usec = 0;

	// check number command line arguments
	if ( argc != 3 ) {
		// incorrect number of command line arguments
		fprintf(
		    stderr ,
		    "need two arguments\n"
		);
		// exit with error status
		return( 1 );
	}

	// read first command line argument:
	// number of data values
	int N = atoi( argv[ 1 ] );
	// sanity check: echo argument value
	fprintf(
		stderr ,
		"N = %d\n" ,
		N
	);

	// read second command line argument:
	// (absolute/relative) path to input data file
	char *input_data_file_location = strdup( argv[ 2 ] );
	// sanity check: echo argument value
	fprintf(
		stderr ,
		"input_data_file_location = '%s'\n" ,
		input_data_file_location
	);

	// allocate memory for input array
	int *input_array = malloc( N * sizeof(int) );
	if ( input_array == NULL ) {
		// report error
		fprintf(
			stderr ,
			"malloc failed for input_array (%zu bytes)\n" ,
			N * sizeof(int)
		);
		// free allocated memory
		free( input_data_file_location );
		// exit with error status
		return( 1 );
	}

	// allocate memory for result array
	int *result_array = malloc( N * sizeof(int) );
	if ( result_array == NULL ) {
		// report error
		fprintf(
			stderr ,
			"malloc failed for result_array (%zu bytes)\n" ,
			N * sizeof(int)
		);
		// free allocated memory
		free( input_array );
		free( input_data_file_location );
		// exit with error status
		return( 1 );
	}

	// arbitrarily initialize result array
	// with a value we don't use, here:
	// INT_MIN == -2^31 == -2147483648
	for ( int i = 0 ; i < N ; i++ ) {
		result_array[ i ] = INT_MIN;
	}

	// open input data file for reading
	FILE *input_data_file = fopen( input_data_file_location , "r" );
	if ( input_data_file == NULL ) {
		// report error
		perror( "fopen failed" );
		// free allocated memory
		free( result_array );
		free( input_array );
		free( input_data_file_location );
		// exit with error status
		return( 1 );
	}

	// load content of input data file in to input array;
	// measure and report time it takes
	gettimeofday( &before , NULL );
	for ( int i = 0 ; i < N ; i++ ) {
		int e = fscanf(
			input_data_file ,
			"%d\n" ,
			input_array + i
		);
		if ( e == EOF ) {
			// report error
			perror( "fscanf: premature EOF" );
			// close input data file
			fclose( input_data_file );
			// free allocated memory
			free( result_array );
			free( input_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
		if ( e < 1 ) {
			// report error
			perror( "fscanf failed" );
			// close input data file
			fclose( input_data_file );
			// free allocated memory
			free( result_array );
			free( input_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
	}
	gettimeofday( &after , NULL );
	usec = (long long int)(after.tv_sec - before.tv_sec) * 1000000
			   + (after.tv_usec - before.tv_usec);
	fprintf(
		stderr ,
		"load data: %9lld usec ; input_array[ 0 ] = %10d , input_array[ %d ] = %10d , input_array[ %d ] = %10d\n" ,
		usec , input_array[ 0 ] , N / 2 , input_array[ N / 2 ] , N - 1 , input_array[ N - 1 ]
	);

	// close input data file
	fclose( input_data_file );

	// sort input array out-of-place (into result array);
	// measure the time it takes
	gettimeofday( &before , NULL );
	sort_predicated_outofplace( N , input_array , result_array );
	gettimeofday( &after , NULL );

	// check result correctness
	for ( int i = 1 ; i < N ; i++ ) {
		if ( result_array[ i - 1 ] > result_array[ i ] ) {
			// report error
			fprintf(
				stderr ,
				"result not sorted:\n result_array[ %d ] == %10d\n result_array[ %d ] == %10d\n" ,
				i - 1 , result_array[ i - 1 ] , i , result_array[ i ]
			);
			// free allocated memory
			free( result_array );
			free( input_array );
			free( input_data_file_location );
			// exit with error status
			return( 1 );
		}
	}

	// calculate and report sorting time
	usec = (long long int)(after.tv_sec - before.tv_sec) * 1000000
			   + (after.tv_usec - before.tv_usec);
	fprintf(
		stderr ,
		"sort data: %9lld usec ; result_array[ 0 ] = %10d , result_array[ %d ] = %10d , result_array[ %d ] = %10d\n" ,
		usec , result_array[ 0 ] , N / 2 , result_array[ N / 2 ] , N - 1 , result_array[ N - 1 ]
	);

	// output sorted data array to console (stdout)
	// (one value per line)
	for ( int i = 0 ; i < N ; i++ ) {
		printf(
			"%d\n" ,
			result_array[ i ]
		);
	}

	// free allocated memory
	free( result_array );
	free( input_array );
	free( input_data_file_location );

	// exit without error status
	return( 0 );
}
