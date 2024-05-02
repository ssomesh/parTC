
#include <cstdlib>
#include <cassert>
#include <unordered_map>
#include <vector>

using namespace std;

#include "extern/cxsparse/cs.h"

bool isEqualAfterSorting(unsigned* arr, unsigned d, unsigned ind1, unsigned ind2) {
  for(unsigned i=0; i < d; ++i) {
     if(arr[ind1 * d + i] != arr[ind2 * d + i])
         return false;
  }

  return true;
}

/*Taken from Sparta source code*/
void Swap(unsigned * arr, unsigned * aux, unsigned d, unsigned ind1, unsigned ind2) {

    for(unsigned i = 0; i < d; ++i) {
        unsigned tmp = arr[ind1 * d + i];
        arr[ind1 * d + i] = arr[ind2 * d + i];
        arr[ind2 * d + i] = tmp;
    }

    unsigned tmp = aux[ind1];
    aux[ind1] = aux[ind2];
    aux[ind2] = tmp;
}

/*Taken from Sparta source code*/
void Swap(unsigned * arr, unsigned * aux, double * vals, unsigned d, unsigned ind1, unsigned ind2) {
    unsigned tmp;
    double tmp_val;
    for(unsigned i = 0; i < d; ++i) {
        tmp = arr[ind1 * d + i];
        arr[ind1 * d + i] = arr[ind2 * d + i];
        arr[ind2 * d + i] = tmp;
    }

    tmp = aux[ind1];
    aux[ind1] = aux[ind2];
    aux[ind2] = tmp;

    tmp_val = vals[ind1];
    vals[ind1] = vals[ind2];
    vals[ind2] = tmp_val;
}

/*Taken from Sparta source code*/
int CompareIndices(unsigned * arr, unsigned d, unsigned loc1, unsigned loc2)
{
    unsigned i;
    for(i = 0; i < d; ++i) {
        unsigned eleind1 = arr[loc1 * d + i];
        unsigned eleind2 = arr[loc2 * d + i];
        if(eleind1 < eleind2) {
            return -1;
        } else if(eleind1 > eleind2) {
            return 1;
        }
    }
    return 0;
}

/*Taken from Sparta source code*/
static void QuickSort(unsigned * arr, unsigned * aux, unsigned d, unsigned l, unsigned r) 
{

// Note: to sort array of size N, starting at index 0,
// call the function with (l = 0 and r = N)
    unsigned i, j, p;
    if(r - l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(CompareIndices(arr, d, i, p) < 0) { // compare the elements at location i and j in arr
            ++i;
        }
        while(CompareIndices(arr, d, p, j) < 0) { // compare the elements at location i and j in arr
            --j;
        }
        if(i >= j) {
            break;
        }
        Swap(arr, aux, d, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }

    #pragma omp task 	
	{ QuickSort(arr, aux, d, l, i); }
	#pragma omp task 	
	{ QuickSort(arr, aux, d, i, r); }	
}

/*Taken from Sparta source code*/
static void QuickSort(unsigned * arr, unsigned * aux, double * vals, unsigned d, unsigned l, unsigned r) 
{
// Note: to sort array of size N, starting at index 0,
// call the function with (l = 0 and r = N)
    unsigned i, j, p;
    if(r - l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(CompareIndices(arr, d, i, p) < 0) { // compare the elements at location i and j in arr
            ++i;
        }
        while(CompareIndices(arr, d, p, j) < 0) { // compare the elements at location i and j in arr
            --j;
        }
        if(i >= j) {
            break;
        }
        Swap(arr, aux, vals, d, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }

    #pragma omp task 	
	{ QuickSort(arr, aux, vals, d, l, i); }
	#pragma omp task 	
	{ QuickSort(arr, aux, vals, d, i, r); }	
}



Tensor matricize_sort_spGEMM (Tensor& A, Tensor& B) {

  unsigned nThreads;
#pragma omp parallel 
    	{
         #pragma omp master 
    		{
    			nThreads = omp_get_num_threads();
    		}
    	}
	
    cout << "nThreads = " << nThreads << endl;
    uint64_t elapsed;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

   // Step 1: Collect the contraction indices of A and B and sort them together to assign them unique, but matching ids.
   
    unsigned num_key_dimensions = A.num_key_dimensions; // == B.num_key_dimensions
    unsigned AB_concat_size = A.N + B.N;

    unsigned * aux_ids_array = (unsigned*) malloc(sizeof(unsigned) * AB_concat_size);
    
#pragma omp parallel for num_threads(nThreads)
   for(unsigned i = 0; i < AB_concat_size; ++i) {
	aux_ids_array[i] = i; /*0 to A.N-1 are the ids of nonzeros in A; A.N to A.N+B.N-1 are id of nonzeros in B*/
}

   
   unsigned * AB_contraction_indices_contacatinated = (unsigned*) malloc(sizeof(unsigned) * (num_key_dimensions) * AB_concat_size); 

   
#pragma omp parallel for num_threads(nThreads)
   for(unsigned i = 0; i < A.N; ++i) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        AB_contraction_indices_contacatinated[i*num_key_dimensions + ll] = A.hedges_array[i*A.d + A.dim_tc[ll]]; // the contraction indices are stored
     }
}

#pragma omp parallel for num_threads(nThreads)
   for(unsigned i = 0; i < B.N; ++i) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        AB_contraction_indices_contacatinated[(A.N+i)*num_key_dimensions + ll] = B.hedges_array[i*B.d + B.dim_tc[ll]]; // the contraction indices are stored
     }
}

#pragma omp parallel num_threads(nThreads)
{
#pragma omp single nowait
	{
		QuickSort (AB_contraction_indices_contacatinated, aux_ids_array, num_key_dimensions, 0, AB_concat_size);
	}
}

   // create an indirection array.
   unsigned * arr_indirect = (unsigned*) malloc(sizeof(unsigned) * AB_concat_size);
   
   unsigned * arr_predicate = (unsigned*) malloc(sizeof(unsigned) * AB_concat_size);
   arr_predicate[0] = 0;

	#pragma omp parallel for num_threads(nThreads)
   for(unsigned i=1; i < AB_concat_size; ++i) {
      if(!isEqualAfterSorting(AB_contraction_indices_contacatinated, num_key_dimensions, i, i-1)) {
         arr_predicate[i] = 1;
      }
      else arr_predicate[i] = 0; 
  }

  // compute parallel prefix sum on arr_predicate[].
    const unsigned NSEG = nThreads;

	unsigned seg_len = (AB_concat_size+NSEG-1)/NSEG; /* ceil of (AB_concat_size)/NSEG */

	unsigned * partial_sum = (unsigned*) malloc(sizeof(unsigned) * NSEG); // the result of segment-wise reduction

    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(AB_concat_size));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += arr_predicate[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	
	//assert(tid > 1);

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		arr_predicate[lo+1] += arr_predicate[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),AB_concat_size-1);
		arr_predicate[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			arr_predicate[lo+1] += arr_predicate[lo];		
	}

// arr_predicate[i] stores the new id assigned to the nonzero whose original id was aux_ids_array[i]
  // make a pass over arr_predicate[] and write the corresponding value to arr_indirect[]

	#pragma omp parallel for num_threads(nThreads)
  for(unsigned i = 0; i < AB_concat_size; ++i) {
	arr_indirect[aux_ids_array[i]] = arr_predicate[i];
  }


   // Step-2: Sort A by its noncontraction indices 

   unsigned * A_noncontraction_arr = (unsigned * ) malloc(sizeof(unsigned) * (A.d-num_key_dimensions) * (A.N));
   unsigned * A_noncontraction_arr_aux = (unsigned * ) malloc(sizeof(unsigned) * (A.N));
   double * A_val_arr = (double * ) malloc(sizeof(double) * (A.N));

#pragma omp parallel for num_threads(nThreads)
   for(unsigned i = 0; i < A.N; ++i) {
     for(unsigned ll = num_key_dimensions; ll < A.d; ++ll) {
        A_noncontraction_arr[i*(A.d-num_key_dimensions) + (ll-num_key_dimensions)] = A.hedges_array[i*A.d + A.dim_tc[ll]]; // the non contraction indices are stored
     }
     A_noncontraction_arr_aux[i] = arr_indirect[i];
     A_val_arr[i] = A.val_array[i];
}


#pragma omp parallel num_threads(nThreads)
{
#pragma omp single nowait
	{
		QuickSort (A_noncontraction_arr, A_noncontraction_arr_aux, A_val_arr, A.d -num_key_dimensions, 0, A.N);
	}
}

   // create a reverse map when assigning new id's to noncontraction indices so that we are able to retrieve the noncontraction indices

   unsigned * A_noncontraction_arr_id = (unsigned*) malloc(sizeof(unsigned) * A.N); // this will be part of the COO format for A
   unsigned * A_noncontraction_reverse = (unsigned*) malloc(sizeof(unsigned) * A.N * (A.d - num_key_dimensions)); // this will be used to retieve the noncontraction indices for A for a given 

   A_noncontraction_arr_id[0] = 0;

	#pragma omp parallel for num_threads(nThreads)
   for(unsigned i=1; i < A.N; ++i) {
      if(!isEqualAfterSorting(A_noncontraction_arr, A.d-num_key_dimensions, i, i-1)) {
         A_noncontraction_arr_id[i] = 1;
      }
      else A_noncontraction_arr_id[i] = 0; 
  }


  // compute parallel prefix sum on A_noncontraction_arr_id[].

	seg_len = (A.N+NSEG-1)/NSEG; /* ceil of (A.N)/NSEG */

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(A.N));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += A_noncontraction_arr_id[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		A_noncontraction_arr_id[lo+1] += A_noncontraction_arr_id[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),A.N-1);
		A_noncontraction_arr_id[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			A_noncontraction_arr_id[lo+1] += A_noncontraction_arr_id[lo];		
	}

// A_noncontraction_arr_id[i] stores the new id assigned to the nonzero 

	#pragma omp parallel for num_threads(nThreads)
  for(unsigned i = 0; i < A.N; ++i) {

       for(unsigned j=0; j < A.d - num_key_dimensions; ++j)
         A_noncontraction_reverse[A_noncontraction_arr_id[i] * (A.d-num_key_dimensions) + j] = A_noncontraction_arr[i*(A.d-num_key_dimensions) + j];

  }







#pragma omp parallel num_threads(nThreads)
{
#pragma omp single nowait
	{
		QuickSort (A_noncontraction_arr_aux, A_noncontraction_arr_id, A_val_arr, 1, 0, A.N);
	}
}


   // Step-3: Sort B by its noncontraction indices 

   unsigned * B_noncontraction_arr = (unsigned * ) malloc(sizeof(unsigned) * (B.d-num_key_dimensions) * (B.N));
   unsigned * B_noncontraction_arr_aux = (unsigned * ) malloc(sizeof(unsigned) * (B.N));
   double * B_val_arr = (double * ) malloc(sizeof(double) * (B.N));

#pragma omp parallel for num_threads(nThreads)
   for(unsigned i = 0; i < B.N; ++i) {
     for(unsigned ll = num_key_dimensions; ll < B.d; ++ll) {
        B_noncontraction_arr[i*(B.d-num_key_dimensions) + (ll-num_key_dimensions)] = B.hedges_array[i*B.d + B.dim_tc[ll]]; // the non contraction indices are stored
     }
     B_noncontraction_arr_aux[i] = arr_indirect[A.N+i]; // since the indices A.N to A.N+B.N-1 in arr_indirect[] contain the ids corresponding to the nonzeros in B. 
     B_val_arr[i] = B.val_array[i];
}

#pragma omp parallel num_threads(nThreads)
{
#pragma omp single nowait
	{
		QuickSort (B_noncontraction_arr, B_noncontraction_arr_aux, B_val_arr, B.d -num_key_dimensions, 0, B.N);
	}
}
   
   // create a reverse map when assigning new id's to noncontraction indices so that we are able to retrieve the noncontraction indices

   unsigned * B_noncontraction_arr_id = (unsigned*) malloc(sizeof(unsigned) * B.N); // this will be part of the COO format for B
   unsigned * B_noncontraction_reverse = (unsigned*) malloc(sizeof(unsigned) * B.N * (B.d - num_key_dimensions)); // this will be used to retieve the noncontraction indices for B for a given 

   B_noncontraction_arr_id[0] = 0;

	#pragma omp parallel for num_threads(nThreads)
   for(unsigned i=1; i < B.N; ++i) {
      if(!isEqualAfterSorting(B_noncontraction_arr, B.d-num_key_dimensions, i, i-1)) {
         B_noncontraction_arr_id[i] = 1;
      }
      else B_noncontraction_arr_id[i] = 0; 
  }

  // compute parallel prefix sum on B_noncontraction_arr_id[].

	seg_len = (B.N+NSEG-1)/NSEG; /* ceil of (B.N)/NSEG */


    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(B.N));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += B_noncontraction_arr_id[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		B_noncontraction_arr_id[lo+1] += B_noncontraction_arr_id[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),B.N-1);
		B_noncontraction_arr_id[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			B_noncontraction_arr_id[lo+1] += B_noncontraction_arr_id[lo];		
	}

	#pragma omp parallel for num_threads(nThreads)
  for(unsigned i = 0; i < B.N; ++i) {

       for(unsigned j=0; j < B.d - num_key_dimensions; ++j)
         B_noncontraction_reverse[B_noncontraction_arr_id[i] * (B.d-num_key_dimensions) + j] = B_noncontraction_arr[i*(B.d-num_key_dimensions) + j];

  }


#pragma omp parallel num_threads(nThreads)
{
#pragma omp single nowait
	{
		QuickSort (B_noncontraction_arr_id, B_noncontraction_arr_aux, B_val_arr, 1, 0, B.N);
	}
}


/* Conversion of A from COO to  CSC*/

     unsigned * predicate_array_A = (unsigned*) malloc(sizeof(unsigned) * (A.N));
     predicate_array_A[0] = 0;

#pragma omp parallel for num_threads(nThreads)
 for(unsigned i=1; i < A.N; ++i) {
      if(!isEqualAfterSorting(A_noncontraction_arr_aux, 1, i, i-1)) {
	    predicate_array_A[i] = 1;
         }
         else
             predicate_array_A[i] = 0; 
 }

 
  // compute parallel prefix sum on predicate_array_A[].

	seg_len = (A.N+NSEG-1)/NSEG; /* ceil of (A.N)/NSEG */


    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(A.N));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += predicate_array_A[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		predicate_array_A[lo+1] += predicate_array_A[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),A.N-1);
		predicate_array_A[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			predicate_array_A[lo+1] += predicate_array_A[lo];		
	}

  
  unsigned * col_ptrs_A = (unsigned*) malloc(sizeof(unsigned) * (predicate_array_A[A.N-1]+2));  // correct (this is equal to number of distinct elements + 1)

  memset (col_ptrs_A, 0, sizeof(unsigned) * (predicate_array_A[A.N-1]+2));

   unsigned max_col_id_A = 0;
#pragma omp parallel for reduction(max: max_col_id_A) num_threads(nThreads)
   for (unsigned i = 0; i < A.N; i++)  {

	   if (A_noncontraction_arr_id[i] > max_col_id_A) {
		   max_col_id_A = A_noncontraction_arr_id[i];
	   }
	   __atomic_fetch_add(&col_ptrs_A[predicate_array_A[i]+1],1, __ATOMIC_RELAXED); // atomic add
   }

  // compute parallel prefix sum on col_ptrs_A[].

	seg_len = (predicate_array_A[A.N-1]+2+NSEG-1)/NSEG; /* ceil of (A.N)/NSEG */


    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(predicate_array_A[A.N-1]+2));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += col_ptrs_A[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		col_ptrs_A[lo+1] += col_ptrs_A[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),predicate_array_A[A.N-1]+2-1);
		col_ptrs_A[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			col_ptrs_A[lo+1] += col_ptrs_A[lo];		
	}




/* Conversion of B from COO to  CSC*/

     unsigned * predicate_array_B = (unsigned*) malloc(sizeof(unsigned) * (B.N));
     predicate_array_B[0] = 0;

#pragma omp parallel for num_threads(nThreads)
 for(unsigned i=1; i < B.N; ++i) {
      if(!isEqualAfterSorting(B_noncontraction_arr_id, 1, i, i-1)) {
	    predicate_array_B[i] = 1;
         }
         else
             predicate_array_B[i] = 0; 
 }

 
  // compute parallel prefix sum on predicate_array_B[].

	seg_len = (B.N+NSEG-1)/NSEG; /* ceil of (B.N)/NSEG */


    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(B.N));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += predicate_array_B[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		predicate_array_B[lo+1] += predicate_array_B[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),B.N-1);
		predicate_array_B[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			predicate_array_B[lo+1] += predicate_array_B[lo];		
	}

  
  unsigned * col_ptrs_B = (unsigned*) malloc(sizeof(unsigned) * (predicate_array_B[B.N-1]+2));  // correct (this is equal to number of distinct elements + 1)

  memset (col_ptrs_B, 0, sizeof(unsigned) * (predicate_array_B[B.N-1]+2));

   unsigned max_col_id_B = 0;
#pragma omp parallel for reduction(max: max_col_id_B) num_threads(nThreads)
   for (unsigned i = 0; i < B.N; i++)  {

	   if (B_noncontraction_arr_aux[i] > max_col_id_B) {
		   max_col_id_B = B_noncontraction_arr_aux[i];
	   }
	   __atomic_fetch_add(&col_ptrs_B[predicate_array_B[i]+1],1, __ATOMIC_RELAXED); // atomic add
   }

  // compute parallel prefix sum on col_ptrs_B[].

	seg_len = (predicate_array_B[B.N-1]+2+NSEG-1)/NSEG; /* ceil of (B.N)/NSEG */


    
	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=0; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo + seg_len),(predicate_array_B[B.N-1]+2));
		unsigned my_sum = 0;
		for(; lo < hi; ++lo) 
			my_sum += col_ptrs_B[lo];
		
		partial_sum[tid] = my_sum;
	}

	// prefix sum of partial sums
	for (unsigned tid=0; tid < NSEG-1; ++tid)
		partial_sum[tid+1] += partial_sum[tid];
	

    /* For tid = 0  */ 
	for(unsigned lo = 0; lo < seg_len-1; ++lo) 
		col_ptrs_B[lo+1] += col_ptrs_B[lo];
	       

	#pragma omp parallel for num_threads(nThreads)
	for (unsigned tid=1; tid < NSEG; ++tid) 
	{
		unsigned lo = tid*seg_len;
		unsigned hi = min((lo+seg_len-1),predicate_array_B[B.N-1]+2-1);
		col_ptrs_B[lo] +=  (partial_sum[tid-1]);
		for(; lo < hi; ++lo) 
			col_ptrs_B[lo+1] += col_ptrs_B[lo];		
	}

    chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t3 - t1).count();
    cout << "Total time for matricization = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;

exit(1);


    /* populating the data structure used by CXSparse */

    // For A_mtx
   
    cs* A_csc = new cs; // the csc format matrix to be used by CXSparse

    A_csc->nzmax = A.N;
    A_csc->nz = -1;

   A_csc->m = max_col_id_A + 1; 
   A_csc->n = predicate_array_A[A.N-1]+1; //predicate_array_A[A.N-1]+2-1


  A_csc->p = (int32_t*) malloc(sizeof(int32_t) * ((A_csc->n) + 1)); 
  A_csc->i = (int32_t*) malloc(sizeof(int32_t) * (A_csc->nzmax));  
  A_csc->x = (double*) malloc(sizeof(double) * (A_csc->nzmax));  // stores the val

#pragma omp parallel for num_threads(nThreads)
  for(unsigned i=0; i < A.N; ++i) {
     (A_csc->i)[i] = A_noncontraction_arr_id[i];
     (A_csc->x)[i] = A_val_arr[i];
  }

#pragma omp parallel for num_threads(nThreads)
  for(unsigned i=0; i < predicate_array_A[A.N-1]+2; ++i) {
     (A_csc->p)[i] = col_ptrs_A[i];
  }
 

    /* populating the data structure used by CXSparse */

    // For B_mtx
   
    cs* B_csc = new cs; // the csc format matrix to be used by CXSparse

    B_csc->nzmax = B.N;
    B_csc->nz = -1;

   B_csc->m = max_col_id_B + 1; 
   B_csc->n = predicate_array_B[B.N-1]+1; //predicate_array_B[B.N-1]+2-1


  B_csc->p = (int32_t*) malloc(sizeof(int32_t) * ((B_csc->n) + 1)); 
  B_csc->i = (int32_t*) malloc(sizeof(int32_t) * (B_csc->nzmax));  
  B_csc->x = (double*) malloc(sizeof(double) * (B_csc->nzmax));  // stores the val

#pragma omp parallel for num_threads(nThreads)
  for(unsigned i=0; i < B.N; ++i) {
     (B_csc->i)[i] = B_noncontraction_arr_aux[i];
     (B_csc->x)[i] = B_val_arr[i];
  }

#pragma omp parallel for num_threads(nThreads)
  for(unsigned i=0; i < predicate_array_B[B.N-1]+2; ++i) {
     (B_csc->p)[i] = col_ptrs_B[i];
  }


    cs* C_csc = NULL;
    C_csc= cs_multiply (A_csc, B_csc) ;

    if(C_csc == NULL) {
        cerr << "CXSparse returned a NULL output object!! Check the input!" << endl;
      exit(1);
    }

    Tensor C;

    C.d = (A.d - A.num_key_dimensions + B.d - B.num_key_dimensions);
    C.num_key_dimensions = C.d;
    C.N = C_csc->nzmax;
    C.hedges_array = (unsigned*) malloc(sizeof(unsigned) * C.d * C.N);
    C.val_array = (double*) malloc(sizeof(double) *  C.N);



  unsigned counter_gbl = 0;

#pragma omp parallel for num_threads(nThreads)
  for(unsigned ii = 0; ii < C_csc->n; ++ii) {

      unsigned* C_hedges_array_local = (unsigned*)malloc(sizeof(unsigned) * ((C_csc->p)[ii+1]-(C_csc->p)[ii]) * C.d);
      
      double* C_val_array_local = (double*)malloc(sizeof(double) * ((C_csc->p)[ii+1]-(C_csc->p)[ii]));
      unsigned counter_local = 0;

      for (unsigned p = (C_csc->p)[ii] ; p < (C_csc->p)[ii+1] ; p++) {

       for(unsigned p1 = 0; p1 < A.d - A.num_key_dimensions; ++p1)
           C_hedges_array_local[counter_local*(C.d) + p1] = A_noncontraction_reverse[(C_csc->i)[p] * (A.d - A.num_key_dimensions) + p1]; // this gives the tuple corresponding to the row id

       for(unsigned p1 = 0; p1 < B.d - B.num_key_dimensions; ++p1)
           C_hedges_array_local[counter_local*(C.d) + (A.d - A.num_key_dimensions) + p1] = B_noncontraction_reverse[ii*(B.d - B.num_key_dimensions) + p1];


       C_val_array_local[counter_local] = (C_csc->x)[p];
       counter_local++;
      }

        unsigned start_index = __atomic_fetch_add(&counter_gbl, counter_local, __ATOMIC_RELAXED);

        for(unsigned xx = 0; xx < counter_local; ++xx) {
       for(unsigned p1 = 0; p1 < C.d; ++p1)
           C.hedges_array[(start_index + xx)*(C.d) + p1] = C_hedges_array_local[xx * (C.d)+ p1]; 

           C.val_array[(start_index + xx)]= C_val_array_local[xx];


                   
           
	}
  }


    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Total time for sortMatSpGEMM = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;

    return C;
   
}
