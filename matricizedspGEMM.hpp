#include <cstdlib>
//#include <cassert>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

#include "extern/fastmod-master/include/fastmod.h"

#include "extern/sparsehash/sparse_hash_map"

#include "extern/cxsparse/cs.h"

using google::sparse_hash_map;      // namespace where class lives by default

#define modpMersennehala(y, x)  \
{                                  \
	y = ((x) & (pp_hala_)) + ((x)>>(qq_hala_));\
	y = y >= (pp_hala_) ? y-= (pp_hala_) : y;  \
}




/*Moved to fksCuckooTC.hpp due to compilation dependencies*/
//unsigned int _global_start;
//unsigned int _global_end;

int comp_items(const void *h1,const void*h2)
{
	unsigned int *h1p = (unsigned int*) h1;
	unsigned int *h2p = (unsigned int*) h2;
	for (unsigned int i =  _global_start; i < _global_end; i++)
	{
		if (h1p[i] < h2p[i]) return -1;
		if (h1p[i] > h2p[i]) return 1;
	}
	return 0;
}


void findMersennehala(unsigned int N, unsigned int &p, unsigned int &q)
{

    vector <unsigned int> qqs ({2, 3, 5, 7, 13, 17, 19, 31});
    vector <unsigned int> pps = qqs;
    unsigned int one  =1;
    p = q = 0;
    for (unsigned int i =0 ;i < pps.size(); i++)
        pps[i] = (one << qqs[i]) - 1;

    for (unsigned i = 0; i < qqs.size(); i++)
    {
        if(pps[i]>=N)
        {
            p = pps[i];
            q = qqs[i];
            break;
        }
    }
    if (p == 0)
    {
        cout << "Could not find a proper mersenne prime "<<endl;
        exit(12);
    }
}

unsigned int randomnumberinthala(unsigned int p) {/*from https://stackoverflow.com/questions/56435506/rng-function-c*/
    // Making rng static ensures that it stays the same
    // Between different invocations of the function
    static std::mt19937 rng;

    std::uniform_int_distribution<uint32_t> dist(0, p-1); 
    return dist(rng); 
}

bool est_premierhala(unsigned i)
{
    if (i%2 == 0)
        return false;

    for (unsigned j = 3; j*j <= i; j += 2)
    {
        if (i%j == 0)
            return false;
    }
    return true;
}

void trouver_premierhala(const unsigned int m, unsigned int&p)
{
    unsigned i = m+1;
    while (!est_premierhala(i))
        i+=1;
    p=i;
}

void findHashalaFKSparams(unsigned int *hedges,  unsigned int d, const unsigned int * dimensions,  unsigned int N,
	vector <unsigned long long int> &k, unsigned int &pp_hala, unsigned int &qq_hala)
{
	unsigned int larger=N;
	uint64_t M_fastmod;//, M_p_fastmod ;
	for (unsigned int i = 0; i < d; i++)
	{
		if (dimensions[i]>larger)
			larger=dimensions[i];
	}
/*
	if (larger<N)
		trouver_premier(N,pp_hala);
	else
		trouver_premier(larger, pp_hala);
*/
	pp_hala = 2147483647 ;/*this is 2^31-1*/
	qq_hala = 31;
	if( pp_hala < larger)
	{
		trouver_premierhala(larger, pp_hala);
		qq_hala = 0;

	}

	if (pp_hala == 0)
	{
		cout << "in hashalaFKS could not find a proper mersenne prime"<<endl;
		exit(12);
	}

	srand(time(NULL));
	k.resize(d);
	M_fastmod= fastmod::computeM_u32(N);

    for (unsigned int i = 0; i < d; i++)
		{
			k[i] = (unsigned long long) randomnumberinthala(pp_hala);
		}


}



struct hashalaFKS
{
	hashalaFKS(unsigned int d, unsigned int N, vector<unsigned long long int> k, unsigned int pp_hala, unsigned int qq_hala) {d_ = d; N_ = N;  k_ = k; pp_hala_ = pp_hala; qq_hala_ = qq_hala; M_fastmod= fastmod::computeM_u32(N);}
	size_t operator()(const unsigned int * myKey) const noexcept
	{
		unsigned long long somme, aa ;
		unsigned int		indice;
		somme = 0;
		for (unsigned int j = 0; j < d_; j++)
			somme += myKey[j]*k_[j];
	modpMersennehala(aa, somme);
				indice = fastmod::fastmod_u32((unsigned int) aa, M_fastmod, N_);
		return  (size_t) indice;

	}
  bool operator()(const unsigned int *x, const unsigned int *y) const{
    for(unsigned int i=0;i<d_;i++) if (x[i] != y[i]) return false;
    return true;
  }


	uint64_t M_fastmod;// ,M_p_fastmod;

	unsigned int d_, N_, pp_hala_, qq_hala_;
	vector<unsigned long long int> k_;
};


Tensor matricize_spGEMM (Tensor& A, Tensor& B) {

    //assert(A.num_key_dimensions == B.num_key_dimensions);

    uint64_t elapsed;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    /* create CSC format matrix */

    chrono::high_resolution_clock::time_point t100 = chrono::high_resolution_clock::now();
    // Step 1: Hash the contraction indices of A and B together and assign them unique ids. 
    // The values of the contraction indices need to match in both the matrices.

    unsigned num_key_dimensions = A.num_key_dimensions; // == B.num_key_dimensions
    //unsigned myMap_num_buckets = (A.N >= B.N) ? A.N : B.N;
    unsigned myMap_num_buckets = A.N + B.N;

   unsigned * my_tuples_list = (unsigned*) malloc(sizeof(unsigned) * (num_key_dimensions) * (A.N + B.N)); 


   unsigned kk = 0;
   for (unsigned jj = 0; jj < A.N; ++jj) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        my_tuples_list[kk++] = A.hedges_array[jj*A.d + A.dim_tc[ll]]; // the contraction indices are stored
     }
   }

   for (unsigned jj = 0; jj < B.N; ++jj) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        my_tuples_list[kk++] = B.hedges_array[jj*B.d + B.dim_tc[ll]]; // the contraction indices are stored
     }
   }

    unsigned int pp_hala, qq_hala;
    vector <unsigned long long int> k;

    findHashalaFKSparams(my_tuples_list, num_key_dimensions, A.dimensions_array, myMap_num_buckets, k, pp_hala, qq_hala);

   sparse_hash_map<unsigned int*, unsigned int, hashalaFKS, hashalaFKS> myMap{myMap_num_buckets, hashalaFKS(num_key_dimensions, myMap_num_buckets, k, pp_hala, qq_hala),hashalaFKS(num_key_dimensions, myMap_num_buckets, k, pp_hala, qq_hala) };

    for (unsigned int i=0, counter = 1; i < A.N+B.N; i++) {

            unsigned tmp_val = myMap[&(my_tuples_list[i*(num_key_dimensions)])];
            if (tmp_val) continue; // the item is already in the hash table
            //myMap.insert({&(my_tuples_list[i*(d+1)]),/*i+1*/counter++}); 

            //myMap[&(my_tuples_list[i*(d+1)])] = counter++;
            myMap[&(my_tuples_list[i*(num_key_dimensions)])] = counter++;
            //cout << "counter = " << counter << endl;
    } 

    chrono::high_resolution_clock::time_point t200 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t200 - t100).count();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t200 - t100).count();
    cout << "time for hashing the contraction indices in A and B = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;
            
    /* IMP: if a key is not present in the hash table, querying for it using the [] notation will return 0. As a side-effect it also inserts the key in the table. That side effect is okay for us because we intend to insert the key*/







    // Step 2: For matricizing A, put the non-contraction indices on the row and the contraction indices on the column.

    chrono::high_resolution_clock::time_point t300 = chrono::high_resolution_clock::now();
    // Step 2a: Hash the non-contraction indices in A


    // create a reverse map for A and B inorder to retrieve the structure of the resulting tensor.
    // only create a reverse map for the non-contraction indices for both A and B.

    num_key_dimensions = A.d - A.num_key_dimensions;  // these many non-contraction dimensions.
    unsigned * dim_tc_local_A = (unsigned*)malloc(sizeof(unsigned) * A.d);
    
    for(unsigned xy=0; xy < A.num_key_dimensions; ++xy) {
          dim_tc_local_A[num_key_dimensions+xy] = A.dim_tc[xy];
    }

    for(unsigned xy=0; xy < num_key_dimensions; ++xy) {
          dim_tc_local_A[xy] = A.dim_tc[A.num_key_dimensions+xy];
    }


   unsigned * my_tuples_list_A = (unsigned*) malloc(sizeof(unsigned) * (num_key_dimensions) * (A.N)); 


   kk = 0;
   for (unsigned jj = 0; jj < A.N; ++jj) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        my_tuples_list_A[kk++] = A.hedges_array[jj*A.d + dim_tc_local_A[ll]]; // the non-contraction indices are stored 
     }
   }

    unsigned int pp_hala_A, qq_hala_A;

    findHashalaFKSparams(my_tuples_list_A, num_key_dimensions, A.dimensions_array, A.N, k, pp_hala, qq_hala);


   sparse_hash_map<unsigned int*, unsigned int, hashalaFKS, hashalaFKS> myMap_A{A.N, hashalaFKS(num_key_dimensions, A.N, k, pp_hala, qq_hala),hashalaFKS(num_key_dimensions, A.N, k, pp_hala, qq_hala) };

   unsigned* myMap_A_reverse = (unsigned*) malloc(sizeof(unsigned) * num_key_dimensions * A.N ); // it should be of size (counter-1) instead of A.N.. shrink it to fit the size later

    for (unsigned int i=0, counter = 1; i < A.N; i++) {

            unsigned tmp_val = myMap_A[&(my_tuples_list_A[i*(num_key_dimensions)])];
            if (tmp_val) continue; // the item is already in the hash table
            myMap_A[&(my_tuples_list_A[i*(num_key_dimensions)])] = counter;
            
            for(unsigned p=0; p < num_key_dimensions; ++p)
                myMap_A_reverse[(counter-1) * num_key_dimensions + p] = my_tuples_list_A[i*(num_key_dimensions) + p];

            counter++;
    } 

    chrono::high_resolution_clock::time_point t400 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t400 - t300).count();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t400 - t300).count();
    cout << "time for hashing the non contraction indices in A = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;

    // Step 3a: Hash the non-contraction indices in B
    chrono::high_resolution_clock::time_point t500 = chrono::high_resolution_clock::now();

    num_key_dimensions = B.d - B.num_key_dimensions;  // these many non-contraction dimensions.
    unsigned * dim_tc_local_B = (unsigned*)malloc(sizeof(unsigned) * B.d);
    
    for(unsigned xy=0; xy < B.num_key_dimensions; ++xy) {
          dim_tc_local_B[num_key_dimensions+xy] = B.dim_tc[xy];
    }

    for(unsigned xy=0; xy < num_key_dimensions; ++xy) {
          dim_tc_local_B[xy] = B.dim_tc[B.num_key_dimensions+xy];
    }


   unsigned * my_tuples_list_B = (unsigned*) malloc(sizeof(unsigned) * (num_key_dimensions) * (B.N)); 


   kk = 0;
   for (unsigned jj = 0; jj < B.N; ++jj) {
     for(unsigned ll = 0; ll < num_key_dimensions; ++ll) {
        my_tuples_list_B[kk++] = B.hedges_array[jj*B.d + dim_tc_local_B[ll]]; // the non-contraction indices are stored
     }
   }

    unsigned int pp_hala_B, qq_hala_B;

    findHashalaFKSparams(my_tuples_list_B, num_key_dimensions, B.dimensions_array, B.N, k, pp_hala, qq_hala);


   sparse_hash_map<unsigned int*, unsigned int, hashalaFKS, hashalaFKS> myMap_B{B.N, hashalaFKS(num_key_dimensions, B.N, k, pp_hala, qq_hala),hashalaFKS(num_key_dimensions, B.N, k, pp_hala, qq_hala) };
   unsigned* myMap_B_reverse = (unsigned*) malloc(sizeof(unsigned) * num_key_dimensions * B.N ); // it should be of size (counter-1) instead of B.N.. shrink it to fit the size later

    for (unsigned int i=0, counter = 1; i < B.N; i++) {

            unsigned tmp_val = myMap_B[&(my_tuples_list_B[i*(num_key_dimensions)])];
            if (tmp_val) continue; // the item is already in the hash table
            myMap_B[&(my_tuples_list_B[i*(num_key_dimensions)])] = counter;
            for(unsigned p=0; p < num_key_dimensions; ++p)
                myMap_B_reverse[(counter-1) * num_key_dimensions + p] = my_tuples_list_B[i*(num_key_dimensions) + p];

            counter++;
    } 

    chrono::high_resolution_clock::time_point t600 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t600 - t500).count();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t600 - t500).count();
    cout << "time for hashing the non contraction indices in B = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;


    chrono::high_resolution_clock::time_point t700 = chrono::high_resolution_clock::now();

   unsigned * A_mtx = (unsigned*) malloc(sizeof(unsigned) * (2 + 1) * (A.N)); 


   unsigned* A_temp_indices1 = (unsigned*) malloc(sizeof(unsigned) * A.num_key_dimensions);
   unsigned* A_temp_indices2 = (unsigned*) malloc(sizeof(unsigned) * (A.d - A.num_key_dimensions));
   //unsigned kk = 0;
   for (unsigned jj = 0; jj < A.N; ++jj) {
     for(unsigned ll = 0; ll < A.num_key_dimensions; ++ll) 
         A_temp_indices1[ll] = A.hedges_array[jj*A.d + A.dim_tc[ll]];
         
     for(unsigned ll = 0; ll < A.d - A.num_key_dimensions; ++ll) 
         A_temp_indices2[ll] = A.hedges_array[jj*A.d + dim_tc_local_A[ll]];
  
     // for easy conversion to CSC format:
     A_mtx[jj*3] = myMap[A_temp_indices1] - 1;  // possibly do a -1
     A_mtx[jj*3+1] = myMap_A[A_temp_indices2] - 1; // possibly subtract 1
     A_mtx[jj*3 + 2] = jj;
   }

   unsigned * B_mtx = (unsigned*) malloc(sizeof(unsigned) * (2+1) * (B.N)); 


   unsigned* B_temp_indices1 = (unsigned*) malloc(sizeof(unsigned) * B.num_key_dimensions);
   unsigned* B_temp_indices2 = (unsigned*) malloc(sizeof(unsigned) * (B.d - B.num_key_dimensions));
   for (unsigned jj = 0; jj < B.N; ++jj) {
     for(unsigned ll = 0; ll < B.num_key_dimensions; ++ll) 
         B_temp_indices1[ll] = B.hedges_array[jj*B.d + B.dim_tc[ll]];
     for(unsigned ll = 0; ll < B.d - B.num_key_dimensions; ++ll) 
         B_temp_indices2[ll] = B.hedges_array[jj*B.d + dim_tc_local_B[ll]];
  
     // for easy conversion to CSC format:
     B_mtx[jj*3] = myMap_B[B_temp_indices2] - 1; // possibly subtract 1
     B_mtx[jj*3+1] = myMap[B_temp_indices1] - 1;  // possibly do a -1
     B_mtx[jj*3 + 2] = jj;
   }


  _global_start = 0;
  _global_end = 1;
  qsort(A_mtx, A.N, (3) * sizeof(unsigned int), comp_items);
  qsort(B_mtx, B.N, (3) * sizeof(unsigned int), comp_items);

#if 1
  // uncomment this block to convert to CSC.
  unsigned offset_A = 1;
  unsigned * col_ptrs_A = (unsigned*) malloc(sizeof(unsigned) * (A.N+1)); // this is the upperlimit
  memset (col_ptrs_A, 0, sizeof(unsigned) * (A.N+1));
  unsigned * col_ids_A = (unsigned*) malloc(sizeof(unsigned) * (A.N));  
  double * col_vals_A = (double*) malloc(sizeof(double) * (A.N));  // stores the val
  col_ptrs_A[0] = 0;
  col_ptrs_A[1] = 1;
  
  col_ids_A[0] = A_mtx[1]; //myMap[&(my_tuples_list[0])] - 1;
  unsigned max_col_id_A = col_ids_A[0];
  //col_vals_A[0] =  
  col_vals_A[0] = A.val_array[A_mtx[2]];


 for(unsigned ii=1; ii < A.N; ++ii) {
  col_vals_A[ii] = A.val_array[A_mtx[ii*3+2]];
        col_ids_A[ii] = A_mtx[ii*3 + 1]; 
        if (max_col_id_A < col_ids_A[ii]) 
            max_col_id_A = col_ids_A[ii];
         if(comp_items(&(A_mtx[(ii-1)*3]) , &(A_mtx[ii*3]))  != 0)
         {
             offset_A++;
             col_ptrs_A[offset_A] += col_ptrs_A[offset_A-1] + 1; 
         }

         else
             col_ptrs_A[offset_A] += 1; 
 }

 // Shrink col_ptrs_A[] to size (offset_A+1)
 if (offset_A < A.N) {
    unsigned* temp = (unsigned*)realloc(col_ptrs_A, (offset_A+1) * sizeof(unsigned));
         if (temp == NULL)
         {
             cerr << "Out of memory! Aborting!!" << endl;
             exit(1);
         }
         col_ptrs_A = temp;

 }

 // offset_A is equal to (number of cols).. it is correctly calculated --> CHECKED!!


  


#endif


   // converting B to CSC 

  // uncomment this block to convert to CSC.
  unsigned offset_B = 1;
  unsigned * col_ptrs_B = (unsigned*) malloc(sizeof(unsigned) * (B.N+1)); // this is the upperlimit
  memset (col_ptrs_B, 0, sizeof(unsigned) * (B.N+1));
  unsigned * col_ids_B = (unsigned*) malloc(sizeof(unsigned) * (B.N));  
  double * col_vals_B = (double*) malloc(sizeof(double) * (B.N));  // stores the val
  col_ptrs_B[0] = 0;
  col_ptrs_B[1] = 1;
  
  col_ids_B[0] = B_mtx[1]; //myMap[&(my_tuples_list[0])] - 1;
  unsigned max_col_id_B = col_ids_B[0];
  //col_vals_B[0] =  
  col_vals_B[0] = B.val_array[B_mtx[2]];


  // TODO: intertwine/interleave the hashing of the contraction indicies here

 for(unsigned ii=1; ii < B.N; ++ii) {
        col_vals_B[ii] = B.val_array[B_mtx[ii*3+2]];
        col_ids_B[ii] = B_mtx[ii*3 + 1]; 
        if (max_col_id_B < col_ids_B[ii]) 
            max_col_id_B = col_ids_B[ii];
         if(comp_items(&(B_mtx[(ii-1)*3]) , &(B_mtx[ii*3]))  != 0)
         {
             offset_B++;
             col_ptrs_B[offset_B] += col_ptrs_B[offset_B-1] + 1; 
         }

         else
             col_ptrs_B[offset_B] += 1; 
 }

 // Shrink col_ptrs_B[] to size (offset_B+1)
 if (offset_B < B.N) {
    unsigned* temp = (unsigned*)realloc(col_ptrs_B, (offset_B+1) * sizeof(unsigned));
         if (temp == NULL)
         {
             cerr << "Out of memory! Aborting!!" << endl;
             exit(1);
         }
         col_ptrs_B = temp;

 }

 // offset_B is equal to (number of cols).. it is correctly calculated --> CHECKED!!

    chrono::high_resolution_clock::time_point t800 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t800 - t700).count();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t800 - t700).count();
    cout << "time for sorting+creating the CSC format for A and B = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;
  

#if 1
    /* populating the data structure used by CXSparse */

    // For A_mtx
   
    cs* A_csc = new cs; // the csc format matrix to be used by CXSparse

    //printf("nzmax = %d, nrows (m) = %d, ncols (n) = %d, %d\n", A->nzmax, A->m, A->n, A->nz);
    A_csc->nzmax = A.N;
    A_csc->nz = -1;

   A_csc->m = max_col_id_A + 1; 
   A_csc->n = offset_A;


  A_csc->p = (int32_t*) malloc(sizeof(int32_t) * ((A_csc->n) + 1)); 
  A_csc->i = (int32_t*) malloc(sizeof(int32_t) * (A_csc->nzmax));  
  A_csc->x = (double*) malloc(sizeof(double) * (A_csc->nzmax));  // stores the val

  for(unsigned ii=0; ii < A.N; ++ii) {
     (A_csc->i)[ii] = col_ids_A[ii];
     (A_csc->x)[ii] = col_vals_A[ii];
  }

  for(unsigned ii=0; ii < offset_A+1; ++ii) {
     (A_csc->p)[ii] = col_ptrs_A[ii];
  }


    cs* B_csc = new cs; // the csc format matrix to be used by CXSparse

    //printf("nzmax = %d, nrows (m) = %d, ncols (n) = %d, %d\n", B->nzmax, B->m, B->n, B->nz);
    B_csc->nzmax = B.N;
    B_csc->nz = -1;

   B_csc->m = max_col_id_B + 1; 
   B_csc->n = offset_B;

  B_csc->p = (int32_t*) malloc(sizeof(int32_t) * ((B_csc->n) + 1)); 
  B_csc->i = (int32_t*) malloc(sizeof(int32_t) * (B_csc->nzmax));  
  B_csc->x = (double*) malloc(sizeof(double) * (B_csc->nzmax));  // stores the val

  for(unsigned ii=0; ii < B.N; ++ii) {
     (B_csc->i)[ii] = col_ids_B[ii];
     (B_csc->x)[ii] = col_vals_B[ii];
  }

  for(unsigned ii=0; ii < offset_B+1; ++ii) {
     (B_csc->p)[ii] = col_ptrs_B[ii];
  }

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Total time for matricization = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;


    chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();

    cs* C_csc = NULL;
    C_csc= cs_multiply (A_csc, B_csc) ;

    if(C_csc == NULL) {
        cerr << "CXSparse returned a NULL output object!! Check the input!" << endl;
      exit(1);
    }

    chrono::high_resolution_clock::time_point t4 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t4 - t3).count();
    cout << "Total time for spGEMM = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;

    chrono::high_resolution_clock::time_point t5 = chrono::high_resolution_clock::now();

    Tensor C;
    C.d = (A.d - A.num_key_dimensions + B.d - B.num_key_dimensions);
    C.num_key_dimensions = C.d;
    C.N = C_csc->nzmax;
    C.hedges_array = (unsigned*) malloc(sizeof(unsigned) * C.d * C.N);
    C.val_array = (double*) malloc(sizeof(double) *  C.N);


  unsigned counter = 0;
  for(unsigned ii = 0; ii < C_csc->n; ++ii) {

      for (unsigned p = (C_csc->p)[ii] ; p < (C_csc->p)[ii+1] ; p++) {

          //cout << ii << " " << (C_csc->i)[p] << " " << (C_csc->x)[p] << endl;

       for(unsigned p1 = 0; p1 < A.d - A.num_key_dimensions; ++p1)
           //C_coo[counter*(A.d - A.num_key_dimensions + B.d - B.num_key_dimensions) + p1] = myMap_A_reverse[(C_csc->i)[p] * (A.d - A.num_key_dimensions) + p1]; // this gives the tuple corresponding to the row id
           C.hedges_array[counter*(C.d) + p1] = myMap_A_reverse[(C_csc->i)[p] * (A.d - A.num_key_dimensions) + p1]; // this gives the tuple corresponding to the row id

       for(unsigned p1 = 0; p1 < B.d - B.num_key_dimensions; ++p1)
           C.hedges_array[counter*(C.d) + (A.d - A.num_key_dimensions) + p1] = myMap_B_reverse[ii*(B.d - B.num_key_dimensions) + p1];
// this gives the tuple corresponding to the col id


       C.val_array[counter] = (C_csc->x)[p];
       counter++;
      }
  }


    chrono::high_resolution_clock::time_point t6 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t6 - t5).count();
    cout << "Total time for converting result from CSC to COO = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;
    
// NOTE: CSparse computes the multiplication using the  column-column formulation and generates the output also in the CSC format, so no transpose is required.


#endif
 return C;
}
