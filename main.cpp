/* This is the driver file for all experiments */

#include<iostream>
#include<iomanip>
#include<string>
#include<vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <ctime> 
#include <limits>
#include <unistd.h>
#include <cstdlib>
#include <getopt.h>

#include "tensor.hpp"
#include "mtxTnsIO.hpp"
#include "fksCuckoo.hpp"
#include "fksCuckooTC.hpp"
#include "matricizedspGEMM.hpp"
#include "matricizedspGEMM_sorting.hpp"
#include "sparse_tensor_generator.hpp"

using namespace std;



int main(int argc, char* argv[])
{
	string fileNameA, fileName_A_meta; 




	unsigned int numTests ;	
    int isMtx = 0; // by default set it to not a matrix
	unsigned int d, N;
	string msg, dimMsg;

    if(argc < 5)
	{
		cerr<<"input not parsed correctly!\n"<<endl;

	cerr << "Usage : \n\t"<< argv[0] << "\n"
	"\tfilename tensor A   : tensor A\n"
	"\tfilename tensor A meta info : contraction specific info for tensor A\n"
	"\tisMtx : 0 => tensor; 1 => matrix\n"
	"\tnumber of rounds : number of times to run the experiment\n";

		return 1;
	}

    fileNameA = string(argv[1]);
    fileName_A_meta = string(argv[2]);
    isMtx = atoi(argv[3]);
    numTests = atoi(argv[4]);

    Tensor A;
		if(isMtx)		
			readMatrix(fileNameA, &(A.hedges_array), &(A.dimensions_array), &(A.val_array), A.d, A.N);		 // read a matrix
		else		
        	readTensor(fileNameA, &(A.hedges_array), &(A.dimensions_array), &(A.val_array), A.d, A.N); // read a tensor (read in the frostt file using PIGO)

        A.populate_meta_info(fileName_A_meta); 


      Tensor A_copy; // with the noncontraction indices in the beginning of dim_tc;


       A_copy.d = A.d;
       A_copy.N = A.N;

      A_copy.num_key_dimensions = A.d - A.num_key_dimensions;
      A_copy.dim_tc = (unsigned*) malloc (sizeof(unsigned) * A_copy.d);


      for(unsigned i=A.num_key_dimensions; i < A.d; ++i) {
          A_copy.dim_tc[i - A.num_key_dimensions] = A.dim_tc[i];
      }

      for(unsigned i=0; i < A.num_key_dimensions; ++i) {
          A_copy.dim_tc[A.d - A.num_key_dimensions + (i)] = A.dim_tc[i];
      }


      A_copy.hedges_array = (unsigned *) malloc (sizeof(unsigned) *  A_copy.N * A_copy.d);
      A_copy.val_array = (double *) malloc (sizeof(double) *  A_copy.N);

      
      for(unsigned i=0; i < A.N; i++) {
         
          for(unsigned xy = 0; xy < A_copy.d; ++xy) {
             A_copy.hedges_array[i*A_copy.d+xy] = A.hedges_array[i*A.d+xy];
          }

          A_copy.val_array[i] = A.val_array[i];

      }

      A_copy.dimensions_array = (unsigned*) malloc (sizeof(unsigned) * A_copy.d); 

      for(unsigned j=0; j < A_copy.d; ++j) {
          A_copy.dimensions_array[j] = A.dimensions_array[j];
      }

      for(unsigned j=0; j < A_copy.d; ++j) {
          cout << A_copy.dimensions_array[j] << " \n"[j+1 == A_copy.d];
      }


      Tensor B = get_sparse_tensor(A);


      Hash_Table* HT_B = buildHashTable(B); // create the hash data structure for B
      Hash_Table* HT_A = buildHashTable(A_copy); // create the hash data structure for A_copy

       Tensor O = tc_fksCuckoo (HT_A, HT_B, A_copy, B); // performs tensor contraction A x B with the specified contraction indices

     Tensor D = matricize_sort_spGEMM(A,B);


     Tensor C = matricize_spGEMM(A,B); //spMM using cxsparse

       





#if 1

  // verify the correctness of the result

  if(C.N != O.N) {
      cout << C.N << " != " << O.N << endl;
      cerr << "Erroneous result! nnz mismatch." << endl;
      exit(1);
  }


  double* O_val_array = (double*) malloc (sizeof (double) * O.N);
  double* C_val_array = (double*) malloc (sizeof (double) * C.N);
  for (unsigned ii=0; ii < O.N; ++ii) {
     O_val_array[ii] = O.val_array[ii];
     C_val_array[ii] = C.val_array[ii];
  }

  std::sort(O_val_array, O_val_array+(O.N));
  std::sort(C_val_array, C_val_array+(C.N));

  for (unsigned ii=0; ii < O.N; ++ii) {
       if ( (C_val_array)[ii] != O_val_array[ii] ) {
            cerr << "Error in result!" << endl;
            exit(1);
       }
  }
#endif

#if 1

  // verify the correctness of the result obtained via the two matricization methods

  if(C.N != D.N) {
      cout << C.N << " != " << D.N << endl;
      cerr << "Erroneous result! nnz mismatch." << endl;
      exit(1);
  }


  double* D_val_array = (double*) malloc (sizeof (double) * D.N);
  //double* C_val_array = (double*) malloc (sizeof (double) * C.N);
  for (unsigned ii=0; ii < D.N; ++ii) {
     D_val_array[ii] = D.val_array[ii];
     C_val_array[ii] = C.val_array[ii];
  }

  std::sort(D_val_array, D_val_array+(D.N));
  std::sort(C_val_array, C_val_array+(C.N));

  for (unsigned ii=0; ii < D.N; ++ii) {
       if ( (C_val_array)[ii] != D_val_array[ii] ) {
            cerr << "Error in result!" << endl;
            exit(1);
       }
  }
#endif


return 0;
}

