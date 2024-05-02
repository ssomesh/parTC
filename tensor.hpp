#include <fstream>
#include <iostream>
#include <cstring>

using std::string;
using std::ifstream;
using std::cout;
using std::cerr;
using std::endl;

/*
Struct members order does make a difference

    It's best to order the struct members in decreasing or increasing order of size; it will minimize the required memory space and keep proper data alignment,

*/


struct /*alignas(4)*/ Tensor {
    unsigned d; // number of dimensions of the tensor
    unsigned N; // number of non-zeros in tensor
    unsigned num_key_dimensions; // the number of dimensions to be used as the hash key
    unsigned * hedges_array; 
    unsigned * dimensions_array;
    double * val_array; // stores the value of nonzeros
    unsigned * dim_tc; // stores the contraction dimensions followed by the remaining dimensions; it is an indirecton array since it stores the dimensions in a shuffled order

    Tensor(void) {
        d = 0u;
        N = 0u;
        num_key_dimensions = 0u;
        hedges_array = NULL;
        dimensions_array = NULL;
        val_array = NULL;
        dim_tc = NULL;

    }

    Tensor(Tensor& A) {
        // this is a constructor which is invoked for creating a tensor with the entries of A reordered

	N = A.N;
	d = A.d;
	num_key_dimensions = A.num_key_dimensions;
	dim_tc = (unsigned*) malloc (sizeof(unsigned) * d);

      for(unsigned i=0; i < d; ++i) 
          dim_tc[i] = A.dim_tc[i];

      hedges_array = (unsigned *) malloc (sizeof(unsigned) *  N * d);
      val_array = (double *) malloc (sizeof(double) *  N);

      dimensions_array = (unsigned*) malloc (sizeof(unsigned) * d); 

      for(unsigned j=0; j < d; ++j) {
          dimensions_array[j] = A.dimensions_array[j];
      }

    }


        d = (A.num_key_dimensions) + (B.d - B.num_key_dimensions) ;
        N = total_elements_in_product;
        if (d == 0 || N == 0) {
            cerr << "tensor contraction not possible!!" << endl;
            exit(1);
        }

        dimensions_array = (unsigned*) malloc (sizeof(unsigned) * d); // not really used! 
        memset (dimensions_array, 0, sizeof(unsigned) * d);

        num_key_dimensions = d; // all the d dimensions are to be used for hashing

        dim_tc = (unsigned*) malloc (sizeof(unsigned) * d); // allocate space for dim_tc

        for (unsigned i=0; i < d; ++i)
            dim_tc[i] = i;


        // for the COO representation of output tensor
        hedges_array = (unsigned *) malloc (sizeof(unsigned) *  N * d);
        val_array = (double *) malloc (sizeof(double) *  N); 


    }

    void populate_meta_info (std::string filename_meta) {
        ifstream in(filename_meta);
        if(!in.is_open()) {
            std::cerr << "Could not open the file \"" << filename_meta << "\""  << std::endl;
            exit(1);

        }
        string line;
        for(int i=0; i<2; ++i) // skipping the first 2 lines from the top
            getline (in, line);

        in >> num_key_dimensions;

        dim_tc = (unsigned*) malloc (sizeof(unsigned) * d); // allocate space for dim_tc

        for(unsigned i=0; i < d; ++i) { 
            unsigned temp;
            in >> temp;
            dim_tc[i] = temp;
        }


    }
};


struct /*alignas(4)*/ TensorSPA {
    unsigned d; // number of dimensions of the tensor
    unsigned N; // number of non-zeros in tensor
    unsigned * hedges_array; 
    unsigned * dimensions_array;
    double * val_array; // stores the value of nonzeros
    //unsigned * dim_tc; // stores the contraction dimensions followed by the remaining dimensions; it is an indirecton array since it stores the dimensions in a shuffled order
    //unsigned num_key_dimensions; // the number of dimensions to be used as the hash key

    TensorSPA(void) {
        d = 0u;
        N = 0u;
        //num_key_dimensions = 0u;
        hedges_array = NULL;
        dimensions_array = NULL;
        val_array = NULL;
        //dim_tc = NULL;

    }

};



