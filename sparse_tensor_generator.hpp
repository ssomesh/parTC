#include <random>
#include <sys/time.h>

Tensor get_sparse_tensor (Tensor& A) {
std::mt19937_64  prng(time(NULL));
        
        Tensor B;
        std::uniform_real_distribution<> distribution(0.0, 1.0);

       double sparsification_factor = 1; // tensor B will contain around a factor of 'sparsification_factor' fewer nnz than tensor A

       assert(sparsification_factor > 0 && sparsification_factor <= 1);

       B.d = A.d;
       //B.N = (A.N + sparsification_factor - 1)/ sparsification_factor; // ceil of A.N/sparsification_factor courtsey integer division
       B.N = (unsigned) A.N * sparsification_factor;

       if (B.N <= 1) {
           cerr << "Tensor B has at most one nonzeros!!" << endl;
           exit(1);
       }

      B.num_key_dimensions = A.num_key_dimensions;
      B.dim_tc = (unsigned*) malloc (sizeof(unsigned) * B.d);


      for(unsigned i=0; i < B.d; ++i) {
          B.dim_tc[i] = A.dim_tc[i];
      }

      B.hedges_array = (unsigned *) malloc (sizeof(unsigned) *  B.N * B.d);
      B.val_array = (double *) malloc (sizeof(double) *  B.N);

      unsigned* shuffled_positions = (unsigned*) malloc(sizeof(unsigned) * A.N);

      for (unsigned i=1; i< A.N-1; ++i) 
          shuffled_positions[i] = i;


      for (unsigned i = 1; i < A.N-1; i++) {
            // choose index uniformly in [i, n-1]
            unsigned r = i +  (unsigned)(distribution( prng ) * ((A.N-1) - i));
            unsigned temp = shuffled_positions[r];
            shuffled_positions[r] = shuffled_positions[i];
            shuffled_positions[i] = temp;
        }


      // adding the first element of A as the first element of B
          for(unsigned xy = 0; xy < B.d; ++xy) {
             B.hedges_array[xy] = A.hedges_array[xy];
          }
          B.val_array[0] = A.val_array[0];


      for(unsigned ii=1; ii < B.N-1; ii++) {
         
          //unsigned i = shuffled_positions[ii];
          unsigned i = ii;


          for(unsigned xy = 0; xy < B.d; ++xy) {
             B.hedges_array[ii*B.d+xy] = A.hedges_array[i*A.d+xy];
          }

          B.val_array[ii] = A.val_array[i];


      }

      // adding the last element of A as the last element of B
          for(unsigned xy = 0; xy < B.d; ++xy) {
             B.hedges_array[(B.N-1)*B.d+xy] = A.hedges_array[(A.N-1)*A.d+xy];
          }

          B.val_array[B.N-1] = A.val_array[A.N-1];

      //assert(counter == B.N);

      B.dimensions_array = (unsigned*) malloc (sizeof(unsigned) * B.d); 

      for(unsigned j=0; j < B.d; ++j) {
          B.dimensions_array[j] = A.dimensions_array[j];
      }
      

        return B;
}
