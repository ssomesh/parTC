#if 1

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <climits>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <omp.h>
#include "extern/fastmod-master/include/fastmod.h"
#include "fksCuckooSPA.hpp"

using namespace std;

unsigned int _global_start;
unsigned int _global_end;
int comp_items_descending(const void *h1,const void*h2)
{
    unsigned int *h1p = (unsigned int*) h1;
    unsigned int *h2p = (unsigned int*) h2;
    for (unsigned int i =  _global_start; i < _global_end; i++)
    {
        if (h1p[i] > h2p[i]) return -1;
        if (h1p[i] < h2p[i]) return 1;
    }
    return 0;
}

Tensor reorder_tensor (Hash_Table* HT_B, Tensor& B) {



    Tensor B_reordered (B);

    unsigned * old_to_new = (unsigned *) malloc (sizeof(unsigned) *  B.N); // old_to_new[old id] gives new id for an element

    unsigned new_id = 0; 

    for(unsigned i=0; i < HT_B->num_buckets; ++i) {
        if(new_id >= B.N) break;
        if ( HT_B->bucket_ptrs[i] == NULL ) continue;
        unsigned currSz = (HT_B->bucket_ptrs[i])->nslots;
        for(unsigned j = 0; j < currSz; ++j ) {
            if(((HT_B->bucket_ptrs[i])->aux_ptrs)[j] == NULL) continue;

            for(unsigned x = 0; x < (HT_B->bucket_ptrs[i]->aux_ptrs)[j]->size; ++x) {
                unsigned ele = (HT_B->bucket_ptrs[i]->aux_ptrs)[j]->aux_array[x];
                (HT_B->bucket_ptrs[i]->aux_ptrs)[j]->aux_array[x] = new_id; 
                old_to_new[ele] = new_id;
                new_id++;
            }
        }
    }



    for(unsigned i=0; i < B.N; ++i) {

        unsigned myNewId = old_to_new[i];

        for(unsigned xy = 0; xy < B.d; ++xy) {
            B_reordered.hedges_array[myNewId*B.d+xy] = B.hedges_array[i*B.d+xy];
        }

        B_reordered.val_array[myNewId] = B.val_array[i];


    }




    return B_reordered;

}


Tensor tc_fksCuckoo (Hash_Table* HT_A, Hash_Table* HT_B, Tensor& A, Tensor& B) {

    /* The function performs tensor contraction (A x B) with the specified contraction indices */

#if 1
    uint64_t elapsed1;
    chrono::high_resolution_clock::time_point t10 = chrono::high_resolution_clock::now();

    /*Reorder the nonzeros in the COO representation of B*/


    Tensor B_reordered = reorder_tensor (HT_B, B); 


    unsigned long long total_flops_in_product = 0;
    unsigned num_nonempty_rows_A = 0;


    for(unsigned i=0; i < HT_A->num_buckets /*A.N*/; ++i) {
        if ( HT_A->bucket_ptrs[i] == NULL) continue;
        unsigned currSz = (HT_A->bucket_ptrs[i])->nslots;
        for(unsigned j = 0; j < currSz; ++j ) {
            if(((HT_A->bucket_ptrs[i])->aux_ptrs)[j] == NULL) continue;
            num_nonempty_rows_A++;
        }
    }

    unsigned * num_nonzeros_per_row_C = (unsigned*) malloc (sizeof(unsigned) * num_nonempty_rows_A); 
    unsigned * row_id_A = (unsigned*) malloc (sizeof(unsigned) * num_nonempty_rows_A); 
    for(unsigned ii=0; ii < num_nonempty_rows_A; ++ii) {
        num_nonzeros_per_row_C[ii] = 0;
    }

    unsigned *contraction_indices_A = (unsigned*) malloc(sizeof(unsigned) * (A.d - A.num_key_dimensions)) ; //this is used to search for the contraction indices in HT_B
    Ele_pos ele_pos_B; 


    unsigned counter = 0;
    for(unsigned i=0; i < HT_A->num_buckets /*A.N*/; ++i) {
        if(counter >= num_nonempty_rows_A) break;
        if ( HT_A->bucket_ptrs[i] == NULL ) continue;

        unsigned currSz = (HT_A->bucket_ptrs[i])->nslots;
        for(unsigned j = 0; j < currSz; ++j ) {
            if(((HT_A->bucket_ptrs[i])->aux_ptrs)[j] == NULL) continue;

            row_id_A[counter] = ((HT_A->bucket_ptrs[i]->aux_ptrs)[j]->aux_array[0]);


            for(unsigned x = 0; x < (HT_A->bucket_ptrs[i]->aux_ptrs)[j]->size; ++x) {

                unsigned ele = (HT_A->bucket_ptrs[i]->aux_ptrs)[j]->aux_array[x];


                unsigned ele_key_indices_pos = 0;
                for(unsigned cc=A.num_key_dimensions; cc < A.d; ++cc) {
                    contraction_indices_A[cc-A.num_key_dimensions] = A.hedges_array[ele * A.d + A.dim_tc[cc]];
                }


                if (!(HT_B->find_element_new (ele_pos_B, B_reordered, contraction_indices_A))) continue;

                unsigned mysize_B = (HT_B->bucket_ptrs[ele_pos_B.bid]->aux_ptrs)[ele_pos_B.sid]->size;

                total_flops_in_product += mysize_B;
                num_nonzeros_per_row_C[counter] += mysize_B;


            }



            counter++;
        }
    }

    cout << "number of flops in tensor contraction = " << total_flops_in_product << endl;

    if (total_flops_in_product > 100000000000) 
        cout << "number of flops in tensor contraction > 10^11" << endl;
    //exit(1); // for debugging

    Tensor O (A, B_reordered, (unsigned) total_flops_in_product); // this is the output tensor.
    unsigned ele_count_O = 0U; // number of elements in the tensor O

    chrono::high_resolution_clock::time_point t20 = chrono::high_resolution_clock::now();
    elapsed1 = chrono::duration_cast<chrono::nanoseconds>(t20 - t10).count();
    cout << "Total time for preprocessing before tensor contraction = " << (double)(elapsed1 * 1.E-9 ) << " (s)" << endl;
#endif



#if 1

    uint64_t elapsed;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    unsigned nThreads = 1; //omp_get_num_threads();

    omp_set_num_threads(nThreads); // Use  nThreads for all consecutive parallel regions

    unsigned num_contraction_indices_A = A.d - A.num_key_dimensions;

    unsigned *ele_indices_O = (unsigned*) malloc(sizeof(unsigned) * O.d * nThreads) ; 
    unsigned *contraction_indices_A_i = (unsigned*) malloc(sizeof(unsigned) * num_contraction_indices_A * nThreads) ; 

#pragma omp parallel for schedule (dynamic) // dynamic scheduling
    for(unsigned i = 0 ; i < num_nonempty_rows_A ; ++i) {

        if(num_nonzeros_per_row_C[i] > 0) { 

        unsigned tid = omp_get_thread_num();


        Ele_pos ele_pos_B_i; 

        Ele_pos ele_pos_A; 
        Ele_pos ele_pos; 

        unsigned * curr_indices_A = (unsigned*) malloc(sizeof(unsigned) * A.num_key_dimensions); 
        unsigned *ele_key_indices = (unsigned*) malloc(sizeof(unsigned) * (B_reordered.d - B_reordered.num_key_dimensions)); 



        TensorSPA SPA;
        SPA.d = B_reordered.d - B_reordered.num_key_dimensions ; 
        SPA.N = num_nonzeros_per_row_C[i];



        SPA.hedges_array = (unsigned *) malloc (sizeof(unsigned) *  SPA.N * SPA.d);
        SPA.val_array = (double *) malloc (sizeof(double) *  SPA.N);


        SPA.dimensions_array = (unsigned*) malloc (sizeof(unsigned) * SPA.d);

        for(unsigned xcx = 0; xcx < SPA.d; ++xcx)
            SPA.dimensions_array[xcx] = 0;

        unsigned ele_count_SPA = 0; 

        unsigned ii = row_id_A[i]; 
        for(unsigned cc=0; cc < A.num_key_dimensions; ++cc) {
            curr_indices_A[cc] = A.hedges_array[ii* A.d + A.dim_tc[cc]];
            ele_indices_O[tid * O.d + cc] = curr_indices_A[cc]; 
        }
        HT_A->find_element_new (ele_pos_A, A, curr_indices_A); 
        if ((HT_A->bucket_ptrs[ele_pos_A.bid]->aux_ptrs)[ele_pos_A.sid]->size == 1) {


            unsigned ele0_gl = (HT_A->bucket_ptrs[ele_pos_A.bid]->aux_ptrs)[ele_pos_A.sid]->aux_array[0];
	    double curr_val_A = A.val_array[ele0_gl];
            for(unsigned cc=A.num_key_dimensions; cc < A.d; ++cc) {
                contraction_indices_A_i[(num_contraction_indices_A * tid) + cc-A.num_key_dimensions] = A.hedges_array[ele0_gl* A.d + A.dim_tc[cc]];
            }

            if(HT_B->find_element_new (ele_pos_B_i, B_reordered, &contraction_indices_A_i[(num_contraction_indices_A * tid)]))
            {

                for(unsigned j=0; j < (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->size; ++j) {

                    unsigned ele2 = (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->aux_array[j];


                    double curr_val = curr_val_A  * B_reordered.val_array[ele2];

                    for(unsigned cc = 0; cc < SPA.d; ++cc)
                        SPA.hedges_array[ele_count_SPA * SPA.d + cc] = B_reordered.hedges_array[ele2 * B_reordered.d + B_reordered.dim_tc[cc+B_reordered.num_key_dimensions]];

                    SPA.val_array[ele_count_SPA] = curr_val;

                    ele_count_SPA++;



                }
            }

        }


        else { 

            Hash_Table_SPA* HT_SPA = new Hash_Table_SPA(SPA);




            unsigned ele0_gl = (HT_A->bucket_ptrs[ele_pos_A.bid]->aux_ptrs)[ele_pos_A.sid]->aux_array[0];
	    double curr_val_A = A.val_array[ele0_gl];

            for(unsigned cc=A.num_key_dimensions; cc < A.d; ++cc) {
                contraction_indices_A_i[(num_contraction_indices_A * tid) + cc-A.num_key_dimensions] = A.hedges_array[ele0_gl* A.d + A.dim_tc[cc]];
            }

            if(HT_B->find_element_new (ele_pos_B_i, B_reordered, &contraction_indices_A_i[(num_contraction_indices_A * tid)])) {


                for(unsigned j=0; j < (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->size; ++j) {

                    unsigned ele2 = (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->aux_array[j];

                    double curr_val = curr_val_A  * B_reordered.val_array[ele2];

                    for(unsigned cc = 0; cc < SPA.d; ++cc)
                        SPA.hedges_array[ele_count_SPA * SPA.d + cc] = B_reordered.hedges_array[ele2 * B_reordered.d + B_reordered.dim_tc[cc+B_reordered.num_key_dimensions]];

                    SPA.val_array[ele_count_SPA] = curr_val;

                    HT_SPA->insert_element (ele_count_SPA, SPA);

                    ele_count_SPA++;



                }
            }



            for(unsigned k=1; k < (HT_A->bucket_ptrs[ele_pos_A.bid]->aux_ptrs)[ele_pos_A.sid]->size; ++k) {
                unsigned ele1 = (HT_A->bucket_ptrs[ele_pos_A.bid]->aux_ptrs)[ele_pos_A.sid]->aux_array[k];
		double curr_val_ele1 = A.val_array[ele1];
                for(unsigned cc=A.num_key_dimensions; cc < A.d; ++cc) {
                    contraction_indices_A_i[(num_contraction_indices_A * tid) + cc-A.num_key_dimensions] = A.hedges_array[ele1* A.d + A.dim_tc[cc]];
                }

                if (!(HT_B->find_element_new (ele_pos_B_i, B_reordered, &contraction_indices_A_i[(num_contraction_indices_A * tid)]))) continue;



                for(unsigned j=0; j < (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->size; ++j) {

                    unsigned ele2 = (HT_B->bucket_ptrs[ele_pos_B_i.bid]->aux_ptrs)[ele_pos_B_i.sid]->aux_array[j];

                    for(unsigned cc=B_reordered.num_key_dimensions; cc < B_reordered.d; ++cc) {
                        ele_key_indices[cc-B_reordered.num_key_dimensions] = B_reordered.hedges_array[ele2 * B_reordered.d + B_reordered.dim_tc[cc]];
                    }

                    double curr_val = curr_val_ele1  * B_reordered.val_array[ele2];


                    if(HT_SPA->find_element_new (ele_pos, SPA, ele_key_indices)) { // element is found in the output hash table
                        unsigned myindex = (HT_SPA->bucket_ptrs[ele_pos.bid]->aux_ptrs)[ele_pos.sid]->aux_array[0];
                        SPA.val_array[myindex] += curr_val;

                    }

                    else {  

                        for(unsigned cc = 0; cc < SPA.d; ++cc)
                            SPA.hedges_array[ele_count_SPA * SPA.d + cc] = ele_key_indices[cc];
                        SPA.val_array[ele_count_SPA] = curr_val;

                        bool isInserted = HT_SPA->insert_element_at_bucket (ele_count_SPA, ele_pos.bid, SPA);

                        ele_count_SPA++;
                    }



                }

            }


            delete (HT_SPA);
        } 

        /*Writing the contents of the sparse accumulator to the output tensor, O */

        unsigned start_index = __atomic_fetch_add(&ele_count_O, ele_count_SPA, __ATOMIC_RELAXED);

        for(unsigned xx = 0; xx < ele_count_SPA; ++xx) {
            for(unsigned cc=0; cc < SPA.d; ++cc) {
                ele_indices_O[tid * O.d + A.num_key_dimensions + cc] = SPA.hedges_array[xx * SPA.d +cc];
            }
            for(unsigned cc = 0; cc < O.d; ++cc)
                O.hedges_array[(start_index + xx) * O.d + cc] = ele_indices_O[tid * O.d + cc];
            O.val_array[start_index + xx] = SPA.val_array[xx];

        }
    }
 }


    O.N = ele_count_O;

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Total time for tensor contraction = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;
    cout << "Total number of nonzeros in the output tensor = " << ele_count_O << endl;
#endif

   
    return O;

}

#endif
