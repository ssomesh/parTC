
#if 1

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <climits>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include "extern/fastmod-master/include/fastmod.h"
//#include "matchmaker.h"

using namespace std;

//#define SANITY_DEBUG 1


/* computes y = x mod pp*/
#define modpMersenneExtSPA(y, x, pp, qq)  \
{                                  \
    y = ((x) & (pp)) + ((x)>>(qq));\
    y = y >= (pp) ? y-= (pp) : y;  \
}



//utility functions

inline bool isEqualSPA(unsigned ele1, unsigned ele2, TensorSPA& A) {

        unsigned n = A.d;
        unsigned ele1_memoized = ele1*n;
        unsigned ele2_memoized = ele2*n;

        for (unsigned i = 0; i < n; i++){
            if( A.hedges_array[ele1_memoized + i] != A.hedges_array[ele2_memoized + i] ) {
                return false;
            }
        }

        return true;


}

// overloaded function
inline bool isEqualSPA(unsigned* arr, unsigned ele, TensorSPA& A) { 

        unsigned n = A.d;
        unsigned ele_memoized = ele*n;

        for (unsigned i = 0; i < n; i++){
            if( arr[i] != A.hedges_array[ele_memoized + i] ) {
                return false;
            }
        }

        return true;
}




inline unsigned long long get_index_SPA (unsigned ele, TensorSPA& A, unsigned long long * kDict, unsigned keyid, unsigned p, unsigned q, unsigned n) {

    // this is used for obtaining a slot in the second level hashing

    unsigned ele_memoized = ele * A.d;
    unsigned num_key_dimensions = A.d;
    unsigned key_memoized = keyid * num_key_dimensions;
    //unsigned N2 = num_key_dimensions % 2;
        unsigned N2 = (1U) & num_key_dimensions; //computing ((b - 1) & a), Here b = 2; a = num_key_dimensions. It is equivalent to a % b where b is a  power of 2
    unsigned long long  somme = 0;
        for (unsigned j = 0; j < N2; j++){
            somme += (1 + A.hedges_array[ele_memoized + j]) * kDict[key_memoized + j];
        }

        unsigned long long somme_0 = 0;
        unsigned long long somme_1 = 0;
        for (unsigned j = N2; j < num_key_dimensions; j += 2){
            somme_0 += (1 + A.hedges_array[ele_memoized + j]) * kDict[key_memoized + j];
            somme_1 += (1 + A.hedges_array[ele_memoized + j+1]) * kDict[key_memoized + j + 1];
        }
            somme += somme_0 + somme_1;

    unsigned long long indice;
        modpMersenneExtSPA(indice, somme, p, q);
        //indice = indice % n; // this is correct since there are nslots many neighbors of num_distinct_elements many items

        return (indice % n);

}

void findMersenneExtSPA(unsigned N, unsigned &p, unsigned &q)
{

    vector <unsigned> qqs ({2, 3, 5, 7, 13, 17, 19, 31});
    vector <unsigned> pps = qqs;
    unsigned one  =1;
    p = q = 0;
    for (unsigned i =0 ;i < pps.size(); i++)
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

unsigned randomnumberintExtSPA(unsigned p) {/*from https://stackoverflow.com/questions/56435506/rng-function-c*/
    // Making rng static ensures that it stays the same
    // Between different invocations of the function
    static std::mt19937 rng;

    std::uniform_int_distribution<uint32_t> dist(0, p-1); 
    return dist(rng); 
}

bool est_premierExtSPA(unsigned i)
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

void trouver_premierExtSPA(const unsigned m, unsigned&p)
{
    unsigned i = m+1;
    while (!est_premierExtSPA(i))
        i+=1;
    p=i;
}

//struct Ele_pos {
//    unsigned bid; // bucket id
//    unsigned sid; // slot id within the bucket
//};


struct /*alignas(4)*/ Aux_data_SPA {
    // store the array of integers, which are the ids in the COO data structure
    unsigned * aux_array;
    // Growable buffer for ids

    // Capacity of aux_array
    //unsigned capacity; 

    // Number of entries actually in aux_array
    //unsigned size; 

    void insert(unsigned);

    Aux_data_SPA();
};

struct /*alignas(4)*/ BucketSPA {
    unsigned num_distinct_elements; // keeps a count of the number of distinct "contraction indices values" in the bucket. The contraction indices are the indices that are used for hashing throughout.
    unsigned nslots; // number of slots for cuckoo hashing
    unsigned k[2]; // keys for cuckoo hash within a bucket.. (stores the ids of kDict)
    Aux_data_SPA ** aux_ptrs;  // this array of pointers forms a bucket. Each pointer in this array points to an Aux_data_SPA struct.

    //note: even if there is a single unique element, create an aux_array for it

    BucketSPA();

    void manageBucketSPASize();

    void cuckooHash(unsigned eleId, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned); // perform cuckoo hash.. it will potentially update k[] and aux_ptrs

  void augment (unsigned currSlot, Aux_data_SPA** prev, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned);

  void augment (unsigned slotId, Aux_data_SPA** prev, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned k1, unsigned k2, unsigned p, unsigned q);
    void rehash(unsigned eleId, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned);


};

struct /*alignas(4)*/ Hash_Table_SPA {  // the entire hash structure for a tensor

    unsigned size; // number of elements in the hash table
    unsigned p, q;
    unsigned MAX_KEYS;
    unsigned num_buckets; // user supplied. usually we inialize it to N
    uint64_t M_fastmod; // used for computing the mod efficiently for first level hashing

    boost::dynamic_bitset< > bucket_ptrs_bits; // bit vector to check of size 'nslots' bits; all 0's by default
    BucketSPA** bucket_ptrs; // this is an array of pointers to buckets. the size of the array should be num_buckets

    // Note: have an array of pointers to buckets. Then, the hash_table will have an array of 'num_buckets' many pointers. We'll allocate space for these only when we need to. Otherwise, they will be pointing to NULL for empty buckets. 

    unsigned long long * kDict; // stores the actual l-tuple; where l is the number of dimensions to be used as key


    Hash_Table_SPA(TensorSPA&);  // constructor. it initializes the num_buckets and allocates space for the buckets. 

    bool insert_element (unsigned, TensorSPA&); // insert one element into the bucket. // returns a flag value to indicate success or failure
    bool insert_element_at_bucket (unsigned, unsigned, TensorSPA&); // insert one element into the specified bucket id. // returns a flag value to indicate success or failure
    bool find_element(unsigned, Ele_pos&, TensorSPA&); // test if an element is present in the hash table. It returns true if the search is successful, false otherwise. 

    bool find_element_new(Ele_pos&, TensorSPA&, unsigned * ele_key_incides); // test if an element is present in the hash table. It returns true if the search is successful, false otherwise. 
    

    unsigned get_bucket_id(unsigned, TensorSPA&);
    void perform_second_level_hashing(unsigned, TensorSPA&, unsigned); // possibly return a bool for status
    //void rehash (); // perform a rehash of all the elements present in the hash table so far.
};

Aux_data_SPA::Aux_data_SPA() {
    //capacity = 1; // start at capacity = 1 then double
    //size = 0;

    aux_array = (unsigned*) malloc (sizeof(unsigned*));
}

void Aux_data_SPA::insert(unsigned eleId) {

    // append current element to buffer
    aux_array[0] = eleId;

}


BucketSPA::BucketSPA() {
    num_distinct_elements = 1;
    nslots = 1; // 2^0
    //k[0] = -1;
    //k[1] = -1;
    aux_ptrs =  (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*)); // allocate space for a single element
}



void BucketSPA::manageBucketSPASize() {

    // grow buffer if necessary

    if ((num_distinct_elements + num_distinct_elements + 2) > nslots) // 2*(num_distinct_elements+1)... the num_distinct_elements here are before incrementing the value
    {

        unsigned old_nslots = nslots;

        if (nslots == 4) nslots = 16;

        //if (nslots <= (UINT_MAX / 2))
            //nslots += nslots;
        else   nslots = nslots << 1;
      //  else
      //  {
      //      cerr << "Too many elements to insert! Aborting!!" << endl;
      //      exit(1);
      //  }

        // extend buffer's capacity
        Aux_data_SPA** temp = (Aux_data_SPA**)realloc(aux_ptrs, nslots * sizeof(Aux_data_SPA*));
        if (temp == NULL)
        {
            cerr << "Out of memory! Aborting!!" << endl;
            exit(1);
        }
        aux_ptrs = temp;

        //Attention: A linear scan of nslots/2
        for(unsigned i= old_nslots; i < nslots; i +=2 ) { // nslots/2 will also be a multiple of 2 since we always start at nslots = 8
            aux_ptrs[i] = NULL; // inialize the new slots to NULL because comparison with NULL later
            aux_ptrs[i+1] = NULL; // inialize the new slots to NULL because comparison with NULL later
        }

    }

}

void BucketSPA::rehash(unsigned eleId, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {
    //cout << "rehashed!!" << endl;

    Aux_data_SPA ** aux_ptrs_copy = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * (num_distinct_elements + 1)); // a densely packed array containing all elements
    Aux_data_SPA ** items_left_after_initialization = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * (num_distinct_elements + 1));

    for(unsigned ii=0, counter = 0; ii < nslots; ++ii) { // the nslots being used here is after potentially being resized.. it may be better to make the copy before resizing the bucket
        if(aux_ptrs[ii] == NULL) continue;
        aux_ptrs_copy[counter++] = aux_ptrs[ii];
    }

    // also adding the new element to the packed array, which will contain all the elements to be inserted
    aux_ptrs_copy[num_distinct_elements] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
    aux_ptrs_copy[num_distinct_elements]->aux_array[0] = eleId;
    //aux_ptrs_copy[num_distinct_elements]->size = 1;

    //aux_ptrs_copy[num_distinct_elements]->insert(eleId); 

    manageBucketSPASize(); // resizing the bucket, if required
   // Aux_data_SPA ** all_NULL_arr = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * (nslots)); 
//    for(int ii=0; ii < nslots; ++ii) {
//        all_NULL_arr[ii] = NULL;
//    }
  //  for(int ii = 0; ii < nslots; ii += 4) { // nslots will always be a multiple of 4
  //      all_NULL_arr[ii] = NULL;
  //      all_NULL_arr[ii+1] = NULL;
  //      all_NULL_arr[ii+2] = NULL;
  //      all_NULL_arr[ii+3] = NULL;
  //  }

    for(unsigned k1 = k[0]; k1 < MAX_KEYS; ++k1 ) {
        for (unsigned k2 = (k1 == k[0]) ? (k[1]+1) : 0; k2 < MAX_KEYS; ++k2) {
            if (k1 == k2) continue;

            // reinitialize aux_ptrs[] to all NULL
            //memcpy(aux_ptrs, all_NULL_arr, sizeof(Aux_data_SPA*) * nslots);
  for(unsigned ii=0; ii < nslots; ii += 2) {
      aux_ptrs[ii] = NULL;
      aux_ptrs[ii+1] = NULL;
    //  aux_ptrs[ii] = all_NULL_arr[ii];
    //  aux_ptrs[ii+1] = all_NULL_arr[ii+1];
  }

            bool fg = false;

            /*Initialization for finding a matching */
            unsigned items_left_after_initialization_size = 0;
            for (unsigned i=0; i < num_distinct_elements+1; ++i) { // master loop over the items to rehash
                Aux_data_SPA* curr_item_ptr = aux_ptrs_copy[i]; // the element to insert
                unsigned currEle = curr_item_ptr->aux_array[0];

                unsigned currSlot1 = (unsigned) get_index_SPA (currEle, A, kDict, k1, p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index_SPA (currEle, A, kDict, k2, p, q, nslots);

                if (currSlot1 == currSlot2) {
                    fg = true;
                    break;

                }

                if (aux_ptrs[currSlot1] == NULL) {
                    aux_ptrs[currSlot1] = curr_item_ptr;
                    continue;
                }
                if (aux_ptrs[currSlot2] == NULL) {
                    aux_ptrs[currSlot2] = curr_item_ptr;
                    continue;
                }
                items_left_after_initialization[items_left_after_initialization_size++] = curr_item_ptr; // this element will be added using augmentation

            }
            if(fg) {continue;} // try the next k2

                bool insert_success = true;
                if(items_left_after_initialization_size > 0)
        {

            // attempting to insert the elements that could not be inserted during initialization
    Aux_data_SPA ** prev = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * nslots);
    Aux_data_SPA ** queue = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * nslots);

    boost::dynamic_bitset< > visited(nslots); // bit vector of size 'nslots' bits; all 0's by default
            for(unsigned i=0; i < items_left_after_initialization_size; ++i) {
                insert_success = false;

  //          memcpy(prev, all_NULL_arr, sizeof(Aux_data_SPA*) * nslots);
  for(unsigned ii=0; ii < nslots; ii += 2) {
      prev[ii] = NULL;
      prev[ii+1] = NULL;
      //prev[ii] = all_NULL_arr[ii];
      //prev[ii+1] = all_NULL_arr[ii+1];
  }

    queue[0] = items_left_after_initialization[i]; // intialized with the new element to be inserted
    unsigned qptr = 0; // head of the queue
    unsigned qsize = 1; 

    while (qsize > qptr) { // queue is not empty

        Aux_data_SPA* qitem = queue[qptr++];
        unsigned currEle = qitem->aux_array[0];
        
          unsigned eleSlot1 = (unsigned) get_index_SPA (currEle, A, kDict, k1, p, q, nslots);

        if(aux_ptrs[eleSlot1] == NULL) {
            prev[eleSlot1] = qitem;
            augment (eleSlot1, prev, A, kDict, MAX_KEYS, k1, k2, p, q);
            insert_success = true;
            break; // also set a flag to indicate that we need to go to the next element
        }

        unsigned eleSlot2 = (unsigned) get_index_SPA (currEle, A, kDict, k2, p, q, nslots);

        if(aux_ptrs[eleSlot2] == NULL) {
            prev[eleSlot2] = qitem;
            augment (eleSlot2, prev, A, kDict, MAX_KEYS, k1, k2, p, q);
            insert_success = true;
            break; // also set a flag to indicate that we need to go to the next element
        }

        if(visited[eleSlot1] == 0 /*false*/) {
            prev[eleSlot1] = qitem;
            visited[eleSlot1] = 1; // true;
            queue[qsize++] = aux_ptrs[eleSlot1];
        }

        else if(visited[eleSlot2] == 0 /*false*/) { // Note: if two positions of prev[] hold the same item.. then the same item gets added to two slots, which is not desirable!! Hence, the else if (..)
            prev[eleSlot2] = qitem;
            visited[eleSlot2] = 1; //true;
            queue[qsize++] = aux_ptrs[eleSlot2];
        }

    }
         if(!insert_success) { // try a different value of k2
             break;
         }

                visited.reset(); // reset all bits to 0
            }
        }

            if(insert_success) {

                //cout << k1 << " " << k2 << " worked!!" << endl;
                k[0] = k1;
                k[1] = k2;
                
                num_distinct_elements += 1; // the new element is successfully added
                return;
            }





        }
    }

        cerr << "None of the keys in kDict worked.. ABORTING!"  << endl;
        fflush(stderr);
        exit(1);
}

  void BucketSPA::augment (unsigned slotId, Aux_data_SPA** prev, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {

    //cout << "augment called" << endl;

    boost::dynamic_bitset< > marked(nslots); // bit vector of size 'nslots' bits; all 0's by default
      
      unsigned r = slotId;
      while (prev[r] != NULL && (marked[r] == 0)) 
      //while (true) 
      {
          // do pointer chasing
          marked[r] = 1; //true;
          Aux_data_SPA * item = prev[r];
          unsigned itemId = item->aux_array[0];

/*********************************************************************************/
                unsigned currSlot1 = (unsigned) get_index_SPA (itemId, A, kDict, k[0], p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index_SPA (itemId, A, kDict, k[1], p, q, nslots);

/*********************************************************************************/

    unsigned tmp; 
    if (currSlot1 == r)
        tmp = currSlot2;
    else 
        tmp = currSlot1;

    aux_ptrs[r] = item;
    r = tmp;

   // if (ele_ptr == item)
   //   break;
    }

  }

  void BucketSPA::augment (unsigned slotId, Aux_data_SPA** prev, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned k1, unsigned k2, unsigned p, unsigned q) {

    //cout << "augment called" << endl;
    boost::dynamic_bitset< > marked(nslots); // bit vector of size 'nslots' bits; all 0's by default
      unsigned r = slotId;
      while (prev[r] != NULL && (marked[r] == 0)) 
      //while(true)
      {
          // do pointer chasing
          marked[r] = 1; //true;
          Aux_data_SPA * item = prev[r];
          unsigned itemId = item->aux_array[0];

/*********************************************************************************/
                //unsigned long long indice1 = get_index_SPA (itemId, A, kDict, k1, p, q, nslots);
                //unsigned long long indice2 = get_index_SPA (itemId, A, kDict, k2, p, q, nslots);
                unsigned currSlot1 = (unsigned) get_index_SPA (itemId, A, kDict, k1, p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index_SPA (itemId, A, kDict, k2, p, q, nslots);
    //unsigned currSlot1 = (unsigned)indice1; // the slot to which the itemId is mapped
    //unsigned currSlot2 = (unsigned)indice2; // the slot to which the itemId is mapped

/*********************************************************************************/

    unsigned tmp; 
    if (currSlot1 == r)
        tmp = currSlot2;
    else 
        tmp = currSlot1;

    aux_ptrs[r] = item;
    r = tmp;

   // if (ele_ptr == item)
   //   break;
    }

  }


// This implments the augmentation starting from the element to be inserted

void BucketSPA::cuckooHash(unsigned eleId, TensorSPA& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {
    // it is invoked to add a new element.
    // so, after its invokation, the num_distinct_elements will be incremented by 1
    // currently, it is invoked only for inserting the third or further elements.
    // So, it first performs augmentation, and if the chosen keys do not work, then matching from scratch is the only option

//    assert (num_distinct_elements >= 2);


    Aux_data_SPA* curr_item_ptr = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
    curr_item_ptr->aux_array[0] = eleId;
    //curr_item_ptr->size = 1;
    //curr_item_ptr->insert(eleId); 


    // performing an elaborate augmentation

//#if 0
//    // Not required, since aux_ptrs[] will be updated only if we know that augmentation is successful
//    Aux_data_SPA ** aux_ptrs_copy = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * nslots);
//    // this copy is needed to restore the state of aux_ptrs in case the augmentation does not succeed!
//            //memcpy(aux_ptrs_copy, aux_ptrs, sizeof(Aux_data_SPA*) * nslots);
//    for(unsigned ii=0; ii < nslots; ii += 2) {
//        aux_ptrs_copy[ii] = aux_ptrs[ii];
//        aux_ptrs_copy[ii+1] = aux_ptrs[ii+1];
//    }
//#endif


    boost::dynamic_bitset< > visited(nslots); // bit vector of size 'nslots' bits; all 0's by default
    Aux_data_SPA ** prev = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * nslots);
    Aux_data_SPA ** queue = (Aux_data_SPA**) malloc(sizeof(Aux_data_SPA*) * nslots);
    for(unsigned i = 0; i < nslots; i += 2) {
        prev[i] = NULL;
        prev[i+1] = NULL;
    }

    queue[0] = curr_item_ptr; // intialized with the new element to be inserted
    unsigned qptr = 0; // head of the queue
    unsigned qsize = 1; 

    while (qsize > qptr) { // queue is not empty

        Aux_data_SPA* qitem = queue[qptr++];
        unsigned currEle = qitem->aux_array[0];
        
                //unsigned long long indice1 = get_index_SPA (currEle, A, kDict, k[0], p, q, nslots);
                unsigned currSlot1 = (unsigned) get_index_SPA (currEle, A, kDict, k[0], p, q, nslots);

        //unsigned currSlot1 = (unsigned)indice1; // the slot to which the eleId is mapped

        if(aux_ptrs[currSlot1] == NULL) {
            prev[currSlot1] = qitem;
            augment (currSlot1, prev, A, kDict, MAX_KEYS, p, q);
            num_distinct_elements += 1; // element is successfully added
            return;
        }

        //unsigned long long indice2 = get_index_SPA (currEle, A, kDict, k[1], p, q, nslots);
        unsigned currSlot2 = (unsigned) get_index_SPA (currEle, A, kDict, k[1], p, q, nslots);
        //unsigned currSlot2 = (unsigned)indice2; // the slot to which the eleId is mapped

        if(aux_ptrs[currSlot2] == NULL) {
            prev[currSlot2] = qitem;
            augment (currSlot2, prev, A, kDict, MAX_KEYS, p, q);
            num_distinct_elements += 1; // element is successfully added
            return;
        }

        if(visited[currSlot1] == 0 /*false*/) {
            prev[currSlot1] = qitem;
            visited[currSlot1] = 1; // true;
            queue[qsize++] = aux_ptrs[currSlot1];
        }

        else if(visited[currSlot2] == 0 /*false*/) { // Note: if two positions of prev[] hold the same item.. then the same item gets added to two slots, which is not desirable!! Hence, the else if (..)
            prev[currSlot2] = qitem;
            visited[currSlot2] = 1; //true;
            queue[qsize++] = aux_ptrs[currSlot2];
        }

    }
    


  // the item could not be succesfully added, so rehash

//#if 0
//    // Not required, since aux_ptrs[] will be updated only if we know that augmentation is successful
//  // restore the state of aux_ptrs
//        //Attention: A linear scan of nslots
//            //memcpy(aux_ptrs, aux_ptrs_copy, sizeof(Aux_data_SPA*) * nslots);
//  for(unsigned ii=0; ii < nslots; ii += 2) {
//      aux_ptrs[ii] = aux_ptrs_copy[ii];
//      aux_ptrs[ii+1] = aux_ptrs_copy[ii+1];
//  }
//#endif
              //bool change = manageBucketSPASize(); 
              rehash( eleId,  A,  kDict,  MAX_KEYS, p, q);
}






Hash_Table_SPA::Hash_Table_SPA(TensorSPA& A) {
    num_buckets = A.N;

    bucket_ptrs = (BucketSPA**) malloc (sizeof(BucketSPA*) * num_buckets);

    bucket_ptrs_bits.resize(num_buckets, 0);

    //unsigned N4 = num_buckets % 4;
    //    unsigned N4 = (3U) & num_buckets; //computing ((b - 1) & a), Here b = 4; a = num_buckets. It is equivalent to a % b where b is a power of 2

    //for(unsigned i=0; i < N4; ++i)
    //    bucket_ptrs[i] = NULL;

    //for(unsigned i=N4; i < num_buckets; i += 4) {
    //    bucket_ptrs[i] = NULL;
    //    bucket_ptrs[i+1] = NULL;
    //    bucket_ptrs[i+2] = NULL;
    //    bucket_ptrs[i+3] = NULL;
    //}


    size = 0;

    M_fastmod = fastmod::computeM_u32(num_buckets); 

    //MAX_KEYS = (64u > (unsigned) ceil(log(num_buckets)/log(2))) ? 64u : (unsigned) ceil(log(num_buckets)/log(2)); 
    MAX_KEYS = 32;  

    kDict = (unsigned long long*) malloc(sizeof(unsigned long long) * MAX_KEYS * A.d);

    unsigned larger = num_buckets;

    for (unsigned i = 0; i < A.d; i++)
    {
        if (A.dimensions_array[i] > larger)	
            larger = A.dimensions_array[i];		
    }

    p = 2147483647 ;/*this is 2^31-1*/
    q = 31;
    if( p < larger)
    {
        trouver_premierExtSPA(larger, p);
        q = 0;
        cout << "We fixed max num elements to 2^31-1"<<endl;
        exit(12);
    }

    srand(time(NULL));

    for(unsigned i = 0; i < MAX_KEYS; ++i) {
        for(unsigned dim = 0; dim < A.d; ++dim)
            kDict[i*A.d + dim] = randomnumberintExtSPA(p);
    }
}

unsigned Hash_Table_SPA::get_bucket_id (unsigned eleId, TensorSPA& A) {
    unsigned long long  somme= 0;
    unsigned num_key_dimensions = A.d;
    unsigned ele_memoized =  eleId*A.d;

    for (unsigned j = 0; j < num_key_dimensions; j++)
        somme += (1 + A.hedges_array[ele_memoized + j]) * kDict[j];

    unsigned long long indice;
    modpMersenneExtSPA(indice, somme, p, q);
    unsigned bucket_id = fastmod::fastmod_u32((unsigned) indice, M_fastmod, num_buckets); // this is the bucket id of the element 

    return bucket_id;

}

void Hash_Table_SPA::perform_second_level_hashing (unsigned eleId, TensorSPA& A, unsigned bucketId) {

    if(bucket_ptrs_bits[bucketId] == 0) { // the bucket is empty.. insert first element
    //if(bucket_ptrs[bucketId] == NULL) { // the bucket is empty.. insert first element
    	bucket_ptrs_bits[bucketId] = 1;  // mark the bucket as not empty
        bucket_ptrs[bucketId] = new BucketSPA(); 
        (bucket_ptrs[bucketId]->aux_ptrs)[0] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
    (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0] = eleId;
    //(bucket_ptrs[bucketId]->aux_ptrs)[0]->size = 1;
        //(bucket_ptrs[bucketId]->aux_ptrs)[0]->insert(eleId); 

        //bucket_ptrs[bucketId]->num_distinct_elements = 1;
        return;
    }

    else if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {

		// a second distinct element is to be added

            bucket_ptrs[bucketId]->nslots = 2; 
            Aux_data_SPA** temp = (Aux_data_SPA**)realloc(bucket_ptrs[bucketId]->aux_ptrs, 2 * sizeof(Aux_data_SPA*));
            bucket_ptrs[bucketId]->aux_ptrs = temp;

                        (bucket_ptrs[bucketId]->aux_ptrs)[1] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
                        (bucket_ptrs[bucketId]->aux_ptrs)[1]->aux_array[0] = eleId;

                        bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
                        return;
    }


    else if(bucket_ptrs[bucketId]->num_distinct_elements <= 3 ) { // either 2 or 3

       if (bucket_ptrs[bucketId]->num_distinct_elements == 2) {
            bucket_ptrs[bucketId]->nslots = 4; 
            Aux_data_SPA** temp = (Aux_data_SPA**)realloc(bucket_ptrs[bucketId]->aux_ptrs, 4 * sizeof(Aux_data_SPA*));
            bucket_ptrs[bucketId]->aux_ptrs = temp;

            bucket_ptrs[bucketId]->aux_ptrs[2] = NULL;
            bucket_ptrs[bucketId]->aux_ptrs[3] = NULL;
       }

                        (bucket_ptrs[bucketId]->aux_ptrs)[bucket_ptrs[bucketId]->num_distinct_elements] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
                        (bucket_ptrs[bucketId]->aux_ptrs)[bucket_ptrs[bucketId]->num_distinct_elements]->aux_array[0] = eleId;

                        bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
                        return;
	}

            else if (bucket_ptrs[bucketId]->num_distinct_elements == 4) {
// a fifth distinct element is found


             bucket_ptrs[bucketId]->manageBucketSPASize();
bucket_ptrs[bucketId]->k[0] = 0;
bucket_ptrs[bucketId]->k[0] = 1;
            bucket_ptrs[bucketId]->rehash( eleId,  A,  kDict,  MAX_KEYS, p, q);
            return;
     }

    else { // number of distinct elements is >= 5

        // cuckoo hash by finding augmenting paths in the matching

        // first check if the incoming element can be added to an existing slot

               unsigned ele_slot1  = (unsigned) get_index_SPA (eleId, A, kDict, bucket_ptrs[bucketId]->k[0], p, q, bucket_ptrs[bucketId]->nslots);
               unsigned ele_slot2  = (unsigned) get_index_SPA (eleId, A, kDict, bucket_ptrs[bucketId]->k[1], p, q, bucket_ptrs[bucketId]->nslots);


        if(ele_slot1 == ele_slot2) // since the two slots are the same for the new item, so we need to find a new set of keys. Therefore, rehash the bucket
        {
            //bool change = bucket_ptrs[bucketId]->manageBucketSPASize(); 
            bucket_ptrs[bucketId]->rehash( eleId,  A,  kDict,  MAX_KEYS, p, q );
            return;
        }

            // shortcutting elaborate augmentation through initialization 
        if ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] == NULL) {
        
        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
        //(bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->insert(eleId); 

                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->aux_array[0] = eleId;
                        //(bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->size = 1;

            bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
        return;
        }
            if  ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] == NULL) {
        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] = new Aux_data_SPA(); //the object is allocated on the heap and the pointer is returned. 
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->aux_array[0] = eleId;
                        //(bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->size = 1;
        //(bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->insert(eleId); 
            bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
        return;
            }
            
            bucket_ptrs[bucketId]->cuckooHash(eleId, A, kDict, MAX_KEYS, p, q);
    }


#ifdef SANITY_DEBUG
    //print the contents of the slots in a bucket
    for(unsigned yy = 0; yy < bucket_ptrs[bucketId]->nslots; ++yy) {
        if((bucket_ptrs[bucketId]->aux_ptrs)[yy] != NULL) {

            for(unsigned xx = 0; xx < (bucket_ptrs[bucketId]->aux_ptrs)[yy]->size; ++xx)
                cout << (bucket_ptrs[bucketId]->aux_ptrs)[yy]->aux_array[xx] << " ";
            cout << endl;
        }
    }
#endif

}

    bool Hash_Table_SPA::insert_element_at_bucket (unsigned eleId, unsigned bucketId, TensorSPA& A) // insert one element into the specified bucket id. // returns a flag value to indicate success or failure
{

        perform_second_level_hashing(eleId, A, bucketId);

    size++; // incrementing the number of elements in the bucket by 1
    return true; // insertion successful

}

bool Hash_Table_SPA::insert_element (unsigned eleId, TensorSPA& A) {
    // insert element with id eleId of tensor A into the hash table

    /*Step 1: Perform first level hashing -- determine the bucketId of eleId */

    unsigned bucketId = get_bucket_id(eleId, A);


    /*Step 2: Perform second level hashing */

        perform_second_level_hashing(eleId, A, bucketId);

    size++; // incrementing the number of elements in the bucket by 1
    return true; // insertion successful
}

bool Hash_Table_SPA::find_element_new (Ele_pos& ele_pos, TensorSPA& A, unsigned * ele_key_incides) {
    // find element with id "eleId" in the hash table
    // using only the key_dimensions, NOT the auxillary data.

    unsigned long long  somme= 0;
        unsigned num_key_dimensions = A.d;
        //unsigned N2 = num_key_dimensions % 2; // replace the computation with (find next multiple of 2 after num_key_dimensions and subtract 2 from it)
        unsigned N2 = (1U) & num_key_dimensions; //computing ((b - 1) & a), Here b = 2; a = num_key_dimensions. It is equivalent to a % b where b is a power of 2


    //for (unsigned j = 0; j < A.d; j++)
    //    somme += (1 + ele_key_incides[j]) * kDict[j];

        for (unsigned j = 0; j < N2; j++)
        somme += (1 + ele_key_incides[j]) * kDict[j];

        unsigned long long somme_0 = 0;
        unsigned long long somme_1 = 0;
        for (unsigned j = N2; j < num_key_dimensions; j += 2){
            somme_0 += (1 + ele_key_incides[j]) * kDict[j];
            somme_1 += (1 + ele_key_incides[j+1]) * kDict[j+1];
        }
            somme += somme_0 + somme_1;


    unsigned long long indice;
    modpMersenneExtSPA(indice, somme, p, q);
    unsigned bucketId = fastmod::fastmod_u32((unsigned) indice, M_fastmod, num_buckets); // this is the bucket id of the element 

    ele_pos.bid = bucketId;

    if (bucket_ptrs_bits[bucketId] == 0) {return false;}

    if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {
        unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0];
        //bool flag = isEqualSPA (ele_key_incides, myitem, A);

        if (isEqualSPA (ele_key_incides, myitem, A)) {
            // the eleId found
            ele_pos.sid = 0;
            return true;
        }

    }

    else if (bucket_ptrs[bucketId]->num_distinct_elements <= 4) {


       for(unsigned pqr = 0; pqr < bucket_ptrs[bucketId]->num_distinct_elements; ++pqr) {
        unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[pqr]->aux_array[0];

        if (isEqualSPA (ele_key_incides, myitem, A)) {
            // the eleId found
            ele_pos.sid = pqr;
            return true;
        }
	  
       }






    }


    else if (bucket_ptrs[bucketId]->num_distinct_elements > 4) {

        unsigned nslots_local = bucket_ptrs[bucketId]->nslots;

        unsigned long long indice1;
        unsigned long long somme1 = 0;
        unsigned k1 = bucket_ptrs[bucketId]->k[0];
        unsigned k1_memoized = k1 * num_key_dimensions;


        for (unsigned j = 0; j < N2; j++){
            somme1 += (1 + ele_key_incides[j]) * kDict[k1_memoized + j];
        }
        unsigned long long somme1_0 = 0;
        unsigned long long somme1_1 = 0;
        for (unsigned j = N2; j < num_key_dimensions; j += 2){
            somme1_0 += (1 + ele_key_incides[j]) * kDict[k1_memoized + j];
            somme1_1 += (1 + ele_key_incides[j+1]) * kDict[k1_memoized + j+1];
        }
            somme1 += somme1_0 + somme1_1;


        modpMersenneExtSPA(indice1, somme1, p, q);
        unsigned ele_slot1 = (unsigned)(indice1 % (nslots_local));

            Aux_data_SPA * ele_slot1_ptr = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1];
            if (ele_slot1_ptr != NULL) {
                unsigned myitem = ele_slot1_ptr->aux_array[0];
                //bool myflag = isEqualSPA (ele_key_incides, myitem, A);

                if (isEqualSPA (ele_key_incides, myitem, A)) {
                    ele_pos.sid = ele_slot1;
                    return true;
                }
            }

        unsigned k2 = bucket_ptrs[bucketId]->k[1];
        unsigned k2_memoized = k2 * num_key_dimensions;


        unsigned long long somme2 = 0;
        //for (unsigned j = 0; j < num_key_dimensions; j++){
        //    somme2 += (1 + ele_key_incides[j]) * kDict[k2_memoized + j];
        //}

        for (unsigned j = 0; j < N2; j++){
            somme2 += (1 + ele_key_incides[j]) * kDict[k2_memoized + j];
        }
        unsigned long long somme2_0 = 0;
        unsigned long long somme2_1 = 0;
        for (unsigned j = N2; j < num_key_dimensions; j += 2){
            somme2_0 += (1 + ele_key_incides[j]) * kDict[k2_memoized + j];
            somme2_1 += (1 + ele_key_incides[j+1]) * kDict[k2_memoized + j+1];
        }
            somme2 += somme2_0 + somme2_1;
        unsigned long long indice2;
        modpMersenneExtSPA(indice2, somme2, p, q);
         unsigned ele_slot2 = (unsigned) (indice2 % (nslots_local));

            Aux_data_SPA * ele_slot2_ptr = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2];
            if (ele_slot2_ptr != NULL) {

                unsigned myitem = ele_slot2_ptr->aux_array[0];

                if (isEqualSPA (ele_key_incides, myitem, A)) {
                    ele_pos.sid = ele_slot2;
                    return true;
                }

            }
    }

    return false; // the control flow will reach here only if the element is not found.
}

bool Hash_Table_SPA::find_element (unsigned eleId, Ele_pos& ele_pos, TensorSPA& A) {
    // find element with id "eleId" in the hash table
    // using only the key_dimensions, NOT the auxillary data.

    unsigned bucketId = get_bucket_id(eleId, A);

    ele_pos.bid = bucketId;

    if (bucket_ptrs_bits[bucketId] == 0) {return false;}

    if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {
            unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0];

        if (isEqualSPA(eleId, myitem, A)) {
            // the eleId found
            ele_pos.sid = 0;
            return true;
        }

    }

    else if (bucket_ptrs[bucketId]->num_distinct_elements <= 4) {


       for(unsigned pqr = 0; pqr < bucket_ptrs[bucketId]->num_distinct_elements; ++pqr) {
        unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[pqr]->aux_array[0];

        if (isEqualSPA (eleId, myitem, A)) {
            // the eleId found
            ele_pos.sid = pqr;
            return true;
        }
	  
       }






    }


    else if (bucket_ptrs[bucketId]->num_distinct_elements > 4) {

           unsigned ele_slot1 = (unsigned) get_index_SPA (eleId, A, kDict, bucket_ptrs[bucketId]->k[0], p, q, bucket_ptrs[bucketId]->nslots);



        if((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] != NULL) {
                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->aux_array[0];
                //bool myflag = isEqualSPA(eleId, myitem, A);

                if (isEqualSPA(eleId, myitem, A)) {
                    ele_pos.sid = ele_slot1;
                    return true;
                }
        
        }

                unsigned ele_slot2 = (unsigned) get_index_SPA (eleId, A, kDict, bucket_ptrs[bucketId]->k[1], p, q, bucket_ptrs[bucketId]->nslots);


            if (((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] != NULL)) {

                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->aux_array[0];
                bool myflag = isEqualSPA(eleId, myitem, A);

                if (myflag) {
                    ele_pos.sid = ele_slot2;
                    return true;
                }

            }
    }
    return false; // the control flow will reach here only if the element is not found.
}

#endif
