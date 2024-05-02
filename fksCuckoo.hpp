
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
//#define MAXSLOTS 1024


/* computes y = x mod pp*/
#define modpMersenneExt(y, x, pp, qq)  \
{                                  \
    y = ((x) & (pp)) + ((x)>>(qq));\
    y = y >= (pp) ? y-= (pp) : y;  \
}



//utility functions

inline bool isEqual(unsigned ele1, unsigned ele2, Tensor& A) {

        unsigned n = A.num_key_dimensions;
        unsigned ele1_memoized = ele1*A.d;
        unsigned ele2_memoized = ele2*A.d;

        for (unsigned i = 0; i < n; i++){
            if( A.hedges_array[ele1_memoized + A.dim_tc[i]] != A.hedges_array[ele2_memoized + A.dim_tc[i]] ) {
                return false;
            }
        }

        return true;


}

// overloaded function
inline bool isEqual(unsigned* arr, unsigned ele, Tensor& A) { 

        unsigned n = A.num_key_dimensions;
        unsigned ele_memoized = ele*A.d;

        for (unsigned i = 0; i < n; i++){
            if( arr[i] != A.hedges_array[ele_memoized + A.dim_tc[i]] ) {
                return false;
            }
        }

        return true;
}




inline unsigned long long get_index (unsigned ele, Tensor& A, unsigned long long * kDict, unsigned keyid, unsigned p, unsigned q, unsigned n) {

    // this is used for obtaining a slot in the second level hashing

    unsigned ele_memoized = ele * A.d;
    unsigned num_key_dimensions = A.num_key_dimensions;
    unsigned key_memoized = keyid * num_key_dimensions;
    //unsigned N2 = num_key_dimensions % 2;
        unsigned N2 = (1U) & num_key_dimensions; //computing ((b - 1) & a), Here b = 2; a = num_key_dimensions. It is equivalent to a % b where b is a  power of 2
    unsigned long long  somme = 0;
        for (unsigned j = 0; j < N2; j++){
            somme += (1 + A.hedges_array[ele_memoized + A.dim_tc[j]]) * kDict[key_memoized + j];
        }

        unsigned long long somme_0 = 0;
        unsigned long long somme_1 = 0;
        for (unsigned j = N2; j < num_key_dimensions; j += 2){
            somme_0 += (1 + A.hedges_array[ele_memoized + A.dim_tc[j]]) * kDict[key_memoized + j];
            somme_1 += (1 + A.hedges_array[ele_memoized + A.dim_tc[j+1]]) * kDict[key_memoized + j + 1];
        }
            somme += somme_0 + somme_1;

    unsigned long long indice;
        modpMersenneExt(indice, somme, p, q);
        //indice = indice % n; // this is correct since there are nslots many neighbors of num_distinct_elements many items

        return (indice % n);

}

void findMersenneExt(unsigned N, unsigned &p, unsigned &q)
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

unsigned randomnumberintExt(unsigned p) {/*from https://stackoverflow.com/questions/56435506/rng-function-c*/
    // Making rng static ensures that it stays the same
    // Between different invocations of the function
    static std::mt19937 rng;

    std::uniform_int_distribution<uint32_t> dist(0, p-1); 
    return dist(rng); 
}

bool est_premierExt(unsigned i)
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

void trouver_premierExt(const unsigned m, unsigned&p)
{
    unsigned i = m+1;
    while (!est_premierExt(i))
        i+=1;
    p=i;
}

struct Ele_pos {
    unsigned bid; // bucket id
    unsigned sid; // slot id within the bucket
};


struct Aux_data {
    // store the array of integers, which are the ids in the COO data structure
    unsigned * aux_array;
    // Growable buffer for ids

    // Capacity of aux_array
    unsigned capacity; 

    // Number of entries actually in aux_array
    unsigned size; 

    void insert(unsigned);

    Aux_data();
};
    

struct Bucket {
    unsigned num_distinct_elements; // keeps a count of the number of distinct "contraction indices values" in the bucket. The contraction indices are the indices that are used for hashing throughout.
    unsigned nslots; // number of slots for cuckoo hashing
    unsigned k[2]; // keys for cuckoo hash within a bucket.. (stores the ids of kDict)
    Aux_data ** aux_ptrs;  // this array of pointers forms a bucket. Each pointer in this array points to an Aux_data struct.

    //note: even if there is a single unique element, create an aux_array for it

    Bucket();

    void manageBucketSize();

    void cuckooHash(unsigned eleId, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned); // perform cuckoo hash.. it will potentially update k[] and aux_ptrs

  void augment (unsigned currSlot, Aux_data** prev, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned);

  void augment (unsigned slotId, Aux_data** prev, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned k1, unsigned k2, unsigned p, unsigned q);
    void rehash(unsigned eleId, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned, unsigned);


};

struct Hash_Table {  // the entire hash structure for a tensor

    unsigned size; // number of elements in the hash table
    unsigned p, q;
    unsigned MAX_KEYS;
    unsigned num_buckets; // user supplied. usually we inialize it to N
    uint64_t M_fastmod; // used for computing the mod efficiently for first level hashing

    Bucket** bucket_ptrs; // this is an array of pointers to buckets. the size of the array should be num_buckets

    // Note: have an array of pointers to buckets. Then, the hash_table will have an array of 'num_buckets' many pointers. We'll allocate space for these only when we need to. Otherwise, they will be pointing to NULL for empty buckets. 

    unsigned long long * kDict; // stores the actual l-tuple; where l is the number of dimensions to be used as key


    Hash_Table(Tensor&);  // constructor. it initializes the num_buckets and allocates space for the buckets. 

    bool insert_element (unsigned, Tensor&); // insert one element into the bucket. // returns a flag value to indicate success or failure
    bool insert_element_at_bucket (unsigned, unsigned, Tensor&); // insert one element into the specified bucket id. // returns a flag value to indicate success or failure
    bool find_element(unsigned, Ele_pos&, Tensor&); // test if an element is present in the hash table. It returns true if the search is successful, false otherwise. 

    bool find_element_new(Ele_pos&, Tensor&, unsigned * ele_key_incides); // test if an element is present in the hash table. It returns true if the search is successful, false otherwise. 
    

    unsigned get_bucket_id(unsigned, Tensor&);
    void perform_second_level_hashing(unsigned, Tensor&, unsigned); // possibly return a bool for status
    //void rehash (); // perform a rehash of all the elements present in the hash table so far.
};

Aux_data::Aux_data() {
    capacity = 1; // start at capacity = 1 then double
    size = 0;

    aux_array = (unsigned*) malloc (sizeof(unsigned*) * capacity);
}

void Aux_data::insert(unsigned eleId) {


    // grow buffer if necessary
    if (size + 1 > capacity)
    {
        //if (capacity <= (UINT_MAX / 2))
            capacity += capacity;

        // extend buffer's capacity
        unsigned* temp = (unsigned*)realloc(aux_array, capacity * sizeof(unsigned));
        if (temp == NULL)
        {
            cerr << "Out of memory! Aborting!!" << endl;
            exit(1);
        }
        aux_array = temp;
    }

    // append current element to buffer
    aux_array[size++] = eleId;

}


Bucket::Bucket() {
    num_distinct_elements = 1;
    nslots = 1; // 2^0
    //k[0] = -1;
    //k[1] = -1;
    aux_ptrs =  (Aux_data**) malloc(sizeof(Aux_data*)); // allocate space for a single element
}


void Bucket::manageBucketSize() {

    // grow buffer if necessary

    if ((num_distinct_elements + num_distinct_elements + 2) > nslots) // 2*(num_distinct_elements+1)... the num_distinct_elements here are before incrementing the value
    {

        unsigned old_nslots = nslots;

            nslots = nslots << 1;

        // extend buffer's capacity
        Aux_data** temp = (Aux_data**)realloc(aux_ptrs, nslots * sizeof(Aux_data*));
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

void Bucket::rehash(unsigned eleId, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {

    Aux_data ** aux_ptrs_copy = (Aux_data**) malloc(sizeof(Aux_data*) * (num_distinct_elements + 1)); // a densely packed array containing all elements
    Aux_data ** items_left_after_initialization = (Aux_data**) malloc(sizeof(Aux_data*) * (num_distinct_elements + 1));

    for(unsigned ii=0, counter = 0; ii < nslots; ++ii) { // the nslots being used here is after potentially being resized.. it may be better to make the copy before resizing the bucket
        if(aux_ptrs[ii] == NULL) continue;
        aux_ptrs_copy[counter++] = aux_ptrs[ii];
    }

    aux_ptrs_copy[num_distinct_elements] = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 
    aux_ptrs_copy[num_distinct_elements]->aux_array[0] = eleId;
    aux_ptrs_copy[num_distinct_elements]->size = 1;


    manageBucketSize(); // resizing the bucket, if required

    for(unsigned k1 = k[0]; k1 < MAX_KEYS; ++k1 ) {
        for (unsigned k2 = (k1 == k[0]) ? (k[1]+1) : 0; k2 < MAX_KEYS; ++k2) {
            if (k1 == k2) continue;

            // reinitialize aux_ptrs[] to all NULL
            //memcpy(aux_ptrs, all_NULL_arr, sizeof(Aux_data*) * nslots);
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
                Aux_data* curr_item_ptr = aux_ptrs_copy[i]; // the element to insert
                unsigned currEle = curr_item_ptr->aux_array[0];

                unsigned currSlot1 = (unsigned) get_index (currEle, A, kDict, k1, p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index (currEle, A, kDict, k2, p, q, nslots);

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
    Aux_data ** prev = (Aux_data**) malloc(sizeof(Aux_data*) * nslots);
    Aux_data ** queue = (Aux_data**) malloc(sizeof(Aux_data*) * nslots);

    boost::dynamic_bitset< > visited(nslots); // bit vector of size 'nslots' bits; all 0's by default
            for(unsigned i=0; i < items_left_after_initialization_size; ++i) {
                insert_success = false;

  //          memcpy(prev, all_NULL_arr, sizeof(Aux_data*) * nslots);
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

        Aux_data* qitem = queue[qptr++];
        unsigned currEle = qitem->aux_array[0];
        
          unsigned eleSlot1 = (unsigned) get_index (currEle, A, kDict, k1, p, q, nslots);

        if(aux_ptrs[eleSlot1] == NULL) {
            prev[eleSlot1] = qitem;
            augment (eleSlot1, prev, A, kDict, MAX_KEYS, k1, k2, p, q);
            insert_success = true;
            break; // also set a flag to indicate that we need to go to the next element
        }

        unsigned eleSlot2 = (unsigned) get_index (currEle, A, kDict, k2, p, q, nslots);

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

  void Bucket::augment (unsigned slotId, Aux_data** prev, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {

    boost::dynamic_bitset< > marked(nslots); // bit vector of size 'nslots' bits; all 0's by default
      
      unsigned r = slotId;
      while (prev[r] != NULL && (marked[r] == 0)) 
      {
          // do pointer chasing
          marked[r] = 1; //true;
          Aux_data * item = prev[r];
          unsigned itemId = item->aux_array[0];

                unsigned currSlot1 = (unsigned) get_index (itemId, A, kDict, k[0], p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index (itemId, A, kDict, k[1], p, q, nslots);


    unsigned tmp; 
    if (currSlot1 == r)
        tmp = currSlot2;
    else 
        tmp = currSlot1;

    aux_ptrs[r] = item;
    r = tmp;

    }

  }

  void Bucket::augment (unsigned slotId, Aux_data** prev, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned k1, unsigned k2, unsigned p, unsigned q) {

    boost::dynamic_bitset< > marked(nslots); // bit vector of size 'nslots' bits; all 0's by default
      unsigned r = slotId;
      while (prev[r] != NULL && (marked[r] == 0)) 
      {
          // do pointer chasing
          marked[r] = 1; //true;
          Aux_data * item = prev[r];
          unsigned itemId = item->aux_array[0];

                unsigned currSlot1 = (unsigned) get_index (itemId, A, kDict, k1, p, q, nslots);
                unsigned currSlot2 = (unsigned) get_index (itemId, A, kDict, k2, p, q, nslots);


    unsigned tmp; 
    if (currSlot1 == r)
        tmp = currSlot2;
    else 
        tmp = currSlot1;

    aux_ptrs[r] = item;
    r = tmp;

    }

  }


// This implments the augmentation starting from the element to be inserted

void Bucket::cuckooHash(unsigned eleId, Tensor& A, unsigned long long * kDict, unsigned MAX_KEYS, unsigned p, unsigned q) {
    // it is invoked to add a new element.
    // so, after its invokation, the num_distinct_elements will be incremented by 1
    // currently, it is invoked only for inserting the third or further elements.
    // So, it first performs augmentation, and if the chosen keys do not work, then matching from scratch is the only option

//    assert (num_distinct_elements >= 2);


    Aux_data* curr_item_ptr = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 
    curr_item_ptr->aux_array[0] = eleId;
    curr_item_ptr->size = 1;
    //curr_item_ptr->insert(eleId); 


    // performing an elaborate augmentation

    Aux_data ** aux_ptrs_copy = (Aux_data**) malloc(sizeof(Aux_data*) * nslots);
    // this copy is needed to restore the state of aux_ptrs in case the augmentation does not succeed!
            //memcpy(aux_ptrs_copy, aux_ptrs, sizeof(Aux_data*) * nslots);
    for(unsigned ii=0; ii < nslots; ii += 2) {
        aux_ptrs_copy[ii] = aux_ptrs[ii];
        aux_ptrs_copy[ii+1] = aux_ptrs[ii+1];
    }


    boost::dynamic_bitset< > visited(nslots); // bit vector of size 'nslots' bits; all 0's by default
    Aux_data ** prev = (Aux_data**) malloc(sizeof(Aux_data*) * nslots);
    Aux_data ** queue = (Aux_data**) malloc(sizeof(Aux_data*) * nslots);
    for(unsigned i = 0; i < nslots; i += 2) {
        prev[i] = NULL;
        prev[i+1] = NULL;
    }

    queue[0] = curr_item_ptr; // intialized with the new element to be inserted
    unsigned qptr = 0; // head of the queue
    unsigned qsize = 1; 

    while (qsize > qptr) { // queue is not empty

        Aux_data* qitem = queue[qptr++];
        unsigned currEle = qitem->aux_array[0];
        
                //unsigned long long indice1 = get_index (currEle, A, kDict, k[0], p, q, nslots);
                unsigned currSlot1 = (unsigned) get_index (currEle, A, kDict, k[0], p, q, nslots);

        //unsigned currSlot1 = (unsigned)indice1; // the slot to which the eleId is mapped

        if(aux_ptrs[currSlot1] == NULL) {
            prev[currSlot1] = qitem;
            augment (currSlot1, prev, A, kDict, MAX_KEYS, p, q);
            num_distinct_elements += 1; // element is successfully added
            return;
        }

        //unsigned long long indice2 = get_index (currEle, A, kDict, k[1], p, q, nslots);
        unsigned currSlot2 = (unsigned) get_index (currEle, A, kDict, k[1], p, q, nslots);
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

  // restore the state of aux_ptrs
        //Attention: A linear scan of nslots
            //memcpy(aux_ptrs, aux_ptrs_copy, sizeof(Aux_data*) * nslots);
  for(unsigned ii=0; ii < nslots; ii += 2) {
      aux_ptrs[ii] = aux_ptrs_copy[ii];
      aux_ptrs[ii+1] = aux_ptrs_copy[ii+1];
  }
              //bool change = manageBucketSize(); 
              rehash( eleId,  A,  kDict,  MAX_KEYS, p, q);
}






Hash_Table::Hash_Table(Tensor& A) {
    num_buckets = A.N;

    bucket_ptrs = (Bucket**) malloc (sizeof(Bucket*) * num_buckets);

    //unsigned N4 = num_buckets % 4;
        unsigned N4 = (3U) & num_buckets; //computing ((b - 1) & a), Here b = 4; a = num_buckets. It is equivalent to a % b where b is a power of 2

    for(unsigned i=0; i < N4; ++i)
        bucket_ptrs[i] = NULL;

    for(unsigned i=N4; i < num_buckets; i += 4) {
        bucket_ptrs[i] = NULL;
        bucket_ptrs[i+1] = NULL;
        bucket_ptrs[i+2] = NULL;
        bucket_ptrs[i+3] = NULL;
    }


    size = 0;

    M_fastmod = fastmod::computeM_u32(num_buckets); 

    //MAX_KEYS = (64u > (unsigned) ceil(log(num_buckets)/log(2))) ? 64u : (unsigned) ceil(log(num_buckets)/log(2)); 
    MAX_KEYS = 32;  

    kDict = (unsigned long long*) malloc(sizeof(unsigned long long) * MAX_KEYS * A.num_key_dimensions);

    unsigned larger = num_buckets;

    for (unsigned i = 0; i < A.num_key_dimensions; i++)
    {
        if (A.dimensions_array[A.dim_tc[i]] > larger)	
            larger = A.dimensions_array[A.dim_tc[i]];		
    }

    p = 2147483647 ;/*this is 2^31-1*/
    q = 31;
    if( p < larger)
    {
        trouver_premierExt(larger, p);
        q = 0;
        cout << "We fixed max num elements to 2^31-1"<<endl;
        exit(12);
    }

    srand(time(NULL));

    for(unsigned i = 0; i < MAX_KEYS; ++i) {
        for(unsigned dim = 0; dim < A.num_key_dimensions; ++dim)
            kDict[i*A.num_key_dimensions + dim] = randomnumberintExt(p);
    }
}

unsigned Hash_Table::get_bucket_id (unsigned eleId, Tensor& A) {
    unsigned long long  somme= 0;
    unsigned num_key_dimensions = A.num_key_dimensions;
    unsigned ele_memoized =  eleId*A.d;

    for (unsigned j = 0; j < num_key_dimensions; j++)
        somme += (1 + A.hedges_array[ele_memoized + A.dim_tc[j]]) * kDict[j];

    unsigned long long indice;
    modpMersenneExt(indice, somme, p, q);
    unsigned bucket_id = fastmod::fastmod_u32((unsigned) indice, M_fastmod, num_buckets); // this is the bucket id of the element 

    return bucket_id;

}

void Hash_Table::perform_second_level_hashing (unsigned eleId, Tensor& A, unsigned bucketId) {

    if(bucket_ptrs[bucketId] == NULL) { // the bucket is empty.. insert first element
        bucket_ptrs[bucketId] = new Bucket(); 
        (bucket_ptrs[bucketId]->aux_ptrs)[0] = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 
    (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0] = eleId;
    (bucket_ptrs[bucketId]->aux_ptrs)[0]->size = 1;
        //(bucket_ptrs[bucketId]->aux_ptrs)[0]->insert(eleId); 

        //bucket_ptrs[bucketId]->num_distinct_elements = 1;
        return;
    }

    else if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {
        unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0];

        if (isEqual(eleId, myitem, A)) {
            // the eleId has to be added to the auxillary data 
            (bucket_ptrs[bucketId]->aux_ptrs)[0]->insert(eleId); 
            return;
        }
        else {

            // a second distinct element found! Need to insert the new element by performing cuckoo hashing

            // resize the bucket
            bucket_ptrs[bucketId]->nslots = 8; // nslots is 2*(num items in bucket after the insertion of the new item)
            //Aux_data** temp = (Aux_data**)realloc(bucket_ptrs[bucketId]->aux_ptrs, bucket_ptrs[bucketId]->nslots * sizeof(Aux_data*));
            Aux_data** temp = (Aux_data**)malloc(bucket_ptrs[bucketId]->nslots * sizeof(Aux_data*));
            temp[0] = (bucket_ptrs[bucketId]->aux_ptrs)[0];
            temp[1] = NULL;
            temp[2] = NULL;
            temp[3] = NULL;
            temp[4] = NULL;
            temp[5] = NULL;
            temp[6] = NULL;
            temp[7] = NULL;


            bucket_ptrs[bucketId]->aux_ptrs = temp;

            unsigned ele1_id = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0]; // the element that is already in the bucket

            //bool insufficient_keys = true;
            for(unsigned k1 = 0; k1 < MAX_KEYS; ++k1) {
                for(unsigned k2 = 0; k2 < MAX_KEYS; ++k2) { 
                    if(k1 == k2) continue;

         unsigned ele1_slot1 = (unsigned)get_index (ele1_id, A, kDict, k1, p, q, bucket_ptrs[bucketId]->nslots);
         unsigned ele1_slot2 = (unsigned)get_index (ele1_id, A, kDict, k2, p, q, bucket_ptrs[bucketId]->nslots);

                     if(ele1_slot1 == ele1_slot2) { // the element maps to the same slot with both the keys, so try a different pair of keys.
                         continue;
                     }
        
                     
         unsigned ele_slot1 = (unsigned) get_index (eleId, A, kDict, k1, p, q, bucket_ptrs[bucketId]->nslots);
         unsigned ele_slot2 = (unsigned) get_index (eleId, A, kDict, k2, p, q, bucket_ptrs[bucketId]->nslots);

                     if(ele_slot1 == ele_slot2) {
                         continue;
                     }

                     // check if both the slots of the two items are the same
                     if ((ele1_slot1 == ele_slot1 && ele1_slot2 == ele_slot2) || (ele1_slot1 == ele_slot2 && ele1_slot2 == ele_slot1))
                         continue;


                     // always inserting the existing element at its first (new) possible slot.
                     // before hashing, the existing element was at slot 0.
                     if(ele1_slot1 > 0) {
                            (bucket_ptrs[bucketId]->aux_ptrs)[ele1_slot1] = (bucket_ptrs[bucketId]->aux_ptrs)[0]; // moving the existing element in the bucket to the new slot
                            (bucket_ptrs[bucketId]->aux_ptrs)[0] = NULL;
                     }

                     // finding the slot for the new incoming element
                            unsigned ele_slot = ele_slot1;
                            if (ele1_slot1 == ele_slot1)
                                ele_slot = ele_slot2;



                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot] = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot]->aux_array[0] = eleId;
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot]->size = 1;


                        bucket_ptrs[bucketId]->k[0] = k1;
                        bucket_ptrs[bucketId]->k[1] = k2;

                        bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
                        return;

                }
            } 

                cerr << "None of the keys in kDict worked.. ABORTING!"  << endl;
                fflush(stderr);
                exit(1);


        }

    }

    else { // number of distinct elements is >= 2


               unsigned ele_slot1  = (unsigned) get_index (eleId, A, kDict, bucket_ptrs[bucketId]->k[0], p, q, bucket_ptrs[bucketId]->nslots);
               unsigned ele_slot2  = (unsigned) get_index (eleId, A, kDict, bucket_ptrs[bucketId]->k[1], p, q, bucket_ptrs[bucketId]->nslots);




        if(ele_slot1 == ele_slot2) // since the two slots are the same for the new item, so we need to find a new set of keys. Therefore, rehash the bucket
        {
            bucket_ptrs[bucketId]->rehash( eleId,  A,  kDict,  MAX_KEYS, p, q );
            return;
        }


            if ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] != NULL) {
                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->aux_array[0];

                if (isEqual(eleId, myitem, A)) {
                    (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->insert(eleId); 
                    return;
                }
            }


            if ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] != NULL) {

                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->aux_array[0];

                if (isEqual(eleId, myitem, A)) {
                    (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->insert(eleId); 
                    return;
                }
            }


            // shortcutting elaborate augmentation through initialization 
        if ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] == NULL) {
        
        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 

                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->aux_array[0] = eleId;
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->size = 1;

            bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
        return;
        }
            if  ((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] == NULL) {
        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] = new Aux_data(); //the object is allocated on the heap and the pointer is returned. 
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->aux_array[0] = eleId;
                        (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->size = 1;
            bucket_ptrs[bucketId]->num_distinct_elements += 1; // the element is successfully added
        return;
            }
            
            bucket_ptrs[bucketId]->cuckooHash(eleId, A, kDict, MAX_KEYS, p, q);
    }


}

    bool Hash_Table::insert_element_at_bucket (unsigned eleId, unsigned bucketId, Tensor& A) // insert one element into the specified bucket id. // returns a flag value to indicate success or failure
{

        perform_second_level_hashing(eleId, A, bucketId);

    size++; // incrementing the number of elements in the bucket by 1
    return true; // insertion successful

}

bool Hash_Table::insert_element (unsigned eleId, Tensor& A) {
    // insert element with id eleId of tensor A into the hash table

    /*Step 1: Perform first level hashing -- determine the bucketId of eleId */

    unsigned bucketId = get_bucket_id(eleId, A);


    /*Step 2: Perform second level hashing */

        perform_second_level_hashing(eleId, A, bucketId);

    size++; // incrementing the number of elements in the bucket by 1
    return true; // insertion successful
}

bool Hash_Table::find_element_new (Ele_pos& ele_pos, Tensor& A, unsigned * ele_key_incides) {

    unsigned long long  somme= 0;
        unsigned num_key_dimensions = A.num_key_dimensions;
        unsigned N2 = (1U) & num_key_dimensions; //computing ((b - 1) & a), Here b = 2; a = num_key_dimensions. It is equivalent to a % b where b is a power of 2



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
    modpMersenneExt(indice, somme, p, q);
    unsigned bucketId = fastmod::fastmod_u32((unsigned) indice, M_fastmod, num_buckets); // this is the bucket id of the element 

    ele_pos.bid = bucketId;

    if (bucket_ptrs[bucketId] == NULL) {return false;}

    if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {
        unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0];
        //bool flag = isEqual (ele_key_incides, myitem, A);

        if (isEqual (ele_key_incides, myitem, A)) {
            // the eleId found
            ele_pos.sid = 0;
            return true;
        }

    }


    else if (bucket_ptrs[bucketId]->num_distinct_elements > 1) {

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


        modpMersenneExt(indice1, somme1, p, q);
        unsigned ele_slot1 = (unsigned)(indice1 % (nslots_local));

            Aux_data * ele_slot1_ptr = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1];
            if (ele_slot1_ptr != NULL) {
                unsigned myitem = ele_slot1_ptr->aux_array[0];

                if (isEqual (ele_key_incides, myitem, A)) {
                    ele_pos.sid = ele_slot1;
                    return true;
                }
            }

        unsigned k2 = bucket_ptrs[bucketId]->k[1];
        unsigned k2_memoized = k2 * num_key_dimensions;


        unsigned long long somme2 = 0;

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
        modpMersenneExt(indice2, somme2, p, q);
         unsigned ele_slot2 = (unsigned) (indice2 % (nslots_local));

            Aux_data * ele_slot2_ptr = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2];
            if (ele_slot2_ptr != NULL) {

                unsigned myitem = ele_slot2_ptr->aux_array[0];

                if (isEqual (ele_key_incides, myitem, A)) {
                    ele_pos.sid = ele_slot2;
                    return true;
                }

            }
        //}
    }

    return false; // the control flow will reach here only if the element is not found.
}

bool Hash_Table::find_element (unsigned eleId, Ele_pos& ele_pos, Tensor& A) {
    // find element with id "eleId" in the hash table
    // using only the key_dimensions, NOT the auxillary data.

    unsigned bucketId = get_bucket_id(eleId, A);

    ele_pos.bid = bucketId;

    if (bucket_ptrs[bucketId] == NULL) 
        return false;

    if(bucket_ptrs[bucketId]->num_distinct_elements == 1) {
            unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[0]->aux_array[0];

        if (isEqual(eleId, myitem, A)) {
            // the eleId found
            ele_pos.sid = 0;
            return true;
        }

    }


    else if (bucket_ptrs[bucketId]->num_distinct_elements > 1) {

           unsigned ele_slot1 = (unsigned) get_index (eleId, A, kDict, bucket_ptrs[bucketId]->k[0], p, q, bucket_ptrs[bucketId]->nslots);



        if((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1] != NULL) {
                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot1]->aux_array[0];

                if (isEqual(eleId, myitem, A)) {
                    ele_pos.sid = ele_slot1;
                    return true;
                }
        
        }

                unsigned ele_slot2 = (unsigned) get_index (eleId, A, kDict, bucket_ptrs[bucketId]->k[1], p, q, bucket_ptrs[bucketId]->nslots);


            if (((bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2] != NULL)) {

                unsigned myitem = (bucket_ptrs[bucketId]->aux_ptrs)[ele_slot2]->aux_array[0];
                bool myflag = isEqual(eleId, myitem, A);

                if (myflag) {
                    ele_pos.sid = ele_slot2;
                    return true;
                }

            }


    }





    return false; // the control flow will reach here only if the element is not found.
}

Hash_Table* buildHashTable(Tensor& A) { // possibly the function can return a Hash_Table 
    Hash_Table* H = new Hash_Table(A);


    uint64_t elapsed;
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    for(unsigned i=0; i < A.N; ++i) {
        H->insert_element (i, A);
    }
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();

    cout << "Number of elements inserted = " << H->size << endl;
    cout << "Total time for insertion = " << (double)(elapsed * 1.E-9 ) << " (s)" << endl;

   return H;
}

