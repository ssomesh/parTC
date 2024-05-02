#CXX = g++-12.2.0
#CC = gcc-12.2.0

CXX = g++
CC = gcc

main: main.cpp mtxTnsIO.cpp  fksCuckoo.hpp cs_multiply.o cs_util.o cs_malloc.o cs_scatter.o
	$(CXX) main.cpp mtxTnsIO.cpp cs_multiply.o cs_util.o cs_malloc.o cs_scatter.o -std=c++17 -O3 -fopenmp -g

cs_multiply.o: extern/cxsparse/cs_multiply.c
	$(CC) -c $< -o $@

cs_util.o: extern/cxsparse/cs_util.c
	$(CC) -c $< -o $@

cs_malloc.o: extern/cxsparse/cs_malloc.c
	$(CC) -c $< -o $@

cs_scatter.o: extern/cxsparse/cs_scatter.c
	$(CC) -c $< -o $@

clean:
	rm -f ./a.out *.o

# sample run
#OMP_NUM_THREADS=32 ./a.out lbnl-network.tns lbnl-network.meta 0 1
