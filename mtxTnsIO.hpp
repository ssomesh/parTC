#ifndef MTXTNSIO_HPP
#define MTXTNSIO_HPP

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



using namespace std;

void readMatrixMarket(string ifname, vector <unsigned int> & dimensions, vector <vector<unsigned int>> &hedges, unsigned int &d, unsigned int &N);
void readDimensions(string nom_fichier, vector<unsigned int> &dimensions, unsigned int &d, unsigned int &N);
void readHedges(string nom_fichier, vector <vector<unsigned int>> &hedges, unsigned int d, unsigned int N);
void writeToBinFile(string outFileName,  vector <vector<unsigned int>> &hedges, vector<unsigned int> &dimensions, unsigned int d, unsigned int N);
void readFromBinFile(string fileName, vector <vector<unsigned int>> &hedges, vector<unsigned int> &dimensions, unsigned int &d, unsigned int &N);
void readMatrix(string fileName, unsigned** hedges_array_formal, unsigned **dimensions_array_formal, double **val_array_formal, unsigned &d, unsigned &N);
void readTensor(string fileName, unsigned** hedges_array_formal, unsigned** dimensions_array_formal, double** val_array_formal, unsigned &d, unsigned &N) ;


#endif
