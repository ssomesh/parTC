#include "mtxTnsIO.hpp"

#include "extern/pigo/pigo.hpp"

void readMatrix(string fileName, unsigned** hedges_array_formal, unsigned **dimensions_array_formal, double ** val_array_formal, unsigned &d, unsigned &N)
{
	pigo::COO<uint32_t, uint32_t, uint32_t*, false, false, false, true, double, double*> mmmtx {fileName}; 
	d = 2;
	N = mmmtx.m();
	unsigned int *dimensions_array = (unsigned int *) malloc (sizeof(unsigned) * 2); 
	unsigned * hedges_array = (unsigned int *) malloc (sizeof(unsigned) * N * 2); 
	double * val_array = (double *) malloc (sizeof(double) * N);  // stores the value of each non-zero

	dimensions_array[0] = mmmtx.nrows()-1;
	dimensions_array[1] = mmmtx.ncols()-1;

	for(unsigned nnz = 0; nnz < N; ++nnz) 
	{
		hedges_array[nnz*2] = mmmtx.x()[nnz] - 1;
		hedges_array[nnz*2+1] = mmmtx.y()[nnz] - 1;		
                
                val_array[nnz] = mmmtx.w()[nnz];
	}
	*hedges_array_formal = hedges_array;
	*dimensions_array_formal = dimensions_array;
	*val_array_formal = val_array;

	mmmtx.free();
}
void readTensor(string fileName, unsigned** hedges_array_formal, unsigned** dimensions_array_formal, double** val_array_formal, unsigned &d, unsigned &N) 
{
	pigo::Tensor<uint32_t, uint32_t, uint32_t*, double, double*, true> t { fileName };

	d = (unsigned) t.order();

	N = (unsigned) t.m();
	unsigned * hedges_array = (unsigned int *) malloc (sizeof(unsigned) * N * d); // rows = N; cols = d;
	unsigned * dimensions_array = (unsigned*) malloc (sizeof(unsigned) * d);
	double * val_array = (double *) malloc (sizeof(double) * N);  // stores the value of each non-zero

	for(unsigned nnz = 0; nnz < N; ++nnz) {
		for(unsigned int dim = 0; dim < d; ++dim) 
		{
			unsigned val = (unsigned)t.c()[nnz*d+dim]-1;
			hedges_array[nnz*d+dim] = val;
		}
                        
			val_array[nnz] = (double)t.w()[nnz];
                        //cout << val_array[nnz] << endl;
                   
	}

	vector<unsigned> dimensions = t.max_labels();

	for(unsigned dim = 0; dim < d; ++dim) {
		dimensions_array[dim] = dimensions[dim];
	}

	*hedges_array_formal = hedges_array;
	*dimensions_array_formal = dimensions_array;
	*val_array_formal = val_array;

	t.free();

}

void readMatrixMarket(string ifname, vector <unsigned int> & dimensions, vector <vector<unsigned int>> &hedges, unsigned int &d, unsigned int &N)
{


	string s, banner, mtx, crd, data_type, storage_scheme;
	istringstream iss;
	int diags;

	d=2;
	dimensions.resize(d);

//	s = ifname +".mtx";
//	ifstream mmf(s.c_str());

	ifstream mmf(ifname);
	if(mmf.fail())
	{
		cerr<< "File "<< ifname +".mtx"<< ": does not exist. "<<endl;
		mmf.close();
		exit(1);
	}
	getline(mmf, s);
	iss.str(s);

	if (!(iss >> banner>>mtx>>crd>>data_type>>storage_scheme))
	{
		cerr<< "File "<< ifname +".mtx"<< ": could not read the MM header. "<<endl;
		mmf.close();
		exit(1);
	}
	/*parse the first line*/
	if(banner.compare("%%MatrixMarket"))
	{
		cerr<< "File "<< ifname +".mtx"<< ": could not read the MM header. "<<endl;
		mmf.close();
		exit(1);
	}
	transform(mtx.begin(), mtx.end(), mtx.begin(), ::tolower);
	transform(crd.begin(), crd.end(), crd.begin(), ::tolower);
	transform(data_type.begin(), data_type.end(), data_type.begin(), ::tolower);
	transform(storage_scheme.begin(), storage_scheme.end(), storage_scheme.begin(), ::tolower);
	if(mtx.compare("matrix") || crd.compare("coordinate") || !data_type.compare("complex"))
	{
		cerr<< "File "<< ifname +".mtx"<< ": header not right. "<<endl;
		mmf.close();
		exit(1);
	}	

	if( !(storage_scheme.compare("general") || storage_scheme.compare("symmetric")))
	{
		cerr<< "File "<< ifname +".mtx"<< ": data type "<<  data_type << " is not supported."<<endl;
		mmf.close();
		exit(1);
	}
	/*advance until the m, n, nnz line*/
	do
	{
		getline(mmf, s, '\n');
	} while (s[0] == '%');

	iss.clear();
	iss.str(s);

	if( !(iss >> dimensions[0] >> dimensions[1] >>N))
	{
		cerr<< "File "<< ifname +".mtx"<< ": could not read m, n, nnz"<<endl;
		mmf.close();
		exit(1);
	}

	/*read nonzeros*/
	diags = 0;
	hedges.resize(N);
	for (unsigned int i = 0; i < N; i++)
	{
		getline(mmf, s, '\n');
		istringstream issnz (s);
		hedges[i].resize(2);
		if ( ! (issnz >> hedges[i][0] >> hedges[i][1]))
		{
			cerr<< "File "<< ifname +".mtx"<< ": could not read  nnz at: "<< i << endl;		
			mmf.close();
			exit(1);
		}
		if(hedges[i][0 ] == hedges[i][1])
			diags ++;
		hedges[i][0] -=1;
		hedges[i][1] -=1;
	}	
	mmf.close();

	if( !storage_scheme.compare("symmetric"))/*symmetric matrix: we need to repeat things*/
	{
		unsigned int initN = N, j = N;
		N = 2 * N - diags;
		hedges.resize(N);
		for(unsigned int i = 0; i < initN; i++)
		{
			if (hedges[i][0 ] != hedges[i][1])
			{
				hedges[j].resize(2);
				hedges[j][0 ] = hedges[i][1] ; 
				hedges[j][1 ] = hedges[i][0];
				j++;
			}
		}
	}
}

void readDimensions(string nom_fichier, vector<unsigned int> &dimensions, unsigned int &d, unsigned int &N)
{
	string s;

	s = nom_fichier+"-dim.txt";
	ifstream taille(s.c_str()); //On récupère les adresses des différents fichiers ainsi que leur dimension
	if(taille.fail())
	{	
		cerr<< "File "<< s.c_str()<< " does not exist. "<<endl;
		exit(1);
	}
	
	taille >> d;
	taille >> N;
	dimensions.resize(d);	
	for (unsigned int j = 0; j < d; j++)
		taille >> dimensions[j]; 

	taille.close();
}

void readHedges(string nom_fichier, vector <vector<unsigned int>> &hedges, unsigned int d, unsigned int N)
{
	/*
	 *
	 *   PRE: assumes 1-based numbering. 
	 *
	 */
	string s;

	s=nom_fichier+".tns";
  	ifstream fichier(s.c_str()); //On ouvre le i-ème fichier. 
  	if(fichier.fail())
  	{
  		cerr<< "Hedge file "<< s.c_str()<< " does not exist. "<<endl;
  		exit(2);
  	}

  	hedges.resize(N);
  	unsigned int j = 0;
  	while (j<N)
  	{
  		hedges[j].resize(d);
  		for (unsigned int k = 0; k < d; k++)
  		{
			fichier >> hedges[j][k]; //On lit chacun des d-uplets
			hedges[j][k] -= 1;			
		}
		
		getline(fichier,s);		
		j++;
		if(j != N && (fichier.eof()  || fichier.fail()))
		{
			cout << "could not read enough from file "<< nom_fichier<<endl;
			exit(12);
		}

	}
	fichier.close();
}



void writeToBinFile(string outFileName,  vector <vector<unsigned int>> &hedges, vector<unsigned int> &dimensions, unsigned int d, unsigned int N)
{
	unsigned long int tsz  = 2 + d + ((unsigned long)N)  * d;/*in the file: d N d1 d2 ..dd h1, h2,...,hN*/
	unsigned int *allCoordinates = new unsigned int [tsz];
	allCoordinates[0] = d;
	allCoordinates[1] = N;

	unsigned int i;
	for ( i=0; i < d; i++)	
		allCoordinates[2 + i] = dimensions[i];
	unsigned long int at;
	at = 2+ d;

	for(i = 0; i < N; i++)
	{
		for (unsigned int j = 0; j < d; j++)		
			allCoordinates[at++] = hedges[i][j];		
	}
	/*write into the binary file*/
	ofstream ofile;
	ofile.open(outFileName.c_str(), ios::out|ios::binary);
	ofile.write((char *) allCoordinates, tsz * sizeof(allCoordinates[0]));
	ofile.close();

	delete [] allCoordinates;
	allCoordinates = NULL;
}

void readFromBinFile(string fileName, vector <vector<unsigned int>> &hedges, vector<unsigned int> &dimensions, unsigned int &d, unsigned int &N)
{
	ifstream ifile;

	ifile.open(fileName.c_str(), ios::in | ios::binary);

	if(ifile.good())
	{
		ifile.seekg (0, ifile.beg);

		ifile.read((char *)&d, sizeof(d) );
		ifile.read((char *) &N, sizeof(N) );

		unsigned int *dvector = new unsigned int [d];
		dimensions.resize(d);

		ifile.read((char *) dvector, d * sizeof(dvector[0]));
		for (unsigned int i= 0; i < d; i++)
		{
			dimensions[i] = dvector[i];
		}

		unsigned int *allCoordinates = new unsigned int [d*(unsigned long)N];
		ifile.read((char*) allCoordinates, d * (unsigned long)N  * sizeof(allCoordinates[0]));
		if(ifile.gcount() != (long) (d * (long)N  * sizeof(allCoordinates[0])))
		{
			cout << "could not read enough data from "<< fileName<<endl;
			ifile.close();
			exit(12);
		}
		ifile.close();	

		hedges.resize(N);
		unsigned long int at = 0;
		for (unsigned int i = 0; i < N; i++)
		{
			hedges[i].resize(d);
			for(unsigned int j = 0; j < d; j++)		
			{	
				hedges[i][j] = allCoordinates[at++];			
				if(hedges[i][j]>= dimensions[j])
				{
					cout <<"indices are not ok. Read "<< hedges[i][j]<<" but dim "<<dimensions[j]<<endl;
					exit(12);
				}
			}
		}
		delete [] dvector;
		delete [] allCoordinates;
	}
	else//(ifile.fail())
	{
		cout <<"Could not open file "<< fileName<< " to read binary"<<endl;
		exit(12);
	}

	
}

