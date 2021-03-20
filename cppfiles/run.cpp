// Author: Gustav Lang Moesmand
// Small note: 
//      Notes are mostly thought for my own understanding, so sorry in advance,
//      if you are reading this

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <Gel/CGLA/CGLA.h>
#include <GEL/Geometry/Graph.h>
#include <GEL/Hmesh/Hmesh.h>

#include "graph_io.h"
#include "graph_skeletonize.h"
#include "graph_util.h"

#include <iostream>
#include <vector>
 
using namespace std;
using namespace HMesh;
using namespace Geometry;
using namespace CGLA;

// Field
const int __img_size = 785;
const int __num_of_imgs = 10;

void split(const string& str, vector<int>& vec, char delim = ',')
{
    stringstream ss(str);
    string token;

    int i = 0;

    while (std::getline(ss, token, delim)) {
        vec.at(i++) =  std::stoi(token);
    }
}

void print_img(vector<int> vec) 
{
    for(int i=0; i < vec.size(); i++)
        cout << vec.at(i) << " ";
}

// Function that reads a number of mnist numbers from path file given in function
// and inserts them in given vec<vec<int>>.

void read_mnist(int NumberOfImages, int DataOfAnImage, vector<vector<int>> &arr)
{
    arr.resize(NumberOfImages, vector<int>(DataOfAnImage));
    const string path = "E:/GIT/Bachelor/Data/Mnist/MnistSmol.csv";

    ifstream fin;
    string line;

    // Opens file on path    
    fin.open(path);
    // Used for inserting into arr
    int ix = 0;

    while(!fin.eof() && ix < NumberOfImages){
        fin >> line;
        vector<int> img (DataOfAnImage, 0);

        // This function just splits the ","'s and inserts them neatly into img
        // Python equivilant i [int(x) for x in line.split(",")]
        split(line, img);
        arr.at(ix++) = img;
    }
}

// This function does a lot of stuff, here's a quick rundown
void graph_from_imgs(AMGraph3D& G, float M[__num_of_imgs][__num_of_imgs], vector<vector<int>>& imgs) {
    
}



int main(int argc, char* argv[])
{
    // Reading mnist numbers into vector of vectors.
    vector<vector<int>> imgs;
    read_mnist(__num_of_imgs, __img_size, imgs);
    
    // Creating an AMGraph3D for all the points.
    AMGraph3D G;   
    float M[__num_of_imgs][__num_of_imgs];
    graph_from_imgs(G, M, imgs);
    
    cout << "Len of imgs:\t" << imgs.size() << endl;
    cout << "Len of img:\t" << imgs.at(0).size() << endl;

    return 0;
}