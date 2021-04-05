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
const string __path = "E:/GIT/Bachelor/Data/Mnist/mnist_test.csv";

// Functions

void print_img(vector<int> vec) 
{
    for(int i=0; i < vec.size(); i++)
        cout << vec.at(i) << " ";
}

void show_similarity(vector<vector<float>> &M, AMGraph3D &g){
    // First is the total sum of equality
    vector<pair<int, int>> count (10);
    for(int i = 0; i < count.size(); i++){
        count[i] = make_pair(0, 0);
    }
    for(int i = 0; i < M.size(); i++){
        for(int j = i+1; j < M.size(); j++){
            int i_label = g.pos[i][0], j_label = g.pos[j][0];
            if(i_label == j_label) {
                count[i_label].first += M[i][j];
                count[i_label].second++;
            }
        }
    }
    cout << "Summary of similarity label statiscics\n--------------------------------------" << endl;
    for(int i = 0; i < count.size(); i++){
        cout << "Label: " << i << "\t|\tCount: " << count[i].second << "\t|\tavg. sim: " << float(count[i].first/count[i].second) << endl;
    }

}

void split_mnist_line(const string& str, vector<int>& vec, char delim = ',')
{
    stringstream ss(str);
    string token;
    int i= 0;
    while (std::getline(ss, token, delim)) {
        vec.at(i++) = std::stoi(token);
    } 
}

// Function that reads a number of mnist numbers from path file given in function
// and inserts them in given vec<vec<int>>.
int read_mnist(int NumberOfImages, int DataOfAnImage, vector<vector<int>> &imgs)
{
    imgs.resize(NumberOfImages, vector<int>(DataOfAnImage));

    ifstream fin;
    string line;

    // Opens file on path    
    fin.open(__path);
    // Used for inserting into arr
    int ix = 0;

    while(fin.peek()!=EOF && ix < NumberOfImages) {
        fin >> line;
        vector<int> img (DataOfAnImage, 0);
        // This function just splits the ","'s and inserts them neatly into img
        // Python equivilant i [int(x) for x in line.split(",")]
        split_mnist_line(line, img);
        imgs.at(ix++) = img;
    }

    // Removing last elements of vector, if not enought elements of images
    if(ix < NumberOfImages){
        imgs.erase(imgs.begin() + ix, imgs.end());
    }
    return ix;
}

float calc_dist(vector<int> &img_a, vector<int> &img_b)
{
    int sum_a = 0, sum_b = 0;
    for(int pixel : img_a) if(pixel==0) sum_a++;
    for(int pixel : img_b) if(pixel==0) sum_b++;
    return float((__img_size - abs(sum_a - sum_b))/__img_size);
}


// This function does a lot of stuff, here's a quick rundown
// Firstly it inserts points from imgs into the AMGraph3D. Now, the graph is 3D,
// but there are more than 3 values in the imag es.
// That is very true, and therefore it is only the label that is inserted into
// all 3 spots actually. This is okayish, as we only use the height functions to create the graphs
// when using front_separators.
// Next we create this matrix M, which is the distances from all vecs to all other vectors, and this is the
// interesting part.
void graph_from_imgs(AMGraph3D& g, vector<vector<float>>& M, vector<vector<int>>& imgs, int likeness=20) {
    for(int i = 0; i < imgs.size(); i++){
        for(int j = 0; j < imgs.size(); j++){
            float dist = calc_dist(imgs.at(i), imgs.at(j));
            M[i][j] = dist;
        }
    }
    
    vector<Vec3d> positions(imgs.size());

    // Just adding the img to the AMGraph3D
    // Adding the different img vertices to AMGraph    
    VertexAttributeVector<NodeID> v2n;
    int ixx = 0;
    for(vector<int> img : imgs) {
        VertexID vid = VertexID(ixx++);
        v2n[vid] = g.add_node(Vec3d(img[0]));
        cout << vid << endl;
    }

        
}

int main(int argc, char* argv[])
{
    // Reading mnist numbers into vector of vectors.
    vector<vector<int>> imgs;
    int num_of_imgs = read_mnist(100, __img_size, imgs);

    // Creating an AMGraph3D for all the points.
    AMGraph3D g;
    vector<vector<float>> M(num_of_imgs, vector<float> (num_of_imgs, 0.f));
    cout << "imgs dimensions: Rows->" << imgs.size() << ", Cols->" << imgs.at(0).size() << endl;
    cout << "M dimensions:    Rows->" << M.size() << ", Cols->" << M.at(0).size() << endl;

    graph_from_imgs(g, M, imgs);

    // Making the height functions
    vector<AttribVecDouble> dvv; 
    
    cout << "Len of imgs:\t" << imgs.size() << endl;
    cout << "Size of img:\t" << imgs.at(0).size() << endl;
    // show_similarity(M, g);

    return 0;
}