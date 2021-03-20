//
//  skeletor.cpp
//  skeletor
//
//  Created by Andreas Bærentzen on 02/03/2020.
//  Copyright © 2020 J. Andreas Bærentzen. All rights reserved.
//

#include <GEL/CGLA/CGLA.h>
#include <GEL/Geometry/Graph.h>
#include <GEL/HMesh/HMesh.h>

#include "graph_io.h"
#include "graph_skeletonize.h"
#include "graph_util.h"

#include <chrono>
#include <random>
#include <fstream>

using namespace std;
using namespace HMesh;
using namespace Geometry;
using namespace CGLA;

using NodeID = AMGraph3D::NodeID;

double quality_noise_level = 0.0875;
int front_passes = 100;
vector<VertexID> vertexIdsFound;


void mesh_skeleton(const AMGraph3D& g, const std::string& fname) {
    Manifold m;
    auto [c,r] = approximate_bounding_sphere(g);
    graph_to_mesh_cyl(g, m, 0.005 * r);
    obj_save(fname, m);
}

AttribVecDouble convert_to_attrib(VertexAttributeVector<double> inp) {
    AttribVecDouble dist;
    for(int i = 0; i < inp.size(); i++){
        dist[i] = inp.get(VertexID(i));
    }
    return dist;
}

// Finds distance based on the node furthes away from a random node 
AttribVecDouble findFurthestAway(Manifold& the_m) { 
    int random_index = rand() % int(the_m.allocated_vertices());
    VertexID random_id = VertexID(random_index);

    DijkstraOutput first_dij = Dijkstra(the_m, random_id, the_m.vertices());
    VertexAttributeVector<double> first_dist = first_dij.dist;

    double biggest_dist = -INFINITY;
    VertexID biggest_id;

    VertexID incrID;
    for(int i = 0; i < first_dist.size(); i++) {
        incrID = VertexID(i);
        if(first_dist[incrID] > biggest_dist) {
            if (!std::count(vertexIdsFound.begin(), vertexIdsFound.end(), incrID)){
                biggest_dist = first_dist[incrID];
                biggest_id = VertexID(incrID.get_index());
            }
        }
    }

    vertexIdsFound.insert(vertexIdsFound.end(), biggest_id);

    DijkstraOutput scnd_dij = Dijkstra(the_m, biggest_id, the_m.vertices());
    VertexAttributeVector<double> scnd_dist = scnd_dij.dist;

    return convert_to_attrib(scnd_dist);
}


std::pair<double, double> calc_dist(const AMGraph3D& old_g, const AMGraph3D& new_g) {
    // Calculates distance between two graphs.
    // Only making function, as a comment to point to the calculation in graph_util.h
    auto [old_g_avg, old_g_max] = graph_H_dist(old_g, new_g);
    auto [new_g_avg, new_g_max] = graph_H_dist(new_g, old_g);

    auto return_avg = (old_g_avg > new_g_avg) ? old_g_avg : new_g_avg;
    auto return_max = (old_g_max > new_g_max) ? old_g_max : new_g_max;

    return make_pair(return_avg, return_max);
}

int main(int argc, char* argv[]) {

    // Thrre inputs (file_in (.obj or .graph), file out (.graph), folder for data (./*)
    const string fn = argv[1];
    const string fn_out = argv[2];
    const string folder_out = argv[3];

    if(argc == 5) {
        front_passes = atoi(argv[4]);
    }

    AMGraph3D g;
    vector<AttribVecDouble> dvv;

    auto l = fn.length();
    if(fn.substr(l-5,5) == "graph") {
        g = graph_load(fn);
        mesh_skeleton(g,fn_out);
        exit(0);
    }

    Manifold m;
    if (!load(fn, m)){
        cout << "Could not load file" << endl;
        exit(-1);
    }

    VertexAttributeVector<NodeID> v2n;
    for(auto v : m.vertices())
        v2n[v] = g.add_node(m.pos(v));
    for(auto h: m.halfedges()) {
        Walker w = m.walker(h);
        if(h<w.opp().halfedge())
            g.connect_nodes(v2n[w.opp().vertex()], v2n[w.vertex()]);
    }

    srand(time(0));
    for(int i = 0; i < front_passes; i++) {
        dvv.push_back(findFurthestAway(m));
    }

    using hrc = chrono::high_resolution_clock;
    auto t1 = hrc::now();
    NodeSetVec front_seps = front_separators(g, dvv);
    pair<AMGraph3D, Util::AttribVec<AMGraph::NodeID, AMGraph::NodeID>> front_skel = skeleton_from_node_set_vec(g, front_seps, 1, 1);
    auto t2 = hrc::now();
    if (!graph_save(fn_out + "_front.graph", front_skel.first))
        std::cout << "Could not save: " << fn_out + "_front.graph" << endl;
    double front_time = (t2-t1).count() * 1e-9;
    // std::cout << "Using front seperators: " << (t2-t1).count() * 1e-9 << endl;
    
    // Local seps, mainly used for comparison
    t1 = hrc::now();
    NodeSetVec local_seps = local_separators(g, quality_noise_level);
    pair<AMGraph3D, Util::AttribVec<AMGraph::NodeID, AMGraph::NodeID>> local_skel = skeleton_from_node_set_vec(g, local_seps, 1, 1);
    t2 = hrc::now();
    if (!graph_save(fn_out + "_local.graph", local_skel.first))
        std::cout << "Could not save: " << fn_out +  "_local.graph" << endl;
    double local_time = (t2-t1).count() * 1e-9;
    // std::cout << "Using local seperators: " << (t2-t1).count() * 1e-9 << endl;

    auto [avg_dist, max_dist] = calc_dist(front_skel.first, local_skel.first);
    
    auto [center, bound_dist] = approximate_bounding_sphere(g);

    double n_avg_dist = avg_dist / bound_dist * 2 ;
    double n_max_dist = max_dist / bound_dist * 2;

    ofstream myfile;
    myfile.open(folder_out + "/data_out.json");
    myfile.clear();
    myfile << "{";
        myfile << "\"n_avg_dist\": \"" << n_avg_dist << "\",\n";
        myfile << "\"n_max_dist\":\"" << n_max_dist << "\",\n";
        myfile << "\"avg_dist\":\"" << avg_dist << "\",\n";
        myfile << "\"max_dist\":\"" << max_dist << "\",\n";
        myfile << "\"local_time\":\"" <<  local_time << "\",\n";
        myfile << "\"front_time\":\"" <<  front_time << "\",\n";
        myfile << "\"num_of_verts\":\"" <<  m.allocated_vertices() << "\",\n";
        myfile << "\"num_of_faces\":\"" <<  m.allocated_faces() << "\"\n";
    myfile << "}";
    myfile.close();

    return 0;
}
