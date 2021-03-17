#ifndef Nodes_H
#define Nodes_H
#include "Variables.h"
#include <iostream>
#include <set>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "utils/system_utils.h"
using namespace std;

class Node {
public:
    Node(){};
    Node(string type) : type(type){};
    string type;
};

class relation_type_vertex : public Node {
public:
    relation_type_vertex(){};
    relation_type_vertex(string type, vector<string> tuple_type,
                         map<string, AbstractVariable> vars)
        : Node(type), tuple_type(tuple_type), vars(vars){};

    // vector of obj types
    // neigbor object_type_vertices[i]==tuple_type[i]
    vector<string> tuple_type;

    // embedding_sizes for each time-step (not summed) [0,K+1]
    vector<int> embedding_sizes;

    // input_sizes for each time-step [0,K]
    vector<int> input_sizes;

    // AbsVar name -> AbstractVariable
    map<string, AbstractVariable> vars;

    // total embedding_size
    int Total_size = 0;
};

class RelationNode : public relation_type_vertex {
public:
    RelationNode(){};
    RelationNode(relation_type_vertex rtv, vector<string> instantiated_tuples,
                 vector<InstantiatedVariable> Inst_vars)
        : relation_type_vertex(rtv),
          instantiated_tuples(instantiated_tuples),
          Inst_vars(Inst_vars) {
        if (instantiated_tuples.size() != rtv.tuple_type.size() &&
            Inst_vars.size() == rtv.vars.size()) {
            SystemUtils::abort("relvfevf");
        }
    };

    // vector of objs , also neigbor object_nodes
    vector<string> instantiated_tuples;

    // ordered as map<string, AbstractVariable> vars
    vector<InstantiatedVariable> Inst_vars;
};

class object_type_vertex : public Node {
public:
    object_type_vertex(){};
    object_type_vertex(string type, set<string> neighbor_edge_types)
        : Node(type), neighbor_edge_types(neighbor_edge_types){};

    set<string> neighbor_edge_types;

    // input_sizes for each time-step and for each edge types
    map<int, map<string, int>> input_sizes_edges_types;

    // (Not summed) embedding_sizes for each time-step
    vector<int> embedding_sizes;

    // total embedding_size
    int Total_size = 0;
};

class ObjectNode : public object_type_vertex {
public:
    ObjectNode(){};
    ObjectNode(object_type_vertex otv, set<string> neighbor_edges,
               map<string, set<string>> neighbor_type2edges, string obj)
        : object_type_vertex(otv),
          neighbor_edges(neighbor_edges),
          neighbor_type2edges(neighbor_type2edges),
          obj(obj){};

    set<string> neighbor_edges;
    map<string, set<string>> neighbor_type2edges;
    string obj;
};

#endif
