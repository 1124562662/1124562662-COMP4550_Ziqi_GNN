#ifndef EDGE_H
#define EDGE_H
#include <iostream>
#include <set>
#include <string>
#include <vector>
using namespace std;

class domain_edge {
public:
    domain_edge(){};
    domain_edge(string object_type_vertex, string relation_type_vertex,
                int occurence)
        : object_type_vertex(object_type_vertex),
          relation_type_vertex(relation_type_vertex),
          occurence(occurence) {
        this->type = "(" + object_type_vertex + "," + relation_type_vertex +
                     "," + to_string(occurence) + ")";
    };
    string object_type_vertex;
    string relation_type_vertex;
    int occurence;
    string type;
};
class Instance_edge : public domain_edge {
public:
    Instance_edge(){};
    Instance_edge(domain_edge de, string object_node, string relation_node)
        : domain_edge(de),
          object_node(object_node),
          relation_node(relation_node) {
        this->Inst_name = "(" + object_node + "," + relation_node + "," +
                          to_string(de.occurence) + ")";
    };

    string object_node;
    string relation_node;
    string Inst_name;
};
#endif
