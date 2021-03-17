#ifndef Instancegraph_H
#define Instancegraph_H
#include "Domian_graph.h"
#include "Edges.h"
#include "Nodes.h"
#include "Variables.h"
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace std;
class InstanceGraph {
public:
    InstanceGraph(){};
    InstanceGraph(domian_graph& dg, vector<InstantiatedVariable>& StateFluents,
                  vector<InstantiatedVariable>& NonFluents,
                  vector<InstantiatedVariable>& ActionFluents,map<string, string> Obj2types);

    domian_graph* dg;

    map<string, Instance_edge> Instance_edge_maps;

    map<string, RelationNode> Inctance_relation_N_map;
    map<string, ObjectNode> Inctance_Obj_N_map;

private:
    string vec_2_str(vector<string> v1);
};
#endif
