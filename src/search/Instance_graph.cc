#include "Instance_graph.h"

#include "Variables.h"
#include "assert.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace std;
InstanceGraph::InstanceGraph(domian_graph& dg,
                             vector<InstantiatedVariable>& StateFluents,
                             vector<InstantiatedVariable>& NonFluents,
                             vector<InstantiatedVariable>& ActionFluents,
                             map<string, string> Obj2types) {
    vector<InstantiatedVariable> all;
    all.insert(all.end(), StateFluents.begin(), StateFluents.end());
    all.insert(all.end(), NonFluents.begin(), NonFluents.end());
    all.insert(all.end(), ActionFluents.begin(), ActionFluents.end());
    map<string, InstantiatedVariable> Inst_map;
    for (size_t i = 0; i < all.size(); i++) {
        Inst_map[all[i].fullname] = all[i];
    }

    // relation-nodes
    map<string, vector<string>> tupleName2tuple;
    map<string, string> tupleName2types;
    map<string, vector<InstantiatedVariable>> tupleName2Vars;

    //(1)
    for (size_t _ = 0; _ < 1; _++) {
        vector<string> tmp2;
        tmp2.push_back("master");
        string tpN = vec_2_str(tmp2);
        tupleName2tuple[tpN] = tmp2;
        // type
        if (dg.Domian_relation_V_map.count(tpN) != 0) {
            tupleName2types[tpN] = tpN;
        } else {
            SystemUtils::abort("1111111111" + tpN);
        }
        // InstVar
        map<string, AbstractVariable>::iterator it2;
        vector<InstantiatedVariable> InstVar;
        for (it2 = dg.Domian_relation_V_map[tpN].vars.begin();
             it2 != dg.Domian_relation_V_map[tpN].vars.end(); it2++) {
            vector<string> empty_;
            if (Inst_map.count(it2->second.name) == 0) {
                SystemUtils::abort("44444" + it2->second.name);
            }

            string full_Name_ = it2->second.name;
            if (Inst_map.count(full_Name_) == 0) {
                SystemUtils::abort("not found 1." + full_Name_);
            }

            InstantiatedVariable instV = Inst_map[full_Name_];

            InstVar.push_back(instV);
        }
        tupleName2Vars[tpN] = InstVar;
    }

    //(2)
    for (size_t i = 0; i < all.size(); i++) {
        if (all[i].instantiated_patameters.size() > 0) {
            string tpN = vec_2_str(all[i].instantiated_patameters);
            tupleName2tuple[tpN] = all[i].instantiated_patameters;

            // type
            string tptype = vec_2_str(all[i].patameters_types);
            if (dg.Domian_relation_V_map.count(tptype) != 0) {
                tupleName2types[tpN] = tptype;
            } else {
                SystemUtils::abort("222222: " + tpN + "," + tptype);
            }

            // InstVar
            if (tupleName2Vars.count(tpN) == 0) {
                map<string, AbstractVariable>::iterator it2;
                vector<InstantiatedVariable> InstVar;
                for (it2 = dg.Domian_relation_V_map[tptype].vars.begin();
                     it2 != dg.Domian_relation_V_map[tptype].vars.end();
                     it2++) {
                    string fullName_ = it2->second.name + tpN;

                    if (Inst_map.count(fullName_) == 0) {
                        SystemUtils::abort(
                            "full name not in InstVariables, have space?---" +
                            fullName_ + ".");
                    }
                   InstantiatedVariable instV = Inst_map[fullName_];
                    InstVar.push_back(instV);
                }
                tupleName2Vars[tpN] = InstVar;
            }
        }
    }

    //(3)
    map<string, string>::iterator it1;
    for (it1 = Obj2types.begin(); it1 != Obj2types.end(); it1++) {
        vector<string> tmp3;
        tmp3.push_back("master");
        tmp3.push_back(it1->first);
        string tpN = vec_2_str(tmp3);
        tupleName2tuple[tpN] = tmp3;

        // type
        vector<string> tmp4;
        tmp4.push_back("master");
        tmp4.push_back(it1->second);
        string tptype = vec_2_str(tmp4);
        if (dg.Domian_relation_V_map.count(tptype) != 0) {
            tupleName2types[tpN] = tptype;
        } else {
            SystemUtils::abort("333333333: " + tpN + "," + tptype);
        }

        // InstVar
        vector<InstantiatedVariable> empty_;
        tupleName2Vars[tpN] = empty_;
    }
    map<string, vector<string>>::iterator it2;
    for (it2 = tupleName2tuple.begin(); it2 != tupleName2tuple.end(); it2++) {
        string tptype = tupleName2types[it2->first];
        assert(dg.Domian_relation_V_map.count(tptype) > 0);
        RelationNode rn = RelationNode(dg.Domian_relation_V_map[tptype],
                                       it2->second, tupleName2Vars[it2->first]);

        // prune the tuples iff
        // (1) it is merely the arguments of some nonfluents and the values of
        // those nonfluents are false .
        // (2) not the argument of any action flunts.
        // (3) do not have 'master' object in it.
        bool can_prune = true;
        vector<InstantiatedVariable>::iterator it3;
        for (it3 = rn.Inst_vars.begin(); it3 != rn.Inst_vars.end(); it3++) {
            if (it3->variable_type != "NON_FLUENT" ||
                abs(it3->value-(double)0.0) >= 0.000001) {
                can_prune = false;
            }
        }
        for (size_t i = 0; i < ActionFluents.size(); i++) {
            if (this->vec_2_str(ActionFluents[i].instantiated_patameters) ==
                it2->first) {
                can_prune = false;
            }
        }
        for (size_t i = 0; i < it2->second.size(); i++) {
            if (it2->second[i] == "master") {
                can_prune = false;
            }
        }

        if (!can_prune) {
            this->Inctance_relation_N_map[it2->first] = rn;
        } else {
            cout << "tuple :    ." << it2->first << ".     is pruned." << endl;
        }
    };

    // object-node and edges
    Obj2types["master"] = "master";
    for (it1 = Obj2types.begin(); it1 != Obj2types.end(); it1++) {
        map<string, set<string>> Etypes_2_Edges;
        set<string> neighbor_edges_;

        map<std::string, RelationNode>::iterator it3;
        for (it3 = this->Inctance_relation_N_map.begin();
             it3 != this->Inctance_relation_N_map.end(); it3++) {
            for (int i = 0; i < it3->second.instantiated_tuples.size(); i++) {
                if (it1->first == it3->second.instantiated_tuples[i]) {
                    domain_edge dee =
                        domain_edge(it1->second, it3->second.type, i);

                    if (dg.DomainEdgeTypeMap.count(dee.type) == 0) {
                        SystemUtils::abort("inst edge:" + dee.type);
                    }

                    Instance_edge inst_e = Instance_edge(
                        dg.DomainEdgeTypeMap[dee.type], it1->first, it3->first);

                    cout << "Inst egdes: " << inst_e.Inst_name
                         << "    ,type:" << inst_e.type << endl;

                    // inst edges
                    this->Instance_edge_maps[inst_e.Inst_name] = inst_e;

                    if (Etypes_2_Edges.count(dee.type) == 0) {
                        set<string> es;
                        es.insert(inst_e.Inst_name);
                        Etypes_2_Edges[dee.type] = es;
                    } else {
                        Etypes_2_Edges[dee.type].insert(inst_e.Inst_name);
                    }

                    neighbor_edges_.insert(inst_e.Inst_name);
                }
            }
        }

        if (dg.Domian_Object_V_map.count(it1->second) == 0) {
            SystemUtils::abort("66686");
        }

        ObjectNode objN =
            ObjectNode(dg.Domian_Object_V_map[it1->second], neighbor_edges_,
                       Etypes_2_Edges, it1->first);

        this->Inctance_Obj_N_map[it1->first] = objN;
    }
}

string InstanceGraph::vec_2_str(vector<string> v1) {
    string res = "(";
    for (size_t i = 0; i < v1.size(); i++) {
        res += v1[i];
        if (i < v1.size() - 1) {
            res += ",";
        }
    }
    res += ")";
    return res;
}
