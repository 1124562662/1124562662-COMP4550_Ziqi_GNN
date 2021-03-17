#include "Domian_graph.h"
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
void domian_graph::Set_sizes(int Initial_relation_node_output_size,
                             double relation_node_growing_speed,
                             int time_step_K, string aggregationtMethod) {
    this->aggregationtMethod = aggregationtMethod;

    if (aggregationtMethod == "CONCATE") {
        // initial relation embeddings
        map<string, relation_type_vertex>::iterator it2 =
            Domian_relation_V_map.begin();
        while (it2 != Domian_relation_V_map.end()) {
            int ini_size = it2->second.vars.size();
            it2->second.embedding_sizes.push_back(ini_size);
            it2->second.Total_size = ini_size;

            it2++;
        }

        // relation node combine with itself and the latest object embeddings
        // object nodes now aggregate the total relation node

        for (size_t k = 0; k < time_step_K; k++) {
            // Object Node
            map<string, object_type_vertex>::iterator it1 =
                Domian_Object_V_map.begin();

            while (it1 != Domian_Object_V_map.end()) {
                map<string, int> tmp;
                int total_ = 0;
                set<string>::iterator it3 =
                    it1->second.neighbor_edge_types.begin();
                while (it3 != it1->second.neighbor_edge_types.end()) {
                    string r_name =
                        this->DomainEdgeTypeMap[*it3].relation_type_vertex;

                    int edge_size =
                        this->Domian_relation_V_map[r_name].Total_size;
                    tmp[*it3] = edge_size;
                    total_ += edge_size;

                    it3++;
                }
                it1->second.input_sizes_edges_types[k] = tmp;

                it1->second.embedding_sizes.push_back(total_);

                if (k == 0) {
                    it1->second.Total_size = 0;
                }

                it1->second.Total_size += total_;

                it1++;
            }

            // relation Node
            // K+1

            it2 = Domian_relation_V_map.begin();
            while (it2 != Domian_relation_V_map.end()) {
                int tt = 0;
                for (size_t i = 0; i < it2->second.tuple_type.size(); i++) {
                    string objType = it2->second.tuple_type[i];
                    tt += this->Domian_Object_V_map[objType].embedding_sizes[k];
                };
                tt += it2->second.Total_size;

                it2->second.input_sizes.push_back(tt);

                int new_ebd_size = Initial_relation_node_output_size +
                                   (int)(relation_node_growing_speed * k);
                if (tt == 0) {
                    new_ebd_size = 0;
                }
                // ! plus k here
                it2->second.embedding_sizes.push_back(new_ebd_size);

                it2->second.Total_size += new_ebd_size;

                it2++;
            }
        }
    } else if (aggregationtMethod == "ATTENTION") {
        // initial relation embeddings
        map<string, relation_type_vertex>::iterator it2 =
            Domian_relation_V_map.begin();
        while (it2 != Domian_relation_V_map.end()) {
            int ini_size = it2->second.vars.size();
            it2->second.embedding_sizes.push_back(ini_size);
            it2->second.Total_size = ini_size;
            it2++;
        }

        for (size_t k = 0; k < time_step_K; k++) {
            // Object Node
            map<string, object_type_vertex>::iterator it1 =
                Domian_Object_V_map.begin();

            while (it1 != Domian_Object_V_map.end()) {
                map<string, int> tmp;
                int cout = 0;
                set<string>::iterator it3 =
                    it1->second.neighbor_edge_types.begin();
                while (it3 != it1->second.neighbor_edge_types.end()) {
                    string r_name =
                        this->DomainEdgeTypeMap[*it3].relation_type_vertex;

                    int edge_size =
                        this->Domian_relation_V_map[r_name].Total_size;

                    cout = max(cout, edge_size);

                    it3++;
                }
                it1->second.input_sizes_edges_types[k] = tmp;

                // difference!
                // int edge_level_attention_input_size = ;
                // int new_object_ebd_size = ;
                // difference!
                // it1->second.embedding_sizes.push_back(new_object_ebd_size);

                if (k == 0) {
                    it1->second.Total_size = 0;
                }

                // it1->second.Total_size += new_object_ebd_size;

                it1++;
            }
        }

    } else {
        SystemUtils::abort("domain g agg string is :" + aggregationtMethod);
    }
}
