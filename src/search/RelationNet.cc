#include "RelationNet.h"
#include "Variables.h"
#include "assert.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <torch/torch.h>
using namespace std;

torch::Tensor relation_net::forward(torch::Tensor X) {
    // fullname of fluents -> their values

    int siz = X.size(0);

    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    std::map<std::string, at::Tensor>::iterator it1;
    for (it1 = this->Inputs.begin(); it1 != this->Inputs.end(); it1++) {
        double old = (double)(it1->second[0][0].item<double>());
        this->Inputs[it1->first] = torch::full({siz, 1}, old, options);
    }

    X = X.t();

    std::map<std::string, int>::iterator it;
    for (it = this->stateVariableIndices_net.begin();
         it != this->stateVariableIndices_net.end(); it++) {
        torch::Tensor tensor_ = X[it->second].clone().reshape({X.size(1), 1});

        this->Inputs[it->first] = tensor_;
    };
    // fullname of the action -> result of the action module
    map<string, torch::Tensor> ResMap = this->forward_helper(siz);
    // output of no-op is in pos 0 of res
    torch::Tensor res = ResMap["no-op"];

    for (int i = 1; i < this->res_order.size(); i++) {
        res = torch::cat({res, ResMap[(this->res_order)[i]]}, 1);
    };

    return res;
}

torch::Tensor relation_net::forward() {
    if (state_net->size() == 0) {
        SystemUtils::abort(
            "state_pointer_set not set! use set_state_pointer()");
    }
    // std::cout << this->parameters() << std::endl;
    std::vector<double> current_state_ = *(this->state_net);
    cout << "curr state vec: ";
    for (int i = 0; i < current_state_.size(); i++) {
        cout << to_string((int)current_state_[i]);
    }
    cout << endl;
    return this->forward(*(this->state_net));
}

void relation_net::set_instance_tensor() {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    for (int i = 0; i < (this->StateFluents).size(); i++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, (this->StateFluents)[i].value, options);
        this->Inputs[(this->StateFluents)[i].fullname] = tensor_;
    }
    for (int i = 0; i < this->NonFluents.size(); i++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, (this->NonFluents)[i].value, options);
        this->Inputs[(this->NonFluents)[i].fullname] = tensor_;
    }

    vector<string> tmpv(this->action_indices_.size() + 1);
    this->res_order = tmpv;
    map<string, int>::iterator itt;
    for (itt = this->action_indices_.begin();
         itt != this->action_indices_.end(); itt++) {
        (this->res_order)[itt->second] = itt->first;
    };

    for (int i = 0; i < this->ActionFluents.size(); i++) {
        if ((this->ActionFluents)[i].fullname != "no-op") {
            bool found = false;
            for (int j = 0; j < (this->res_order).size(); j++) {
                if ((this->res_order)[j] == (this->ActionFluents)[i].fullname) {
                    found = true;
                };
            };
            if (!found) {
                SystemUtils::abort(
                    " (this->ActionFluents)[i].fullname  NOT IN res_order");
            };
        };
    };
}

map<string, torch::Tensor> relation_net::forward_helper(int batch_size) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    map<int, map<string, torch::Tensor>> last_relation_ebding;
    map<int, map<string, torch::Tensor>> last_obj_ebding;
    // map<string, torch::Tensor> Total_relation_ebding;

    // initial embeddings
    map<std::string, RelationNode>::iterator it2;
    for (it2 = this->ig.Inctance_relation_N_map.begin();
         it2 != this->ig.Inctance_relation_N_map.end(); it2++) {
        torch::Tensor re;
        for (size_t i = 0; i < it2->second.Inst_vars.size(); i++) {
            torch::Tensor tmp = this->Inputs[it2->second.Inst_vars[i].fullname];
            if (!re.defined()) {
                re = tmp;
            } else {
                re = torch::cat({re, tmp}, 1);
            }
        }
        last_relation_ebding[0][it2->first] = re;
    }

    for (size_t k = 0; k < this->time_step_K; k++) {
        // obj aggregation

        map<string, ObjectNode>::iterator it3;
        for (it3 = this->ig.Inctance_Obj_N_map.begin();
             it3 != this->ig.Inctance_Obj_N_map.end(); it3++) {
            torch::Tensor oe;

            // for each neighbor edge types
            set<string>::iterator it4;
            for (it4 = it3->second.neighbor_edge_types.begin();
                 it4 != it3->second.neighbor_edge_types.end(); it4++) {
                torch::Tensor oe_EdgeType;

                int count = it3->second.neighbor_type2edges[*it4].size();
                if (count == 0) {
                    int siz = it3->second.input_sizes_edges_types[k][*it4];
                    if (siz > 0) {
                        oe_EdgeType =
                            torch::full({batch_size, siz}, 0.0, options);
                    }

                } else {
                    torch::Tensor quotient;
                    set<string>::iterator it5;
                    map<string, torch::Tensor> attentions;

                    for (it5 = it3->second.neighbor_type2edges[*it4].begin();
                         it5 != it3->second.neighbor_type2edges[*it4].end();
                         it5++) {
                        string RNname =
                            this->ig.Instance_edge_maps[*it5].relation_node;
                        torch::Tensor attention_V;

                        if (last_relation_ebding[k][RNname].defined()) {
                            attention_V = torch::exp(
                                this
                                    ->ObjectNodeModules[k][it3->second.type]
                                                       [*it4]
                                    .forward(last_relation_ebding[k][RNname]));

                            if (!quotient.defined()) {
                                quotient = attention_V;
                            } else {
                                quotient = torch::add(quotient, attention_V);
                            }
                            attentions[*it5] = attention_V;
                        }
                    }

                    for (it5 = it3->second.neighbor_type2edges[*it4].begin();
                         it5 != it3->second.neighbor_type2edges[*it4].end();
                         it5++) {
                        string RNname =
                            this->ig.Instance_edge_maps[*it5].relation_node;
                        if (last_relation_ebding[k][RNname].defined()) {
                            // quotient = torch::add(
                            //     quotient, torch::full({1, 1}, 0.0, options));
                            torch::Tensor tmp2 =
                                torch::div(attentions[*it5], quotient);

                            // cout<<"quotient"<<tmp2 <<endl;

                            tmp2 = torch::mul(tmp2,
                                              last_relation_ebding[k][RNname]);
                            if (!oe_EdgeType.defined()) {
                                oe_EdgeType = tmp2;
                            } else {
                                oe_EdgeType = torch::add(oe_EdgeType, tmp2);
                            }
                        }
                    }
                }
                if (oe_EdgeType.defined()) {
                    if (!oe.defined()) {
                        oe = oe_EdgeType;
                    } else {
                        oe = torch::cat({oe, oe_EdgeType}, 1);
                    }
                }
            }
            last_obj_ebding[k][it3->first] = oe;
            // cout << "oe: " << it3->first << "," << to_string(k) << ", " << oe
            //      << endl;
        }

        // relation-embeddings
        for (it2 = this->ig.Inctance_relation_N_map.begin();
             it2 != this->ig.Inctance_relation_N_map.end(); it2++) {
            torch::Tensor re;

            if (it2->second.instantiated_tuples.size() == 0) {
                re = last_relation_ebding[k][it2->first];

            } else {
                re = last_relation_ebding[k][it2->first];

                for (size_t i = 0; i < it2->second.instantiated_tuples.size();
                     i++) {
                    string obj_ = it2->second.instantiated_tuples[i];
                    if (!re.defined()) {
                        if (last_obj_ebding[k][obj_].defined()) {
                            re = last_obj_ebding[k][obj_];
                        }
                    } else {
                        if (last_obj_ebding[k][obj_].defined()) {
                            re = torch::cat({re, last_obj_ebding[k][obj_]}, 1);
                        }
                    }
                }
            }
            // cout << "ccccccede" << to_string(k) << it2->first << endl;
            torch::Tensor nre;
            if (re.defined()) {
                nre =
                    this->RelationNodeModules[k][it2->second.type].forward(re);
            }

            if (!last_relation_ebding[k][it2->first].defined()) {
                last_relation_ebding[k + 1][it2->first] = nre;
            } else {
                if (nre.defined()) {
                    torch::Tensor nre_tmp;
                    nre_tmp = torch::cat(
                        {last_relation_ebding[k][it2->first], nre}, 1);
                    last_relation_ebding[k + 1][it2->first] = nre_tmp;
                }
            }
        }
    }

    // fullname of the action -> result of the action module
    map<string, torch::Tensor> ResMap;

    for (int i = 0; i < this->ActionFluents.size(); i++) {
        torch::Tensor input;

        if (this->ActionFluents[i].patameters_types.size() > 0) {
            string tpName =
                this->vec_2_str(ActionFluents[i].instantiated_patameters);

            input = last_relation_ebding[this->time_step_K][tpName];

            for (size_t k = 0; k < this->time_step_K; k++) {
                for (int j = 0;
                     j < ActionFluents[i].instantiated_patameters.size(); j++) {
                    torch::Tensor tmp3 =
                        last_obj_ebding[k][ActionFluents[i]
                                               .instantiated_patameters[j]];

                    if (input.defined()) {
                        if (tmp3.defined()) {
                            input = torch::cat({input, tmp3}, 1);
                        }
                    } else {
                        input = tmp3;
                    }
                }
            }
        } else {
            input = last_relation_ebding[this->time_step_K ]["(master)"];

            for (size_t k = 0; k < this->time_step_K; k++) {
                torch::Tensor ttpp = last_obj_ebding[k]["master"];
                if (input.defined()) {
                    if (ttpp.defined()) {
                        input = torch::cat({input, ttpp}, 1);
                    }
                } else {
                    input = ttpp;
                }
            }
        }

        // cout<< ActionFluents[i].name << "  input:"<<endl;cout<<input<<endl;

        torch::Tensor r = this->actions_m[ActionFluents[i].name].forward(input);
        ResMap[ActionFluents[i].fullname] = r;

        // cout << ActionFluents[i].name << "  output:" << r << endl;
    }

    // no-op

    torch::Tensor input = last_relation_ebding[this->time_step_K]["(master)"];
    for (size_t k = 0; k < this->time_step_K; k++) {
        torch::Tensor ttpp = last_obj_ebding[k]["master"];
        if (input.defined()) {
            if (ttpp.defined()) {
                input = torch::cat({input, ttpp}, 1);
            }
        } else {
            input = ttpp;
        }
    }
    ResMap["no-op"] = this->actions_m["no-op"].forward(input);
    // cout << "no-op" << input<< endl;
    return ResMap;
}

map<string, torch::Tensor> relation_net::forward_sogbofa(
    map<string, torch::Tensor>& state_fluents_map) {
    map<string, torch::Tensor>::iterator it;
    for (it = state_fluents_map.begin(); it != state_fluents_map.end(); it++) {
        this->Inputs[it->first] = it->second;
    };
    // fullname of the action -> result of the action module
    map<string, torch::Tensor> ResMap = this->forward_helper(1);
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    torch::Tensor donimonator = torch::full({1, 1}, 0.0, options);

    torch::Tensor nono;
    ResMap["no-op"] = nono;

    map<string, torch::Tensor>::iterator res_it;
    for (res_it = ResMap.begin(); res_it != ResMap.end(); res_it++) {
        if (res_it->first != "no-op") {
            donimonator = torch::add(donimonator, torch::exp(res_it->second));
        }
    }

    for (res_it = ResMap.begin(); res_it != ResMap.end(); res_it++) {
        if (res_it->first != "no-op") {
            res_it->second =
                torch::div(torch::exp(res_it->second), donimonator);
            // cout<<"donimonator"<<res_it->second<<endl;
        }
    }
    return ResMap;
}

torch::Tensor relation_net::forward(vector<double> const& current_state_) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    // fullname of fluents -> their values

    std::map<std::string, int>::iterator it;
    for (it = this->stateVariableIndices_net.begin();
         it != this->stateVariableIndices_net.end(); it++) {
        torch::Tensor tensor_ =
            torch::full({1, 1}, current_state_[it->second], options);
        this->Inputs[it->first] = tensor_;
    };

    // fullname of the action -> result of the action module
    map<string, torch::Tensor> ResMap = this->forward_helper(1);

    // output of no-op is in pos 0 of res
    torch::Tensor res = ResMap["no-op"];

    for (int i = 1; i < this->res_order.size(); i++) {
        res = torch::cat({res, ResMap[(this->res_order)[i]]}, 1);
    };
    return res;
}

void relation_net::set_modules() {
    for (size_t i = 0; i < this->time_step_K; i++) {
        // object_node
        map<string, object_type_vertex>::iterator it1 =
            this->dg.Domian_Object_V_map.begin();
        while (it1 != this->dg.Domian_Object_V_map.end()) {
            set<string>::iterator it3 = it1->second.neighbor_edge_types.begin();
            while (it3 != it1->second.neighbor_edge_types.end()) {
                int in_size = it1->second.input_sizes_edges_types[i][*it3];
                if (in_size > 0) {
                    vector<int> topo;

                    std::cout << "insize " << it1->first << ",  " << *it3
                              << ",  " << to_string(in_size) << endl;

                    topo.push_back(in_size);
                    int mid = (int)((in_size + 1) * 2 / 3);
                    if (mid <= 0) {
                        SystemUtils::abort("mid not zero!");
                    }
                    topo.push_back(mid);
                    topo.push_back(1);

                    bool use_o;
                    if (i == this->time_step_K) {
                        use_o = true;
                    } else {
                        use_o = false;
                    }

                    auto subm = std::make_shared<nn_module>(
                        "O," + it1->first + "," + *it3 + "," + to_string(i),
                        topo, use_o);
                    subm = torch::nn::Module::register_module(
                        "O," + it1->first + "," + *it3 + "," + to_string(i),
                        subm);
                    this->ObjectNodeModules[i][it1->first][*it3] = *subm;
                }

                it3++;
            }

            it1++;
        }

        // relation-node
        map<string, relation_type_vertex>::iterator it2 =
            this->dg.Domian_relation_V_map.begin();
        while (it2 != this->dg.Domian_relation_V_map.end()) {
            int in_size = it2->second.input_sizes[i];
            int Out_size = it2->second.embedding_sizes[i + 1];

            if (in_size > 0) {
                vector<int> topo;

                std::cout << "insize " << it2->first << ",    "
                          << to_string(in_size) << endl;

                topo.push_back(in_size);
                int mid = (int)(0.85 * in_size);
                if (mid <= 0) {
                    SystemUtils::abort("mid not zero!");
                }
                topo.push_back(mid);
                topo.push_back(Out_size);

                bool use_o;
                if (i >= this->time_step_K - 1) {
                    use_o = true;
                } else {
                    use_o = false;
                }

                auto subm = std::make_shared<nn_module>(
                    "R," + it2->first + "," + to_string(i), topo, use_o);
                subm = torch::nn::Module::register_module(
                    "R," + it2->first + "," + to_string(i), subm);
                this->RelationNodeModules[i][it2->first] = *subm;
            }

            it2++;
        }
    }
    // actions modules
    set<AbstractVariable, classcomp>::iterator it4 =
        ActionFluentsSchema.begin();
    while (it4 != ActionFluentsSchema.end()) {
        int in_size;
        if (it4->patameters_types.size() > 0) {
            string tpname = vec_2_str(it4->patameters_types);

            in_size = this->dg.Domian_relation_V_map[tpname].Total_size;

            for (size_t j = 0; j < it4->patameters_types.size(); j++) {
                in_size +=
                    this->dg.Domian_Object_V_map[it4->patameters_types[j]]
                        .Total_size;
            }
        } else {
            in_size = this->dg.Domian_relation_V_map["(master)"].Total_size;
            in_size += this->dg.Domian_Object_V_map["master"].Total_size;
        }

        if (in_size > 0) {
            vector<int> topo;
            std::cout << "insize " << it4->name << ",    " << to_string(in_size)
                      << endl;

            topo.push_back(in_size);

            int mid = (int)(0.5 * in_size);
            topo.push_back(mid);
            if (mid <= 0) {
                SystemUtils::abort("mid not zero!");
            }
            topo.push_back(mid);
            topo.push_back(1);

            auto subm =
                std::make_shared<nn_module>("A," + it4->name, topo, true);
            subm = torch::nn::Module::register_module("A," + it4->name, subm);
            this->actions_m[it4->name] = *subm;
        }

        it4++;
    }

    int in_size = this->dg.Domian_relation_V_map["(master)"].Total_size;
    in_size += this->dg.Domian_Object_V_map["master"].Total_size;

    if (in_size > 0) {
        vector<int> topo;

        std::cout << "insize master ,    " << to_string(in_size) << endl;

        topo.push_back(in_size);
        topo.push_back((int)(0.9 * in_size));
        topo.push_back((int)(0.7 * in_size));
        topo.push_back((int)(0.4 * in_size));
        assert((int)0.5 * in_size > 0);
        topo.push_back(1);
        auto subm = std::make_shared<nn_module>("A,no-op", topo, true);
        subm = torch::nn::Module::register_module("A,no-op", subm);
        this->actions_m["no-op"] = *subm;
    }
}

void relation_net::read_modules(string path) {
    std::ifstream taskFile;
    string taskFileName = path + "Modules_info";
    taskFile.open(taskFileName.c_str());
    if (taskFile.is_open()) {
        string line;
        getline(taskFile, line);
        assert(line == "relation_node_output_size");
        getline(taskFile, line);
        this->relation_node_output_size = stoi(line);
        std::cout << "dbkwudcb:::" << to_string(this->relation_node_output_size)
                  << endl;

        getline(taskFile, line);
        assert(line == "relation_node_growing_speed");
        getline(taskFile, line);
        this->relation_node_growing_speed = stof(line);

        getline(taskFile, line);
        assert(line == "time_step_K");
        getline(taskFile, line);
        this->time_step_K = stoi(line);

        taskFile.close();
    }

    this->dg.Set_sizes(this->relation_node_output_size,
                       this->relation_node_growing_speed, this->time_step_K,
                       "CONCATE");

    for (size_t i = 0; i < this->time_step_K; i++) {
        // obj node
        map<string, object_type_vertex>::iterator it1 =
            this->dg.Domian_Object_V_map.begin();
        while (it1 != this->dg.Domian_Object_V_map.end()) {
            set<string>::iterator it3 = it1->second.neighbor_edge_types.begin();
            while (it3 != it1->second.neighbor_edge_types.end()) {
                // TODO

                int in_size = it1->second.input_sizes_edges_types[i][*it3];
                if (in_size > 0) {
                    auto subm = std::make_shared<nn_module>();
                    string mname =
                        "O," + it1->first + "," + *it3 + "," + to_string(i);
                    subm->Load_submodel(this->ReadModulesParams_Path, mname);
                    subm = torch::nn::Module::register_module(mname, subm);
                    this->ObjectNodeModules[i][it1->first][*it3] = *subm;
                }

                it3++;
            }
            it1++;
        }

        // relation node
        map<string, relation_type_vertex>::iterator it2 =
            this->dg.Domian_relation_V_map.begin();
        while (it2 != this->dg.Domian_relation_V_map.end()) {
            int in_size = it2->second.input_sizes[i];
            if (in_size > 0) {
                auto subm = std::make_shared<nn_module>();
                string mname = "R," + it2->first + "," + to_string(i);
                subm->Load_submodel(this->ReadModulesParams_Path, mname);
                subm = torch::nn::Module::register_module(mname, subm);
                this->RelationNodeModules[i][it2->first] = *subm;
            }
            it2++;
        }
    }

    // actions modules
    set<AbstractVariable, classcomp>::iterator it4 =
        ActionFluentsSchema.begin();
    while (it4 != ActionFluentsSchema.end()) {
        int in_size;
        if (it4->patameters_types.size() > 0) {
            string tpname = vec_2_str(it4->patameters_types);

            in_size = this->dg.Domian_relation_V_map[tpname].Total_size;

            for (size_t j = 0; j < it4->patameters_types.size(); j++) {
                in_size +=
                    this->dg.Domian_Object_V_map[it4->patameters_types[j]]
                        .Total_size;
            }
        } else {
            in_size = this->dg.Domian_relation_V_map["(master)"].Total_size;
            in_size += this->dg.Domian_Object_V_map["master"].Total_size;
        }

        if (in_size > 0) {
            auto subm = std::make_shared<nn_module>();
            string mname = "A," + it4->name;
            subm->Load_submodel(this->ReadModulesParams_Path, mname);
            subm = torch::nn::Module::register_module(mname, subm);
            this->actions_m[it4->name] = *subm;
        }

        it4++;
    }
    int in_size = this->dg.Domian_relation_V_map["(master)"].Total_size;
    in_size += this->dg.Domian_Object_V_map["master"].Total_size;

    if (in_size > 0) {
        auto subm = std::make_shared<nn_module>();
        string mname = "A,no-op";
        subm->Load_submodel(this->ReadModulesParams_Path, mname);
        subm = torch::nn::Module::register_module(mname, subm);
        this->actions_m["no-op"] = *subm;
    }
}

void relation_net::save_modules(string path) {
    std::ofstream taskFile;
    std::string taskFileName = path + "Modules_info";
    taskFile.open(taskFileName.c_str());
    if (taskFile.is_open()) {
        taskFile << "relation_node_output_size" << endl;
        taskFile << this->relation_node_output_size << endl;
        taskFile << "relation_node_growing_speed" << endl;
        taskFile << this->relation_node_growing_speed << endl;
        taskFile << "time_step_K " << endl;
        taskFile << this->time_step_K << endl;

        taskFile.close();
    }

    map<string, nn_module>::iterator it2;
    map<int, map<string, nn_module>>::iterator it1 =
        RelationNodeModules.begin();
    while (it1 != RelationNodeModules.end()) {
        for (it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
            it2->second.save_submodel(path, it2->second.name);
        }
        it1++;
    }

    map<int, map<string, map<string, nn_module>>>::iterator it3 =
        ObjectNodeModules.begin();
    while (it3 != ObjectNodeModules.end()) {
        map<string, map<string, nn_module>>::iterator it4;
        for (it4 = it3->second.begin(); it4 != it3->second.end(); it4++) {
            for (it2 = it4->second.begin(); it2 != it4->second.end(); it2++) {
                it2->second.save_submodel(path, it2->second.name);
            }
        }
        it3++;
    }

    for (it2 = actions_m.begin(); it2 != actions_m.end(); it2++) {
        it2->second.save_submodel(path, it2->second.name);
    }
}

void relation_net::build_domain_graph() {
    // realtion V

    map<string, vector<string>> tupleName2tuple;
    map<string, map<string, AbstractVariable>> tupleName2AbsVars;

    set<AbstractVariable, classcomp>::iterator it2 =
        knowledgeFluentsSchema.begin();

    while (it2 != knowledgeFluentsSchema.end()) {
        if (it2->patameters_types.size() > 0) {
            string tpName = vec_2_str(it2->patameters_types);
            tupleName2tuple[tpName] = it2->patameters_types;
            tupleName2AbsVars[tpName][it2->name] = *it2;
        } else {
            vector<string> tmp;
            tmp.push_back("master");
            string tpName = vec_2_str(tmp);
            tupleName2tuple[tpName] = tmp;
            tupleName2AbsVars[tpName][it2->name] = *it2;
        }

        it2++;
    }
    it2 = ActionFluentsSchema.begin();
    while (it2 != ActionFluentsSchema.end()) {
        if (it2->patameters_types.size() > 0) {
            string tpName = vec_2_str(it2->patameters_types);
            tupleName2tuple[tpName] = it2->patameters_types;
        }
        it2++;
    }
    set<string>::iterator it1 = ALL_patameters_types.begin();
    while (it1 != ALL_patameters_types.end()) {
        vector<string> tmp;
        tmp.push_back("master");
        tmp.push_back(*it1);
        tupleName2tuple[vec_2_str(tmp)] = tmp;
        it1++;
    }

    map<string, vector<string>>::iterator it3 = tupleName2tuple.begin();
    while (it3 != tupleName2tuple.end()) {
        relation_type_vertex rtv = relation_type_vertex(
            it3->first, it3->second, tupleName2AbsVars[it3->first]);
        this->dg.Domian_relation_V_map[it3->first] = rtv;
        // print the initial embedding
        cout << it3->first << "    has ini embedding: ";
        map<std::string, AbstractVariable>::iterator it5;
        for (it5 = tupleName2AbsVars[it3->first].begin();
             it5 != tupleName2AbsVars[it3->first].end(); it5++) {
            cout << it5->first << ",  ";
        }
        cout << endl;
        cout << endl;
        //
        it3++;
    }

    if (this->dg.Domian_relation_V_map.count("(master)") == 0) {
        vector<string> tmp;
        tmp.push_back("master");
        std::map<std::string, AbstractVariable> p2;
        relation_type_vertex rtv = relation_type_vertex("(master)", tmp, p2);
        this->dg.Domian_relation_V_map["(master)"] = rtv;
        tupleName2tuple["(master)"] = tmp;
    }

    // object V
    set<string> Obj_withMaster = ALL_patameters_types;
    Obj_withMaster.insert("master");
    cout << "edge types:" << endl;
    it1 = Obj_withMaster.begin();
    while (it1 != Obj_withMaster.end()) {
        set<string> neighbor_edge_types;
        it3 = tupleName2tuple.begin();
        while (it3 != tupleName2tuple.end()) {
            vector<int> occurences = this->count_occurence(*it1, it3->second);
            if (occurences.size() > 0) {
                for (size_t i = 0; i < occurences.size(); i++) {
                    domain_edge de =
                        domain_edge(*it1, it3->first, occurences[i]);
                    neighbor_edge_types.insert(de.type);
                    this->dg.DomainEdgeTypeMap[de.type] = de;

                    cout << de.type << endl;
                }
            }
            it3++;
        }
        object_type_vertex otv = object_type_vertex(*it1, neighbor_edge_types);
        this->dg.Domian_Object_V_map[otv.type] = otv;

        it1++;
    }
}

void relation_net::readHelper(string filepath, map<string, string>* Obj2types,
                              set<string>* ALL_patameters_types) {
    std::ifstream taskFile;
    taskFile.open(filepath);
    if (taskFile.is_open()) {
        std::string line;
        set<string> duplicate_check;
        while (getline(taskFile, line)) {
            assert(line == "DIV");
            InstantiatedVariable var = InstantiatedVariable(taskFile);
            var.fullname = this->erasechar(var.fullname);

            if (var.variable_type == "STATE_FLUENT") {
                this->StateFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->StateFluentsSchema.insert(absvar);
                    this->knowledgeFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                }
                for (int i = 0; i < var.instantiated_patameters.size(); i++) {
                    ALL_patameters_types->insert(var.patameters_types[i]);
                    (*Obj2types)[var.instantiated_patameters[i]] =
                        var.patameters_types[i];
                }

            } else if (var.variable_type == "ACTION_FLUENT") {
                this->ActionFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->ActionFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                }
                for (int i = 0; i < var.instantiated_patameters.size(); i++) {
                    ALL_patameters_types->insert(var.patameters_types[i]);
                    (*Obj2types)[var.instantiated_patameters[i]] =
                        var.patameters_types[i];
                }

            } else if (var.variable_type == "NON_FLUENT") {
                this->NonFluents.push_back(var);
                AbstractVariable absvar = var;
                if (duplicate_check.count(absvar.name) == 0) {
                    this->NonFluentsSchema.insert(absvar);
                    this->knowledgeFluentsSchema.insert(absvar);
                    duplicate_check.insert(absvar.name);
                };
                for (int i = 0; i < var.instantiated_patameters.size(); i++) {
                    ALL_patameters_types->insert(var.patameters_types[i]);
                    (*Obj2types)[var.instantiated_patameters[i]] =
                        var.patameters_types[i];
                }
            } else {
                cout << "->" << var.variable_type << ". ";
                SystemUtils::abort("in initilase(), unkonwn type.");
            };
        };
        taskFile.close();
    };

    if (this->remove_file) {
        if (remove(filepath.c_str()) != 0) {
            SystemUtils::abort("remove");
        };
    };
}

void relation_net::save_indices() {
    string fileName = "Indices";
    std::ofstream taskFile;
    taskFile.open(fileName.c_str());

    map<string, int>::iterator it;
    taskFile << "stateVariableIndices_net" << endl;
    for (it = this->stateVariableIndices_net.begin();
         it != this->stateVariableIndices_net.end(); it++) {
        taskFile << it->first << endl;
        taskFile << it->second << endl;
    }
    taskFile << "end" << endl;

    taskFile << "action_indices_" << endl;
    for (it = this->action_indices_.begin(); it != this->action_indices_.end();
         it++) {
        taskFile << it->first << endl;
        taskFile << it->second << endl;
    }
    taskFile << "end" << endl;
    taskFile.close();
}

void relation_net::read_indices(string path) {
    string fileName = "Indices";
    std::ifstream taskFile;
    string taskFileName = path + fileName;
    taskFile.open(taskFileName.c_str());
    if (taskFile.is_open()) {
        std::string line;
        getline(taskFile, line);
        assert(line == "stateVariableIndices_net");
        getline(taskFile, line);
        while (line != "end") {
            string key = line;
            getline(taskFile, line);
            int v = stoi(line);
            this->stateVariableIndices_net[key] = v;
            getline(taskFile, line);
        }

        getline(taskFile, line);
        assert(line == "action_indices_");
        getline(taskFile, line);
        while (line != "end") {
            string key = line;
            getline(taskFile, line);
            int v = stoi(line);
            this->action_indices_[key] = v;
            getline(taskFile, line);
        }

        taskFile.close();
    }
}

bool relation_net::check_domain_schema(
    set<AbstractVariable, classcomp> const& StateFluentsSchema1,
    set<AbstractVariable, classcomp> const& NonFluentsSchema1,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema1,
    set<AbstractVariable, classcomp> const& StateFluentsSchema2,
    set<AbstractVariable, classcomp> const& NonFluentsSchema2,
    set<AbstractVariable, classcomp> const& ActionFluentsSchema2) {
    assert(StateFluentsSchema1.size() == StateFluentsSchema2.size());
    assert(NonFluentsSchema1.size() == NonFluentsSchema2.size());
    assert(ActionFluentsSchema1.size() == ActionFluentsSchema2.size());

    set<AbstractVariable, classcomp>::iterator it;
    set<AbstractVariable, classcomp>::iterator it2;
    it = StateFluentsSchema1.begin();
    it2 = StateFluentsSchema2.begin();
    while (it != StateFluentsSchema1.end()) {
        if (!(*it == *it2)) {
            SystemUtils::abort("domian check fails.");
        }
        it++;
        it2++;
    };
    it = NonFluentsSchema1.begin();
    it2 = NonFluentsSchema2.begin();
    while (it != NonFluentsSchema1.end()) {
        if (!(*it == *it2)) {
            SystemUtils::abort("domian check fails.");
        }
        it++;
        it2++;
    }
    it = ActionFluentsSchema1.begin();
    it2 = ActionFluentsSchema2.begin();
    while (it != ActionFluentsSchema1.end()) {
        if (!(*it == *it2)) {
            SystemUtils::abort("domian check fails.");
        }
        it++;
        it2++;
    }

    return true;
}

void relation_net::change_instance(relation_net& a2) {
    assert(a2.relation_node_output_size == this->relation_node_output_size);
    assert(a2.relation_node_growing_speed == this->relation_node_growing_speed);
    assert(a2.time_step_K == this->time_step_K);
    this->StateFluents = a2.StateFluents;
    this->NonFluents = a2.NonFluents;
    this->ActionFluents = a2.ActionFluents;
    this->Obj2types = a2.Obj2types;
    this->ig = a2.ig;
    this->stateVariableIndices_net = a2.stateVariableIndices_net;
    this->action_indices_ = a2.action_indices_;
    this->res_order = a2.res_order;
    this->Inputs = a2.Inputs;
    //
    // this->set_instance_tensor();
}

string relation_net::vec_2_str(vector<string> v1) {
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

vector<int> relation_net::count_occurence(string obj_type, vector<string> v1) {
    vector<int> res;
    for (size_t i = 0; i < v1.size(); i++) {
        if (v1[i] == obj_type) {
            res.push_back(i);
        }
    }
    return res;
}

bool relation_net::vec_is_intersect(vector<string> v1, vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        for (size_t i2 = 0; i2 < v2.size(); i2++) {
            if (v1[i] == v2[i2]) {
                return true;
            }
        }
    }
    return false;
}

bool relation_net::v2_contains_v1(vector<string> v1, vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        bool res2 = false;
        for (size_t j = 0; j < v2.size(); j++) {
            if (v1[i] == v2[j]) {
                res2 = true;
            };
        };
        if (!res2) {
            return false;
        };
    }
    return true;
}

string relation_net::erasechar(string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}

bool relation_net::v1_contains_diff_from_v2(vector<string> v1,
                                            vector<string> v2) {
    for (size_t i = 0; i < v1.size(); i++) {
        bool r = false;
        for (size_t j = 0; j < v2.size(); j++) {
            if (v1[i] == v2[j]) {
                r = true;
            }
        }
        if (!r) {
            return true;
        }
    }
    return false;
}

vector<double> relation_net::tensor2vector_z(torch::Tensor in) {
    in = torch::softmax(in, 1);
    // cout << "softmax:" << in << endl;
    std::vector<double> res;
    for (size_t i = 0; i < in.size(1); i++) {
        res.push_back((double)in[0][i].item<double>());
        // std::cout << "tensor2vector_z"
        //           << "size:" << std::to_string(in.size(1))
        //           << (double)in[0][i].item<double>() << std::endl;
    }
    return res;
}

// void RDDLTask::Net2write_helper(
//     std::string name, std::string fullname, double value,
//     std::string variable_type, std::vector<std::string> patameters_types,
//     std::vector<std::string> instantiated_patameters) {
//     std::ifstream infile("Net2_information");
//     std::ofstream taskFile;
//     std::string taskFileName = "Net2_information";
//     if (!infile.good()) {
//         taskFile.open(taskFileName.c_str());
//     } else {
//         taskFile.open(taskFileName.c_str(), std::ofstream::app);
//     }
//     taskFile << "DIV" << std::endl;
//     taskFile << "InstantiatedVariable" << std::endl;
//     taskFile << "fullname:" << std::endl;
//     taskFile << fullname << std::endl;
//     taskFile << "value" << std::endl;
//     taskFile << value << std::endl;
//     taskFile << "Name:" << std::endl;
//     taskFile << (name) << std::endl;
//     taskFile << "type:" << std::endl;
//     taskFile << variable_type << std::endl;
//     taskFile << "patameters_types:" << std::endl;
//     for (int i = 0; i < patameters_types.size(); i++) {
//         taskFile << (patameters_types)[i] << std::endl;
//     }
//     taskFile << "end" << std::endl;

//     taskFile << "instantiated_patameters:" << std::endl;
//     for (int i = 0; i < instantiated_patameters.size(); i++) {
//         taskFile << (instantiated_patameters)[i] << std::endl;
//     }
//     taskFile << "end" << std::endl;

//     taskFile.close();
// }
