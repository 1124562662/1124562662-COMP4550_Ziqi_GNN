#ifndef RELATIONNET_H
#define RELATIONNET_H

#include "Domian_graph.h"
#include "Edges.h"
#include "Instance_graph.h"
#include "Nodes.h"
#include "Variables.h"
#include "nn_module.h"

#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <cassert>
#include <iostream>
#include <set>
#include <torch/torch.h>
using namespace std;
class AbstractVariable;

class relation_net : torch::nn::Module {
public:
    relation_net(){};
    relation_net(string ReadModulesParams_Path,
                 map<string, int> stateVariableIndices_net2,
                 map<string, int> action_indices_2)
        : ReadModulesParams_Path(ReadModulesParams_Path) {
        map<string, int>::iterator it;
        for (it = stateVariableIndices_net2.begin();
             it != stateVariableIndices_net2.end(); it++) {
            this->stateVariableIndices_net[this->erasechar(it->first)] =
                it->second;
        };
        for (it = action_indices_2.begin(); it != action_indices_2.end();
             it++) {
            this->action_indices_[this->erasechar(it->first)] = it->second;
        };

        this->ReadModulesParams = true;
        remove_file = true;
        this->readHelper(this->filename, &(this->Obj2types),
                         &(this->ALL_patameters_types));
        this->build_domain_graph();

        this->read_modules(ReadModulesParams_Path);
        // this->dg.Set_sizes(this->relation_node_output_size,
        //                    this->relation_node_growing_speed,
        //                    this->time_step_K);

        this->ig = InstanceGraph(this->dg, this->StateFluents, this->NonFluents,
                                 this->ActionFluents, this->Obj2types);

        this->set_instance_tensor();
    };

    // for testing
    relation_net(int relation_node_output_size,
                 double relation_node_growing_speed, int time_step_K,
                 map<string, int> stateVariableIndices_net2,
                 map<string, int> action_indices_2, bool remove_file)
        : relation_node_output_size(relation_node_output_size),
          relation_node_growing_speed(relation_node_growing_speed),
          time_step_K(time_step_K),
          remove_file(remove_file) {
        map<string, int>::iterator it;
        for (it = stateVariableIndices_net2.begin();
             it != stateVariableIndices_net2.end(); it++) {
            this->stateVariableIndices_net[this->erasechar(it->first)] =
                it->second;
        };
        for (it = action_indices_2.begin(); it != action_indices_2.end();
             it++) {
            this->action_indices_[this->erasechar(it->first)] = it->second;
        };

        this->readHelper(this->filename, &(this->Obj2types),
                         &(this->ALL_patameters_types));
        this->build_domain_graph();
        this->dg.Set_sizes(relation_node_output_size,
                           relation_node_growing_speed, time_step_K, "CONCATE");
        this->set_modules();
        this->ig = InstanceGraph(this->dg, this->StateFluents, this->NonFluents,
                                 this->ActionFluents, this->Obj2types);

        if (!remove_file) {
            save_indices();
        }

        this->set_instance_tensor();
    };

    // training only
    relation_net(int relation_node_output_size,
                 double relation_node_growing_speed, int time_step_K,
                 bool ReadModulesParams, string ReadModulesParams_Path,
                 bool remove_file)
        : relation_node_output_size(relation_node_output_size),
          relation_node_growing_speed(relation_node_growing_speed),
          time_step_K(time_step_K),
          ReadModulesParams(ReadModulesParams),
          remove_file(remove_file) {
        this->ReadModulesParams_Path = ReadModulesParams_Path;
        read_indices(this->ReadModulesParams_Path);
        this->readHelper(ReadModulesParams_Path + this->filename,
                         &(this->Obj2types), &(this->ALL_patameters_types));

        this->build_domain_graph();

        if (!ReadModulesParams) {
            this->dg.Set_sizes(relation_node_output_size,
                               relation_node_growing_speed, time_step_K,
                               "CONCATE");
            this->set_modules();

        } else {
            this->read_modules(ReadModulesParams_Path);
            this->dg.Set_sizes(this->relation_node_output_size,
                               this->relation_node_growing_speed,
                               this->time_step_K, "CONCATE");
        };
        this->ig = InstanceGraph(this->dg, this->StateFluents, this->NonFluents,
                                 this->ActionFluents, this->Obj2types);

        this->set_instance_tensor();
    };

    void set_modules();
    void save_modules(string path);
    void read_modules(string path);

    static void trainnet_();

    torch::Tensor forward(vector<double> const& current_state_);
    torch::Tensor forward();
     torch::Tensor forward(torch::Tensor X);
    map<string, torch::Tensor> forward_sogbofa(
        map<string, torch::Tensor>& state_fluents_map);
    //------(1)domian graph------

    struct classcomp {
        bool operator()(const AbstractVariable& lhs,
                        const AbstractVariable& rhs) const {
            return lhs.name < rhs.name;
        }
    };
    set<AbstractVariable, classcomp> StateFluentsSchema;
    set<AbstractVariable, classcomp> NonFluentsSchema;
    set<AbstractVariable, classcomp> knowledgeFluentsSchema;
    set<AbstractVariable, classcomp> ActionFluentsSchema;

    set<string> ALL_patameters_types;
    domian_graph dg;

    static bool check_domain_schema(
        set<AbstractVariable, classcomp> const& StateFluentsSchema1,
        set<AbstractVariable, classcomp> const& NonFluentsSchema1,
        set<AbstractVariable, classcomp> const& ActionFluentsSchema1,
        set<AbstractVariable, classcomp> const& StateFluentsSchema2,
        set<AbstractVariable, classcomp> const& NonFluentsSchema2,
        set<AbstractVariable, classcomp> const& ActionFluentsSchema2);

    void build_domain_graph();

    // ------(3)instance------
    vector<InstantiatedVariable> StateFluents;
    vector<InstantiatedVariable> NonFluents;
    vector<InstantiatedVariable> ActionFluents;
    map<string, string> Obj2types;
    InstanceGraph ig;
    map<string, int> stateVariableIndices_net;
    map<string, int> action_indices_;
    vector<string> res_order;

    // used in the forward

    map<string, torch::Tensor> Inputs;
    void set_instance_tensor();

    void change_instance(relation_net& a2);

    // ------(4) record index ------
    void set_state_pointer(vector<double>* stateVec) {
        this->state_net = stateVec;
    };
    void save_indices();
    void read_indices(string path);

    // -----------------(2)  modules----------------

    // time-step -> RelationNode type -> module
    map<int, map<string, nn_module>> RelationNodeModules;

    // time-step -> ObjectNode type -> neighbor edge type -> module
    map<int, map<string, map<string, nn_module>>> ObjectNodeModules;

    // action fluent name(including no-op)-> module
    map<string, nn_module> actions_m;

    void UpdateGradients(int& currentRound, torch::Tensor X, torch::Tensor Y);
    static void trainnet_sogbofa();
    vector<double> tensor2vector_z(torch::Tensor in);
    vector<double>* state_net;

    int relation_node_output_size;
    double relation_node_growing_speed;
    int time_step_K;

    bool ReadModulesParams;
    string ReadModulesParams_Path;
    string filename = "Net2_information";
    bool remove_file;

    double helper_double;

private:
    void readHelper(string filepath, map<string, string>* Obj2types,
                    set<string>* ALL_patameters_types);
    string vec_2_str(vector<string> v1);
    vector<int> count_occurence(string obj_type, vector<string> v1);
    static bool vec_is_intersect(vector<string> v1, vector<string> v2);
    static bool v2_contains_v1(vector<string> v1, vector<string> v2);
    static bool v1_contains_diff_from_v2(vector<string> v1, vector<string> v2);
    string erasechar(string str);
    map<string, torch::Tensor> forward_helper(int batch_size);
};

#endif
