#ifndef SOGBOFA_H
#define SOGBOFA_H

#include "RelationNet.h"
#include "parser.h"
#include "sogbofa_parser.h"
#include <torch/torch.h>

using namespace std;

class sogbofa : public torch::nn::Module {
public:
    sogbofa(){};

    sogbofa(string instance_path, relation_net* rn)
        : rn(rn), instance_path(instance_path) {
        this->sogbofa_parser = sog_Parser(instance_path);
    };

    torch::Tensor guided_aggregated_seach(
        map<string, torch::Tensor>& initial_state,int horizon);

    void compute_successor_belif_state(
        map<string, torch::Tensor>& state_fluent_input,
        map<string, torch::Tensor>& action_fluent_input,
        map<string, torch::Tensor>& successor_belif_state);

    torch::Tensor reward_for_one_state(
        map<string, torch::Tensor>& state_fluent_input,map<string, torch::Tensor>& action_fluent_input);

    relation_net* rn;
    sog_Parser sogbofa_parser;

private:
    string instance_path;
};
#endif
