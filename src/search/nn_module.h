#ifndef nn_module_H
#define nn_module_H

#include "Variables.h"
#include <torch/torch.h>
using namespace std;

class nn_module : public torch::nn::Module {
public:
    nn_module(){};
    nn_module(string name, vector<int>& topology, bool Use_One_D_attention);

    // Not used
   // nn_module(string name, vector<int>& topology){};

    torch::Tensor forward(torch::Tensor X_input);

    void save_submodel(string path, string nn_module_name);
    void Load_submodel(string path, string nn_module_name);

    vector<torch::Tensor> layer_weights_sub;
    vector<torch::Tensor> layer_biases_sub;
    torch::Tensor One_D_attention;

    string name;
    vector<int> topology;
    bool Use_One_D_attention;
};
#endif
