#include "nn_module.h"
#include "assert.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <torch/torch.h>
using namespace std;
nn_module::nn_module(string name, vector<int>& topology,
                     bool Use_One_D_attention)
    : name(name), topology(topology), Use_One_D_attention(Use_One_D_attention) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        torch::Tensor tmp_weights =
            torch::empty({topology[i], topology[i + 1]}, options);
        tmp_weights = torch::nn::init::kaiming_normal_(
            tmp_weights, 0.01, torch::kFanOut, torch::kLeakyReLU);

        tmp_weights.requires_grad_(true);
        tmp_weights = register_parameter(name + ",weight:" + std::to_string(i),
                                         tmp_weights);
        this->layer_weights_sub.push_back(tmp_weights);

        // cout<< "qqqqq:"<< to_string(topology[i])<<"," <<to_string(topology[i + 1])<<endl;
        // cout<<  tmp_weights <<endl;
        // cout<<"wwwwwwwwww"<<endl;

        torch::Tensor tmp_bias = torch::randn({topology[i + 1]}, options);
        tmp_bias =
            register_parameter(name + ",bias:" + std::to_string(i), tmp_bias);
        this->layer_biases_sub.push_back(tmp_bias);
    }
    if (Use_One_D_attention) {
        this->One_D_attention = torch::empty({1, topology[0]}, options);
        this->One_D_attention =
            torch::nn::init::uniform_(this->One_D_attention, 0.999999, 1.000001);

        this->One_D_attention.requires_grad_(true);
        this->One_D_attention =
            register_parameter(name + ",oda", this->One_D_attention);

        //     cout<<   this->One_D_attention <<endl;
        //  cout<<"wwwwwwwwww"<<endl;
    }

    //
    // torch::Tensor X = torch::empty({3, 4}, options);
    // X = torch::nn::init::uniform_(X, 2, 5);
    // torch::Tensor a = torch::full({1, 4}, 10, options);
    // torch::Tensor Y = torch::mul(a, X);
    // cout << "X" << X << endl;
    // cout << "A" << a << endl;
    // cout << "Y" << Y << endl;
}

void nn_module::save_submodel(string path, string nn_module_name) {
    std::ofstream taskFile;
    string taskFileName = path + nn_module_name;
    cout << taskFileName << endl;
    taskFile.open(taskFileName.c_str());
    taskFile << "name" << endl;
    taskFile << this->name << endl;
    taskFile << "topologySize" << endl;
    taskFile << this->topology.size() << endl;
    taskFile << "Use_One_D_attention" << endl;
    taskFile << (this->Use_One_D_attention ? "true" : "false") << endl;
    taskFile.close();
    // save params
    for (int i = 0; i < this->layer_weights_sub.size(); i++) {
        torch::save((this->layer_weights_sub)[i],
                    path + name + "weight" + std::to_string(i));
    }
    for (int i = 0; i < this->layer_biases_sub.size(); i++) {
        torch::save((this->layer_biases_sub)[i],
                    path + name + "bias" + std::to_string(i));
    }

    if (this->Use_One_D_attention) {
        torch::save(this->One_D_attention, path + name + ",oda");
    }
}

void nn_module::Load_submodel(string path, string nn_module_name) {
    int topologySize;
    std::ifstream taskFile;
    string taskFileName = path + nn_module_name;

    taskFile.open(taskFileName.c_str());
    if (!taskFile.good()) {
        SystemUtils::abort("No_file nn module, name:"+taskFileName);
    }
    if (taskFile.is_open()) {
        std::string line;
        getline(taskFile, line);
        assert(line == "name");
        getline(taskFile, line);
        this->name = line;
        getline(taskFile, line);
        assert(line == "topologySize");
        getline(taskFile, line);
        topologySize = stoi(line);

        getline(taskFile, line);
        assert(line == "Use_One_D_attention");
        getline(taskFile, line);
        if (line == "true") {
            this->Use_One_D_attention = true;
        } else {
            assert(line == "false");
            this->Use_One_D_attention = false;
        }

        taskFile.close();
    }

    // read Params
    assert(this->layer_weights_sub.size() == 0);
    assert(this->layer_biases_sub.size() == 0);
    for (int i = 0; i < topologySize - 1; i++) {
        torch::Tensor weights;
        torch::load(weights, path + name + "weight" + std::to_string(i));
        weights = weights.requires_grad_(true);
        weights =
            register_parameter(name + ",weight:" + std::to_string(i), weights);

        this->layer_weights_sub.push_back(weights);

        torch::Tensor bias;
        torch::load(bias, path + name + "bias" + std::to_string(i));
        bias = bias.requires_grad_(true);
        bias = register_parameter(name + ",bias:" + std::to_string(i), bias);

        this->layer_biases_sub.push_back(bias);
    }

    if (this->Use_One_D_attention) {
        torch::Tensor t;
        torch::load(t, path + name + ",oda");
        this->One_D_attention = t;
        this->One_D_attention = this->One_D_attention.requires_grad_(true);
        this->One_D_attention =
            register_parameter(name + ",oda", this->One_D_attention);
    }
}

torch::Tensor nn_module::forward(torch::Tensor X_input) {
//    cout<<"x:"<<X_input<<endl;
//     if (this->Use_One_D_attention) {
//         X_input = torch::mul(this->One_D_attention, X_input);
//     }
// cout<<"x22222222:"<<X_input<<endl;
    for (unsigned int i = 0; i < this->layer_weights_sub.size(); ++i) {
        X_input =
            torch::leaky_relu(torch::addmm(this->layer_biases_sub[i], X_input,
                                           (this->layer_weights_sub)[i]),
                             0.01);
        // X_input = torch::elu(torch::addmm(this->layer_biases_sub[i], X_input,
        //                                   (this->layer_weights_sub)[i]));
    }
    return X_input;
}