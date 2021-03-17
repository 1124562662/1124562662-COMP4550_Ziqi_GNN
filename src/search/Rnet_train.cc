
#include "RelationNet.h"
#include "sogbofa.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <torch/torch.h>

using namespace std;

void relation_net::UpdateGradients(int& currentRound, torch::Tensor X,
                                   torch::Tensor Y) {
    // if (X.size() != Y.size()) {
    //     SystemUtils::abort("sizemismatch_______ug:" + to_string(X.size()) +
    //                        "," + to_string(Y.size()));
    // }

    double learningR = 0.01;

    torch::optim::AdamOptions op =
        torch::optim::AdamOptions().weight_decay(0.001).lr(learningR);

    torch::optim::Adam optimizer(this->parameters(), op);
    int iteration = 8000;
    //  int batch_size = 40;
    double Lrate = 0.95;

    // if (currentRound < 20) {
    //     iteration = 800;
    // }

    for (size_t iter = 0; iter < iteration; iter++) {
        optimizer.zero_grad();

        if ( iteration % 100 == 49) {
            /// iteration += (int)((currentRound % 50) * );
            for (auto& group : optimizer.param_groups()) {
                if (group.has_options()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(
                        group.options());
                    options.lr(options.lr() * Lrate);
                }
            }
        };

        // torch::Tensor pred;
        // torch::Tensor labs;
        // bool set = false;
        // for (size_t i = 0; i < batch_size; i++) {
        // int rndi = rand() % (X.size(0));

        torch::Tensor prediction = this->forward(X);
        // cout << "pred:::" << prediction << endl;
        if (at::isnan(prediction).any().item<bool>()) {
            std::cout << "nan" << to_string(iter) << endl;
            continue;
        }

        // pred = (set) ? torch::cat({pred, prediction}, 0) : prediction;

        // torch::Tensor label_tor =
        //     torch::full({1}, (int)(Y[rndi]),
        //                 torch::TensorOptions().dtype(torch::kLong));

        // cout << "swxsdxcsd" << label_tor << endl;

        //  labs = (set) ? torch::cat({labs, label_tor}, 0) : label_tor;
        //  set = true;
        // }

        torch::Tensor loss_ =
            torch::nn::functional::cross_entropy(prediction, Y);
        std::cout << "loss_" << to_string(currentRound) << ",  " << loss_
                  << endl;
        loss_.backward();
        optimizer.step();
    }
}

void relation_net::trainnet_sogbofa() {
    cout << "sogbofa training starts............." << endl;

    const bool ReadModulesParams = false;

    const int horizon = 40;

    const int iter_num = 2000;

    const int start_point = 2;

    const int num_instances = 1;

    int batch_size = 20;

    const int relation_node_output_size = 3;

    const double relation_node_growing_speed = 2;

    const double learning_rate = 0.009;

    const double regulazation = 0.00000000001;

    const int time_step_K = 3;

    const double LRrate = 0.95;

    const string domainFolder = "elevator/";

    relation_net trainnet = relation_net(
        relation_node_output_size, relation_node_growing_speed, time_step_K,
        ReadModulesParams,
        domainFolder + "instance" + to_string(start_point) + "/", false);

    map<int, relation_net> modules;
    map<int, sogbofa> sogbofas;
    map<int, vector<vector<double>>> data_s;

    for (int y = start_point; y < num_instances + start_point; y++) {
        string folder = domainFolder + "instance" + std::to_string(y) + "/";

        modules[y] =
            relation_net(relation_node_output_size, relation_node_growing_speed,
                         time_step_K, ReadModulesParams, folder, false);

        if (y > start_point) {
            relation_net::check_domain_schema(
                modules[y].StateFluentsSchema, modules[y].NonFluentsSchema,
                modules[y].ActionFluentsSchema,
                modules[y - 1].StateFluentsSchema,
                modules[y - 1].NonFluentsSchema,
                modules[y - 1].ActionFluentsSchema);
        }

        sogbofas[y] = sogbofa(folder, &trainnet);

        string line;
        vector<vector<double>> data;

        map<string, string> duplicate_check;
        ifstream myfile(folder + "data.txt");
        int i = 0;
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                if (i % 2 == 0) {
                    vector<double> data_point;

                    string key = "";
                    for (unsigned int i = 0; i < line.length(); i++) {
                        char c = line[i];
                        int tmp = (int)c - 48;

                        if (tmp > 0) {
                            data_point.push_back(1.0);
                            key += "1";
                        } else {
                            data_point.push_back(0.0);
                            key += "0";
                        }
                    }

                    if (duplicate_check[key] != "in") {
                        data.push_back(data_point);
                        duplicate_check[key] = "in";
                    }
                }
                i = i + 1;
            }
            myfile.close();
        }
        if (data.size() == 0) {
            SystemUtils::abort("no training data in :" + to_string(y));
        }

        data_s[y] = data;
    }

    torch::optim::AdamOptions op = torch::optim::AdamOptions()
                                       .weight_decay(regulazation)
                                       .lr(
                                           /*lr=*/learning_rate);

    torch::optim::Adam optimizer(trainnet.parameters(), op);

    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    map<int, map<int, double>> normalizer;

    for (int iter = 0; iter < iter_num; iter++) {
        if (iter == 0) {
            for (int y = start_point; y < num_instances + start_point; y++) {
                trainnet.change_instance(modules[y]);

                for (size_t i = 0; i < data_s[y].size(); i++) {
                    map<string, torch::Tensor> initial_state =
                        modules[y].Inputs;
                    vector<double> data_point = data_s[y][i];
                    auto options = torch::TensorOptions()
                                       .dtype(torch::kDouble)
                                       .requires_grad(true);

                    map<string, int>::iterator it;
                    for (it = modules[y].stateVariableIndices_net.begin();
                         it != modules[y].stateVariableIndices_net.end();
                         it++) {
                        torch::Tensor tensor_ = torch::full(
                            {1, 1}, data_point[it->second], options);
                        initial_state[it->first] = tensor_;
                    };

                    torch::Tensor maxi = sogbofas[y].guided_aggregated_seach(
                        initial_state, horizon);

                    normalizer[y][i] = (double)maxi[0][0].item<double>();
                }
            }
        } else {
            optimizer.zero_grad();

            if (iter % 180 == 9) {
                for (auto& group : optimizer.param_groups()) {
                    if (group.has_options()) {
                        auto& options = static_cast<torch::optim::AdamOptions&>(
                            group.options());
                        options.lr(options.lr() * (LRrate));
                    }
                }
                if (batch_size < 30) {
                    batch_size += 3;
                }
            };

            torch::Tensor objf = torch::full({1, 1}, 0.0, options);
            for (int y = start_point; y < num_instances + start_point; y++) {
                trainnet.change_instance(modules[y]);

                for (size_t batch = 0; batch < batch_size; batch++) {
                    map<string, torch::Tensor> initial_state =
                        modules[y].Inputs;

                    int rndi = rand() % (data_s[y].size());
                    vector<double> data_point = data_s[y][rndi];

                    auto options = torch::TensorOptions()
                                       .dtype(torch::kDouble)
                                       .requires_grad(true);

                    map<string, int>::iterator it;
                    for (it = modules[y].stateVariableIndices_net.begin();
                         it != modules[y].stateVariableIndices_net.end();
                         it++) {
                        torch::Tensor tensor_ = torch::full(
                            {1, 1}, data_point[it->second], options);
                        initial_state[it->first] = tensor_;
                    };

                    torch::Tensor maxi = sogbofas[y].guided_aggregated_seach(
                        initial_state, horizon);

                    double new_maxi = maxi.item<double>();

                    maxi = torch::div(
                        torch::sub(
                            torch::full({1, 1},
                                        (double)(0.1 * normalizer[y][rndi]),
                                        options),
                            maxi),
                        torch::full({1, 1},
                                    (double)(abs(normalizer[y][rndi]) + 0.01),
                                    options));

                    cout << to_string(new_maxi) << "," << normalizer[y][rndi]
                         << endl;

                    normalizer[y][rndi] =
                        max((double)new_maxi, normalizer[y][rndi]);

                    cout << normalizer[y][rndi] << endl;

                    cout << "maxi  ,  " << maxi << endl;

                    objf = torch::add(objf, maxi);

                    if (batch == 0) {
                        map<string, torch::Tensor>::iterator pp;
                        for (pp = initial_state.begin();
                             pp != initial_state.end(); pp++) {
                            cout << pp->first << "," << pp->second << endl;
                        }

                        std::cout << "+++++++++++++++++++++" << endl;

                        // cout <<
                        // "=============================================="
                        //         "========"
                        //         "======"
                        //      << endl;

                        // for (const auto& pair : trainnet.named_parameters())
                        // {
                        //     std::cout << pair.key() << ": " << endl;
                        //     cout << pair.value() << std::endl;
                        // }
                        // cout <<
                        // "=============================================="
                        //         "========"
                        //         "======"
                        //      << endl;
                    }
                }
            }
            objf = torch::div(
                objf,
                torch::full(
                    {1, 1},
                    (double)(0.0001 *
                             (num_instances * batch_size * horizon + 1.0)),
                    options));

            cout << "" << endl;
            cout << "" << endl;
            cout << "objf" << to_string(iter) << ",  " << objf << endl;
            cout << "" << endl;

            objf.backward();
            // torch::nn::utils::clip_grad_norm_(trainnet.parameters(), 10.0, 1.0);

            // for (const auto& pair : trainnet.named_parameters()) {
            //     torch::Tensor t =
            //         torch::nn::functional::normalize(pair.value().grad());
            //     trainnet.named_parameters()[pair.key()].grad_fn()= t;
            // }

            std::ofstream taskFile;
            std::string taskFileName = "tttttt";
            taskFile.open(taskFileName.c_str());

            taskFile << "grad" << endl;
            taskFile << "grad" << endl;
            for (const auto& pair : trainnet.named_parameters()) {
                taskFile << pair.key() << ": " << endl;
                taskFile << pair.value() << std::endl;
            }
            taskFile << "grad-end" << endl;

            optimizer.step();
        }
    }
    trainnet.save_modules(domainFolder + "finalResult/");
}