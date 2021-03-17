
#include "RelationNet.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <torch/torch.h>

using namespace std;

void relation_net::trainnet_() {
    cout << "training starts............." << endl;

    const int iter_num = 10000;

    const int start_point = 1;

    const int num_instances = 3;

    const int relation_node_output_size = 4;

    // const double relation_node_growing_speed = 2;
    const double relation_node_growing_speed = 2;

    const int time_step_K = 4;

    const double learning_rate = 0.01;

    const double weighht_decay_ = 0.00001;

    const double LRrate = 0.98;

    const string domainFolder = "game_of_life/";

    map<int, relation_net> modules;
    map<int, torch::Tensor> data_s;
    map<int, torch::Tensor> label_s;

    for (int y = start_point; y < num_instances + start_point; y++) {
        string folder = domainFolder + "instance" + std::to_string(y) + "/";

        relation_net tmp =
            relation_net(relation_node_output_size, relation_node_growing_speed,
                         time_step_K, false, folder, false);
        modules[y] = tmp;

        if (y > start_point) {
            relation_net::check_domain_schema(
                tmp.StateFluentsSchema, tmp.NonFluentsSchema,
                tmp.ActionFluentsSchema, modules[y - 1].StateFluentsSchema,
                modules[y - 1].NonFluentsSchema,
                modules[y - 1].ActionFluentsSchema);
        }

        string line;

        torch::Tensor X;
        torch::Tensor Y;
        auto opts =
            torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

        ifstream myfile(folder + "data.txt");

        // std::ofstream taskFile;
        // taskFile.open(folder + "data2.txt");
        // map<string, int> tmppp;
        // bool tmpbool = false;

        int i = 0;
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                if (i % 2 == 0) {
                    vector<double> data_point;
                    for (unsigned int i = 0; i < line.length(); i++) {
                        char c = line[i];
                        int tmp = (int)c - 48;
                        if (tmp > 0) {
                            data_point.push_back(1.0);
                        } else {
                            data_point.push_back(0);
                        }
                    }
                    torch::Tensor xx = torch::from_blob(
                        data_point.data(), {1, (long)data_point.size()}, opts);

                    // if (tmppp[line] != 2) {
                    //     tmppp[line] = 2;
                    //     taskFile << line << endl;
                    //     tmpbool = true;
                    // }

                    if (!X.defined()) {
                        X = xx;
                    } else {
                        X = torch::cat({X, xx}, 0);
                    }

                } else {
                    // if (tmpbool) {
                    //     tmpbool = false;
                    //     taskFile << line << endl;
                    // }

                    torch::Tensor yy =
                        torch::full({1}, (double)std::stoi(line),
                                    torch::TensorOptions().dtype(torch::kLong));
                    if (!Y.defined()) {
                        Y = yy;
                    } else {
                        Y = torch::cat({Y, yy}, 0);
                    }
                }
                i = i + 1;
            }
            myfile.close();
        }
        data_s[y] = X;
        label_s[y] = Y;
        
        // taskFile.close();
    };

    auto trainnet = std::make_shared<relation_net>(
        relation_node_output_size, relation_node_growing_speed, time_step_K,
        false, domainFolder + "instance" + to_string(start_point) + "/", false);

    // auto trainnet = std::make_shared<relation_net>(
    //     "E2/", trainnet2->stateVariableIndices_net,
    //     trainnet2->action_indices_);

    torch::optim::AdamOptions op = torch::optim::AdamOptions()
                                       .weight_decay(weighht_decay_)
                                       .lr(
                                           /*lr=*/learning_rate);

    torch::optim::Adam optimizer(trainnet->parameters(), op);

    for (int iter = 0; iter < iter_num; iter++) {
        optimizer.zero_grad();

        if (iter % 200 == 49) {
            for (auto& group : optimizer.param_groups()) {
                if (group.has_options()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(
                        group.options());
                    options.lr(options.lr() * (LRrate));
                }
            }
        };

        torch::Tensor loss;

        for (int y = start_point; y < num_instances + start_point; y++) {
            trainnet->change_instance(modules[y]);

            torch::Tensor prediction = trainnet->forward(data_s[y]);
            // cout << "pred:::" << prediction << endl;
            if (at::isnan(prediction).any().item<bool>()) {
                std::cout << "nan" << to_string(iter) << endl;
                continue;
            }

            torch::Tensor loss_ =
                torch::nn::functional::cross_entropy(prediction, label_s[y]);

            // cout << "loss_" << to_string(y) << ",  " << loss_ << endl;

            if (y == start_point) {
                loss = loss_;
            } else {
                loss = torch::add(loss, loss_);
            };
        }
        auto opts =
            torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

        loss =
            torch::div(loss, torch::full({1, 1}, (double)num_instances, opts));
        cout << "loss" << to_string(iter) << ",  " << loss << endl;
        cout << "" << endl;

        loss.backward();
        optimizer.step();
    }

    trainnet->save_modules(domainFolder + "finalResult/");
}

// for (int batchi = 0; batchi < batch_size_; batchi++) {
//     int rndi = rand() % (data_s[y].size());

//     torch::Tensor prediction =
//     trainnet->forward(data_s[y][rndi]);

//     if (at::isnan(prediction).any().item<bool>()) {
//         cout << "nan" << to_string(batchi) << endl;
//         continue;
//     }
//     if (batchi % 100 == 0) {
//         cout << "predict:" << prediction << endl;
//     }
//     pred = (set) ? torch::cat({pred, prediction}, 0) :
//     prediction;

//     torch::Tensor label_tor =
//         torch::full({1}, (int)label_s[y][rndi],
//                     torch::TensorOptions().dtype(torch::kLong));

//     labs = (set) ? torch::cat({labs, label_tor}, 0) : label_tor;
//     set = true;
// }

// pred = (set) ? torch::cat({pred, prediction}, 0) : prediction;

// torch::Tensor label_tor =
//     torch::full({1}, (int)(Y[rndi]),
//                 torch::TensorOptions().dtype(torch::kLong));

// cout << "swxsdxcsd" << label_tor << endl;

//  labs = (set) ? torch::cat({labs, label_tor}, 0) : label_tor;
//  set = true;
// }