#include "Net.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <set>
#include <string>

using namespace std;

void Net::trainnet_() {
    cout << "training starts............." << endl;

    const int iter_num = 1;

    const int inner_iter_num = 100000;
    const int start_point = 2;
    const int num_instances = 1;

    const int submodel_output_size = 3;

    const int batch_size_ = 320;

    const double learning_rate = 0.01;
    const double LRrate = 0.9;

    const std::string domainFolder = "elevator/";

    // insatnce number -> the net for that instance
    // To-do
    // std::map<int, Net> nets;

    std::map<int, std::map<std::string, sub_model>> torch_sub_models_share;

    for (int iter = 0; iter < iter_num; iter++) {
        for (int y = start_point; y < num_instances + start_point; y++) {
            cout << "training for iteration:" + to_string(iter) +
                        " ,instance:" + to_string(y)
                 << endl;

            string folder = domainFolder + "instance" + std::to_string(y) + "/";

            auto trainnet = std::make_shared<Net>();
            // 1 data containers
            std::map<std::string, double> NonfluentsToValues;
            trainnet->NonfluentsToValues_net = &NonfluentsToValues;
            // key : string of state fluents with parameters
            // value : a vector of fixed-size dependent state fluents with
            // parametetrs
            std::map<std::string, std::set<std::string>> sf2FixedSize_sfs;
            trainnet->sf2FixedSize_sfs_net = &sf2FixedSize_sfs;

            // key : string of state fluents with parameters -> quantifiers
            // -> vector of parameters and dependent state fluents
            std::map<std::string, std::map<std::string, std::set<std::string>>>
                sf2VariedSize_sfs;
            trainnet->sf2VariedSize_sfs_net = &sf2VariedSize_sfs;

            // 2 read parameters
            std::ifstream taskFile2;
            taskFile2.open(folder + "non-fluents-file");
            if (taskFile2.is_open()) {
                std::string line;
                while (getline(taskFile2, line)) {
                    std::string name_ = erasechar(line);
                    getline(taskFile2, line);
                    std::string value_ = line;
                    double value2 = std::stod(value_);
                    NonfluentsToValues[name_] = value2;
                    // std::cout<<
                    // std::to_string((this->NonfluentsToValues)[name_])<<std::endl;
                    // std::cout<< name_<<std::endl;
                }
                taskFile2.close();
            }

            std::ifstream taskFile22;
            taskFile22.open(folder + "CDFs_file_");
            if (taskFile22.is_open()) {
                std::map<std::string, std::set<std::string>> tmp_quantifier_;
                std::string current_sf_name_;
                std::string line;
                while (getline(taskFile22, line)) {
                    if (line == "ExistentialQuantification:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(parameter_z);
                        getline(taskFile22, line);
                        while (line == "Existential-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["Existential"] = tmp_para_sfs; // Todo
                    } else if (line == "UniversalQuantification:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "Universal-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["Universal"] = tmp_para_sfs;

                    } else if (line == "Product:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "prod-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["product"] = tmp_para_sfs;

                    } else if (line == "Sumation:") {
                        std::set<std::string> tmp_para_sfs;
                        getline(taskFile22, line);
                        std::string parameter_z = erasechar(line);
                        tmp_para_sfs.insert(erasechar(line));
                        getline(taskFile22, line);
                        while (line == "sum-fluent_name:") {
                            getline(taskFile22, line);
                            tmp_para_sfs.insert(erasechar(line));
                            getline(taskFile22, line);
                        }
                        tmp_quantifier_["sum"] = tmp_para_sfs;
                    } else if (line == "The sf:") {
                        getline(taskFile22, line);
                        current_sf_name_ = erasechar(line);
                        sf2VariedSize_sfs[current_sf_name_] = tmp_quantifier_;
                        tmp_quantifier_.clear();
                        getline(taskFile22, line);
                        while (line == "obj:") {
                            std::vector<std::string> obj_and_type;
                            getline(taskFile22, line);
                            obj_and_type.push_back(erasechar(line));
                            getline(taskFile22, line);
                            obj_and_type.push_back(erasechar(line));
                            getline(taskFile22, line);
                        }
                    } else if (line == "fix-fluent_name:") {
                        getline(taskFile22, line);
                        if (sf2FixedSize_sfs.count(
                                erasechar(current_sf_name_)) == 0) {
                            std::set<std::string> tmp_sfs;
                            tmp_sfs.insert(erasechar(line));
                            sf2FixedSize_sfs[current_sf_name_] = tmp_sfs;
                        } else {
                            sf2FixedSize_sfs[current_sf_name_].insert(
                                erasechar(line));
                        }
                    }
                }
                taskFile22.close();
            }
            std::map<std::string, int> action_indices_tmp;
            trainnet->action_indices_ = &(action_indices_tmp);
            std::map<std::string, int> stateVariableIndices_tmp;
            trainnet->stateVariableIndices_net = &(stateVariableIndices_tmp);
            std::vector<std::vector<std::string>> stateVariableValues_tmp;
            trainnet->stateVariableValues_net = &(stateVariableValues_tmp);

            trainnet->initialize_(true, folder, submodel_output_size);

            // initialize
            if (iter == 0 && y == start_point) {
                torch_sub_models_share = trainnet->torch_sub_models;
            } else {
                trainnet->torch_sub_models = torch_sub_models_share;
            }

            // construct the input

            string line;
            vector<vector<double>> data;
            vector<double> label;
            // vector<vector<double>> label;
            // int out_size=5;

            ifstream myfile(folder + "data.txt");
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
                        data.push_back(data_point);
                        data_point.clear();

                    } else {
                        // vector<double> label_point(out_size, 0);
                        // label_point[std::stoi(line)] = 1.0;

                        // label.push_back(label_point);
                        // label_point.clear();

                        label.push_back((double)std::stoi(line));
                    }
                    i = i + 1;
                }
                myfile.close();
            }
            assert(label.size() == data.size());

            // inner interation
            //   auto testm = std::make_shared<testmodel>();

            // auto options = torch::TensorOptions()
            //                    .dtype(torch::kDouble)
            //                    .requires_grad(true);

            // vector<int> tetstopo;
            // tetstopo.push_back(10);
            // tetstopo.push_back(7);
            // tetstopo.push_back(3);
            // testm.initialize(tetstopo);
            // cout << "weights" << endl;
            // // cout << testm->thist<< endl;
            // cout <<
            // ((trainnet->torch_sub_models)[1]["elevator-at-floor"])
            //             .layer_biases_sub[0]
            //      << endl;

            cout << "initialize finished,training starts----------------"
                 << endl;
            //   int action_numbers = trainnet->action_f_net.size() + 1;

            auto net = std::make_shared<Net::Nett>();
            torch::optim::AdamOptions op = torch::optim::AdamOptions()
                                               .weight_decay(0.000000001)
                                               .lr(
                                                   /*lr=*/learning_rate);

            torch::optim::Adam optimizer(net->parameters(), op);

            for (int epoch = 0; epoch < inner_iter_num; ++epoch) {
                optimizer.zero_grad();
                torch::Tensor pred;
                torch::Tensor labs;

                // torch::Tensor pred = torch::rand({1, action_numbers},
                // options); torch::Tensor labs = torch::full(
                //     {1}, (int)0, torch::TensorOptions().dtype(torch::kLong));

                // torch::Tensor labs = torch::full({1,out_size}, 0,
                // options);

                //   cout << to_string(label[rndi].size()) << endl;
                // torch::Tensor in = torch::rand({2000, 10}, options );
                if (inner_iter_num % 2000 == 1) {
                    for (auto& group : optimizer.param_groups()) {
                        if (group.has_options()) {
                            auto& options =
                                static_cast<torch::optim::AdamOptions&>(
                                    group.options());
                            options.lr(options.lr() * (LRrate));
                        }
                    }
                }
                bool set = false;
                for (int batchi = 0; batchi < batch_size_; batchi++) {
                    int rndi = rand() % (data.size());

                    torch::Tensor prediction = net->forward(data[rndi]);
                    if (batchi % 100 == 1) {
                        cout << "prediction" << prediction << endl;
                        cout << "=================================" << endl;
                    }
                    //   cout << "prediction" << prediction << endl;
                    if (at::isnan(prediction).any().item<bool>()) {
                        cout << "nan" << to_string(batchi) << endl;
                        continue;
                    }
                    pred =
                        (set) ? torch::cat({pred, prediction}, 0) : prediction;

                    torch::Tensor label_tor =
                        torch::full({1}, (int)label[rndi],
                                    torch::TensorOptions().dtype(torch::kLong));

                    //  torch::Tensor label_tor=torch::full({1,out_size},
                    //  0);
                    //                     label_tor.requires_grad_(false);
                    //                     int counter = 0;
                    //                     for (int i : (label[rndi])) {
                    //                         label_tor[0][counter] =
                    //                         (double)i; counter++;
                    //                     }
                    //                     label_tor =
                    //                     label_tor.requires_grad_(true);

                    labs = (set) ? torch::cat({labs, label_tor}, 0) : label_tor;
                    set = true;
                }

                torch::Tensor loss =
                    torch::nn::functional::cross_entropy(pred, labs);
                //  torch::Tensor loss = torch::mse_loss(pred,
                //  labs.detach());

                cout << "loss" << to_string(epoch) << ",  " << loss << endl;
                loss.backward();
                optimizer.step();
                cout << "" << endl;
                cout << "===============" << endl;
                //     cout << trainnet->parameters() << endl;

                cout << "===============" << endl;
                cout << "" << endl;

                // cout << "weights" << endl;
                // cout << testm->thist << endl;
                // cout <<
                // ((trainnet->torch_sub_models)[1]["elevator-at-floor"])
                //             .layer_biases_sub[0]
                //      << endl;
            }

            data.clear();
            label.clear();
            // save parameters
            torch_sub_models_share = trainnet->torch_sub_models;
            // int u = 0;
            // while (true) {
            //     u = u + 1;
            // }
        }
    }

    // save_parameters
    std::map<int, std::map<std::string, sub_model>>::iterator it4;
    std::map<std::string, sub_model>::iterator it5;
    for (it4 = torch_sub_models_share.begin();
         it4 != torch_sub_models_share.end(); it4++) {
        for (it5 = (it4->second).begin(); it5 != (it4->second).end(); it5++) {
            string model_path = domainFolder + "finalResult/savedModels/" +
                                it5->first + "+" + std::to_string(it4->first) +
                                "+";
            torch_sub_models_share[it4->first][it5->first].save_submodel(
                model_path);
        }
    }
    //
}
