#include "NNselection_two.h"
#include "prost_planner.h"
#include "utils/math_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
using namespace std;

/******************************************************************
                     Search Engine Creation
******************************************************************/

NNselection_two ::NNselection_two()
    : ProbabilisticSearchEngine("NNselection2") {}

NNselection_two ::NNselection_two(relation_net* select_nn)
    : ProbabilisticSearchEngine("NNselection2") {
    this->select_nn = select_nn;
}

std::string NNselection_two ::erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}

bool NNselection_two ::setValueFromString(std::string& param,
                                          std::string& value) {
    // if (param == "-f") {
    //  this->Parameter_folder=value;
    //  this->Parameter_folder+="/finalResult";

    //  cout<< this->Parameter_folder<<endl;
    // return true;
    // }
    return true;
}

/******************************************************************
                       Main Search Functions
******************************************************************/
void NNselection_two ::estimateBestActions(State const& _rootState,
                                           std::vector<int>& bestActions) {
    // std::vector<int> actionsToExpand = getApplicableActions(_rootState);
    std::vector<double> predicted =
        this->select_nn->tensor2vector_z((this->select_nn)->forward());
    double maxi = 0;

    for (unsigned int j = 0; j < predicted.size(); ++j) {
        if (predicted[j] > maxi) {
            bestActions.clear();
            bestActions.push_back(j);
            maxi = predicted[j];
        } else if (predicted[j] == maxi) {
            bestActions.push_back(j);
        }
    }

    // cout<<"s)========="<<endl;
    // cout << "roots" << _rootState.toCompactString() << endl;
    // cout << "stat";
    // for (int i = 0; i < this->select_nn->state_net->size(); i++) {
    //     cout << to_string((*this->select_nn->state_net)[i]) << ", ";
    // }
    // cout << endl;

    // if (bestActions.size() == 0) {
    //     std::vector<int> candidate;
    //     for (unsigned int j = 0; j < actionsToExpand.size(); ++j) {
    //         if (actionsToExpand[j] == j) {
    //             candidate.push_back(j);
    //         }
    //     }
    //     int randint = rand() % (candidate.size());

    //     bestActions.push_back(candidate[randint]);
    // }

    // cout << "cpredicted vec with size " << to_string(bestActions.size()) <<
    // ": "
    //      << to_string(bestActions[0]) << endl;
    ;
    for (int i = 0; i < predicted.size(); i++) {
        cout << to_string(predicted[i]) << ", ";
    }
    cout << endl;

    // std::cout<< "ss"<<*(this->select_nn->state_net)<<std::endl;
    // std::cout << (this->select_nn)->forward() << std::endl;
}

void NNselection_two ::printConfig(std::string indent) const {
    SearchEngine::printConfig(indent);
    indent += "  ";
}
