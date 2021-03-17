#include "NNRollOut.h"
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

NNRollOut::NNRollOut()
    : ProbabilisticSearchEngine("NNRollOut"), numberOfIterations(1) {
    // std::string folder = "elevator/instance" + std::to_string(3) + "/";
    // auto net = std::make_shared<Net>();
}


std::string NNRollOut::erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}

bool NNRollOut::setValueFromString(std::string& param, std::string& value) {
    if (param == "-it") {
        setNumberOfIterations(atoi(value.c_str()));

        return true;
    }

    return SearchEngine::setValueFromString(param, value);
}

/******************************************************************
                       Main Search Functions
******************************************************************/

void NNRollOut::estimateQValue(State const& state, int actionIndex,
                               double& qValue) {
    assert(state.stepsToGo() > 0);
    PDState current(state);
    performNNWalks(current, actionIndex, qValue);
}

void NNRollOut::estimateQValues(State const& state,
                                std::vector<int> const& actionsToExpand,
                                std::vector<double>& qValues) {
    // assert(state.stepsToGo() > 0);
    // PDState current(state);
    // for (size_t index = 0; index < qValues.size(); ++index) {
    //     if (actionsToExpand[index] == index) {
    //         performNNWalks(current, index, qValues[index]);
    //     }
    // }
}
void NNRollOut::performNNWalks(PDState const& root, int firstActionIndex,
                               double& result) {
    // result = 0.0;
    // double reward = 0.0;
    // PDState next;

    // for (unsigned int i = 0; i < numberOfIterations; ++i) {
    //     PDState current(root.stepsToGo() - 1);
    //     sampleSuccessorState(root, firstActionIndex, current, reward);
    //     result += reward;

    //     while (current.stepsToGo() > 0) {
    //         std::vector<int> applicable = getApplicableActions(current);
    //         cout << "->" << current.toCompactString() << endl;
    //         std::vector<double> predicted = a_net->tensor2vector_z(
    //             (this->a_net)->forward(current.ReadState2Vector()));
    //         double rnd = ((double)rand()) / RAND_MAX;
    //         cout<<'l'<<to_string(rnd)<<endl;
    //         int ActionIndex = 0;
    //         double tmp = 0;
    //         while (ActionIndex < predicted.size()) {
    //             double tmp_last = tmp;
    //             tmp += predicted[ActionIndex];
    //             if (rnd >= tmp_last && rnd <= tmp) {
    //                 break;
    //             } else {
    //                 ActionIndex += 1;
    //             }
    //         }

    //         double maxi = 0;
    //         for (unsigned int j = 0; j < predicted.size(); ++j) {
    //             if (predicted[j] >= maxi && j == applicable[j]) {
    //                 ActionIndex = j;
    //             }
    //         }
    //         cout << "a:" << std::to_string(ActionIndex) << endl;
    //         next.reset(current.stepsToGo() - 1);
    //         sampleSuccessorState(current, ActionIndex, next, reward);
    //         result += reward;
    //         current = next;
    //     }
    // }

    // result /= (double)numberOfIterations;

    // cout << result << endl;
}

void NNRollOut::sampleSuccessorState(PDState const& current,
                                     int const& actionIndex, PDState& next,
                                     double& reward) const {
    calcReward(current, actionIndex, reward);
    calcSuccessorState(current, actionIndex, next);
    for (unsigned int varIndex = 0;
         varIndex < State::numberOfProbabilisticStateFluents; ++varIndex) {
        next.sample(varIndex);
    }
    State::calcStateFluentHashKeys(next);
    State::calcStateHashKey(next);
}
