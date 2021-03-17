#include "RandomSelection.h"
#include "prost_planner.h"
#include "utils/math_utils.h"
#include "utils/system_utils.h"

#include <iostream>
#include <numeric>
using namespace std;

/******************************************************************
                     Search Engine Creation
******************************************************************/

randomselection::randomselection()
    : ProbabilisticSearchEngine("randomselection") {}

bool randomselection::setValueFromString(std::string& param,
                                         std::string& value) {
    return true;
}

/******************************************************************
                       Main Search Functions
******************************************************************/
void randomselection::estimateBestActions(State const& _rootState,
                                          std::vector<int>& bestActions) {
    std::vector<int> actionsToExpand = getApplicableActions(_rootState);

  std::vector<int> candidate;

cout<<"sss"<<to_string(actionsToExpand.size())<<endl;
    for (unsigned int j = 0; j < actionsToExpand.size(); ++j) {
       //if (actionsToExpand[j] == j) {
            candidate.push_back(j);
       // }
    }
    int randint = rand() % (actionsToExpand.size());
cout<<"ssrrrs"<<to_string( randint)<<endl;
    bestActions.push_back(candidate[randint]);
}

void randomselection::printConfig(std::string indent) const {
    SearchEngine::printConfig(indent);
    indent += "  ";
}
