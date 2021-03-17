#ifndef randomselection_H
#define randomselection_H

#include "prost_planner.h"
#include "search_engine.h"
#include "states.h"
#include "utils/stopwatch.h"
// Evaluates all actions by simulating a run that starts with that action
// followed by a  NN until a terminal state is reached

class randomselection : public ProbabilisticSearchEngine {
public:
    randomselection();
 
    // Set parameters from command line
    bool setValueFromString(std::string& param, std::string& value) override;

    void estimateBestActions(State const& _rootState,
                             std::vector<int>& bestActions) override;

    //
    bool usesBDDs() const override {
        return false;
    }
    void estimateQValue(State const& state, int actionIndex,
                        double& qValue) override {
        SystemUtils::abort("estimateQValue used in  randomselection ");
    }

    void estimateQValues(State const& state,
                         std::vector<int> const& actionsToExpand,
                         std::vector<double>& qValues) override {
        SystemUtils::abort("estimateQValue used in  randomselection ");
    }
    // Print
    void printConfig(std::string indent) const override;
    void printRoundStatistics(std::string /*indent*/) const override {}
    void printStepStatistics(std::string /*indent*/) const override {}
   

private:
    
};

#endif
