#ifndef PARSER_H
#define PARSER_H

#include <map>
#include <string>
#include <vector>

class PlanningTask;
class ActionFluent;
class StateFluent;
class DeterministicEvaluatable;
class ProbabilisticEvaluatable;
class ConditionalProbabilityFunction;
class RewardFunction;
class ActionState;
class State;

class Parser {
public:
    Parser(std::string _problemFileName, std::vector<ActionFluent*>* action_f_z)
        : problemFileName(_problemFileName) {
        this->action_f_s = (action_f_z);
    }

    void parseTask(std::map<std::string, int>& stateVariableIndices,
                   std::vector<std::vector<std::string>>& stateVariableValues,
                   std::map<std::string, int>* action_indices);

private:
    std::string problemFileName;
    std::vector<ActionFluent*>* action_f_s;

    inline void parseActionFluent(std::stringstream& desc) const;
    inline void parseCPF(std::stringstream& desc,
                         std::vector<std::string>& deterministicFormulas,
                         std::vector<std::string>& probabilisticFormulas,
                         std::vector<std::string>& determinizedFormulas,
                         bool const& isProbabilistic) const;
    inline void parseRewardFunction(std::stringstream& desc) const;
    inline void parseActionPrecondition(std::stringstream& desc) const;
    inline void parseCachingType(std::stringstream& desc,
                                 ProbabilisticEvaluatable* probEval,
                                 DeterministicEvaluatable* detEval) const;
    inline void parseActionHashKeyMap(std::stringstream& desc,
                                      ProbabilisticEvaluatable* probEval,
                                      DeterministicEvaluatable* detEval) const;
    inline void parseActionState(std::stringstream& desc,std::map<std::string,int>* action_indices) const;
    inline void parseHashKeys(std::stringstream& desc) const;
    inline void parseTrainingSet(std::stringstream& desc) const;
};

#endif
