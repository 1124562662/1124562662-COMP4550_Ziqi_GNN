#ifndef SOGBOFAEXPR_H
#define SOGBOFAEXPR_H

#include "assert.h"
#include <map>
#include <set>
#include <torch/torch.h>
#include <vector>
using namespace std;

class sogbofa_expr;
typedef std::pair<sogbofa_expr, sogbofa_expr> sogbofa_expr_Pair;
// STATEFLUENT, ACTIONFLUENT , CONSTANT , CONJUCTION ,
// DISJUNCTION , EQUAL , GREATER , LOWER , GREATEREQUAL,
// LOWEREQUAL , ADDITION , SUBTRACTION , MULTIPLICATION ,
// DIVISION , NEGATION , EXP ,BERN , DISCRETE_DISTRIBUTION ,
// MULTI_CONDITION_CHECKER

class sogbofa_expr {
public:
    static sogbofa_expr createFromString(
        std::string& desc, map<int, string>& actions_indices,
        map<int, string>& statefluents_indices);
    static vector<sogbofa_expr> createExpressions(
        std::string& desc, map<int, string>& actions_indices,
        map<int, string>& statefluents_indices);
    static sogbofa_expr_Pair splitExpressionPair(
        std::string& desc, map<int, string>& actions_indices,
        map<int, string>& statefluents_indices);

    sogbofa_expr(){};

    // used by fluents
    sogbofa_expr(string name, string fluent_full_name)
        : name(name), fluent_full_name(fluent_full_name){};

    // used by constants
    sogbofa_expr(string name, double constant_value)
        : name(name), constant_value(constant_value){};

    // used by connectives
    sogbofa_expr(string name, vector<sogbofa_expr> sub_exprs)
        : name(name), sub_exprs(sub_exprs){};

    // used by DiscreteDistribution
    sogbofa_expr(string name, vector<sogbofa_expr> values,
                 vector<sogbofa_expr> probabilities)
        : name(name), values(values), probabilities(probabilities){};

    // used by multiConditionCheckerc
    sogbofa_expr(string name, vector<sogbofa_expr> conditions,
                 vector<sogbofa_expr> effects, string mcc)
        : name(name), conditions(conditions), effects(effects) {
        assert(mcc == name);
    };

    string name;

    // used by fluents
    string fluent_full_name;

    // constant's value
    double constant_value;

    // used by connectives
    vector<sogbofa_expr> sub_exprs;

    // used by DiscreteDistribution
    vector<sogbofa_expr> values;
    vector<sogbofa_expr> probabilities;

    // used by MultiConditionChecker
    vector<sogbofa_expr> conditions;
    vector<sogbofa_expr> effects;

    void print();
    torch::Tensor evaluate(map<string, torch::Tensor>& state_F_values,
                           map<string, torch::Tensor>& action_F_values);
};

#endif
