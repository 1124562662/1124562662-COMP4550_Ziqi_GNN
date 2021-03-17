#include "sogbofa_expr.h"

#include "assert.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <iostream>

using namespace std;

torch::Tensor sogbofa_expr::evaluate(
    map<string, torch::Tensor>& state_F_values,
    map<string, torch::Tensor>& action_F_values) {
    auto options =
        torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);

    if (this->name == "STATEFLUENT") {
        //assert(state_F_values.count(this->fluent_full_name) != 0);
        return state_F_values[this->fluent_full_name];

    } else if (this->name == "ACTIONFLUENT") {
        //assert(action_F_values.count(this->fluent_full_name) != 0);
        return action_F_values[this->fluent_full_name];

    } else if (this->name == "CONSTANT") {
        return torch::full({1, 1}, this->constant_value, options);

    } else if (this->name == "CONJUCTION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
            torch::mul(res, t1);
        }

        return res;

    } else if (this->name == "DISJUNCTION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
            torch::Tensor tmp1 = torch::add(res, t1);
            torch::Tensor tmp2 = torch::mul(res, t1);
            res = torch::sub(tmp1, tmp2);
        }
        return res;

    } else if (this->name == "EQUAL") {
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);
        torch::Tensor right =
            this->sub_exprs[1].evaluate(state_F_values, action_F_values);

        // option one : suitable if both sides are boolean
        // a*b+(1-a)*(1-b)

        // torch::Tensor tmp1 = torch::mul(left, right);
        // torch::Tensor one_ = torch::full({1, 1}, 1.0, options);
        // torch::Tensor tmp2 = torch::sub(one_, left);
        // torch::Tensor tmp3 = torch::sub(one_, right);
        // torch::Tensor tmp4 = torch::mul(tmp2, tmp3);
        // torch::Tensor res = torch::add(tmp1, tmp4);

        // option two : suitable for real number
        // the method that proposed from th oringinal paper
        // which is sigmoid(a-b+0.5)-sigmoid(a-b-0.5)
        // BUT the sigmoid() here would cause vanishing gradient problem.

        torch::Tensor tmp1 = torch::sub(left, right);
        torch::Tensor tmp2 =
            torch::sigmoid(torch::add(tmp1, torch::full({1, 1}, 0.5, options)));
        torch::Tensor tmp3 =
            torch::sigmoid(torch::sub(tmp1, torch::full({1, 1}, 0.5, options)));
        torch::Tensor res = torch::sub(tmp2, tmp3);

        return res;

    } else if (this->name == "GREATER" || this->name == "GREATEREQUAL") {
        // a<b is  σ(7(a − b))
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);
        torch::Tensor right =
            this->sub_exprs[1].evaluate(state_F_values, action_F_values);

        torch::Tensor tmp1 = torch::sub(left, right);
        torch::Tensor tmp2 =
            torch::mul(tmp2, torch::full({1, 1}, 5.0, options));
        torch::Tensor tmp3 = torch::sigmoid(tmp2);

        return tmp3;
    } else if (this->name == "LOWER" || this->name == "LOWEREQUAL") {
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);
        torch::Tensor right =
            this->sub_exprs[1].evaluate(state_F_values, action_F_values);

        torch::Tensor tmp1 = torch::sub(right, left);
        torch::Tensor tmp2 =
            torch::mul(tmp2, torch::full({1, 1}, 5.0, options));
        torch::Tensor tmp3 = torch::sigmoid(tmp2);

        return tmp3;
    } else if (this->name == "ADDITION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
            res = torch::add(res, t1);
        }
        return res;
    } else if (this->name == "SUBTRACTION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
            res = torch::sub(res, t1);
        }
        return res;
    } else if (this->name == "MULTIPLICATION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
            res = torch::mul(res, t1);
        }
        return res;
    } else if (this->name == "DIVISION") {
        torch::Tensor res =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        for (size_t i = 1; i < this->sub_exprs.size(); i++) {
            torch::Tensor t1 =
                this->sub_exprs[i].evaluate(state_F_values, action_F_values);
                t1=torch::add(t1, torch::full({1,1},0.000000001,options));
            res = torch::div(res, t1);
        }
        return res;

    } else if (this->name == "NEGATION") {
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        return torch::sub(torch::full({1, 1}, 1.0, options), left);

    } else if (this->name == "EXP") {
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        return torch::exp(left);

    } else if (this->name == "BERN") {
        torch::Tensor left =
            this->sub_exprs[0].evaluate(state_F_values, action_F_values);

        return left;
    } else if (this->name == "DISCRETE_DISTRIBUTION") {
        torch::Tensor res = torch::full({1, 1}, 0.0, options);

        for (size_t i = 0; i < this->values.size(); i++) {
            torch::Tensor t1 =
                this->values[i].evaluate(state_F_values, action_F_values);
            torch::Tensor t2 = this->probabilities[i].evaluate(state_F_values,
                                                               action_F_values);
            torch::Tensor t3 = torch::mul(t1, t2);
            res = torch::add(res, t3);
        }
        cout << "res"<<res<<endl;
        return res;
    } else if (this->name == "MULTI_CONDITION_CHECKER") {
        torch::Tensor res;
        torch::Tensor neg_condi;
        for (size_t i = 0; i < this->conditions.size(); i++) {
            if (i == 0) {
                torch::Tensor condi = this->conditions[0].evaluate(
                    state_F_values, action_F_values);
                torch::Tensor eff =
                    this->effects[i].evaluate(state_F_values, action_F_values);
                res = torch::mul(condi, eff);
                neg_condi =
                    torch::sub(torch::full({1, 1}, 1.0, options), condi);

            } else {
                torch::Tensor condi = this->conditions[i].evaluate(
                    state_F_values, action_F_values);

                torch::Tensor finalcondi = torch::mul(neg_condi, condi);

                neg_condi = torch::mul(
                    neg_condi,
                    torch::sub(torch::full({1, 1}, 1.0, options), condi));

                torch::Tensor eff =
                    this->effects[i].evaluate(state_F_values, action_F_values);

                torch::Tensor tmp_res = torch::mul(finalcondi, eff);
                res = torch::add(res, tmp_res);
            }
        }
      //  cout<<"res"<<res<<endl;
        return res;
    } else {
        SystemUtils::abort("abort in exp evaluate,found:" + name);
    }

    torch::Tensor res;
    return res;
}

sogbofa_expr sogbofa_expr::createFromString(
    string& desc, map<int, string>& actions_indices,
    map<int, string>& statefluents_indices) {
    StringUtils::trim(desc);

    if (StringUtils::startsWith(desc, "$s(")) {
        desc = desc.substr(3, desc.length() - 4);
        int index = atoi(desc.c_str());

        if (statefluents_indices.count(index) == 0) {
            SystemUtils::abort("not found");
        }

        return sogbofa_expr("STATEFLUENT", statefluents_indices[index]);

    } else if (StringUtils::startsWith(desc, "$a(")) {
        desc = desc.substr(3, desc.length() - 4);
        int index = atoi(desc.c_str());

        if (actions_indices.count(index) == 0) {
            SystemUtils::abort("not found");
        }
        return sogbofa_expr("ACTIONFLUENT", actions_indices[index]);

    } else if (StringUtils::startsWith(desc, "$c(")) {
        desc = desc.substr(3, desc.length() - 4);
        double value = atof(desc.c_str());
        return sogbofa_expr("CONSTANT", value);

    } else if (StringUtils::startsWith(desc, "and(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("CONJUCTION", exprs);
    } else if (StringUtils::startsWith(desc, "or(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("DISJUNCTION", exprs);
    } else if (StringUtils::startsWith(desc, "==(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("EQUAL", exprs);
    } else if (StringUtils::startsWith(desc, ">(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("GREATER", exprs);
    } else if (StringUtils::startsWith(desc, "<(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("LOWER", exprs);
    } else if (StringUtils::startsWith(desc, ">=(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("GREATEREQUAL", exprs);
    } else if (StringUtils::startsWith(desc, "<=(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("LOWEREQUAL", exprs);
    } else if (StringUtils::startsWith(desc, "+(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("ADDITION", exprs);
    } else if (StringUtils::startsWith(desc, "-(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("SUBTRACTION", exprs);
    } else if (StringUtils::startsWith(desc, "*(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("MULTIPLICATION", exprs);
    } else if (StringUtils::startsWith(desc, "/(")) {
        vector<sogbofa_expr> exprs =
            createExpressions(desc, actions_indices, statefluents_indices);
        return sogbofa_expr("DIVISION", exprs);
    } else if (StringUtils::startsWith(desc, "~(")) {
        desc = desc.substr(2, desc.length() - 3);
        sogbofa_expr expr = sogbofa_expr::createFromString(
            desc, actions_indices, statefluents_indices);

        vector<sogbofa_expr> exprs;
        exprs.push_back(expr);
        return sogbofa_expr("NEGATION", exprs);

    } else if (StringUtils::startsWith(desc, "exp(")) {
        desc = desc.substr(4, desc.length() - 5);
        sogbofa_expr expr = sogbofa_expr::createFromString(
            desc, actions_indices, statefluents_indices);

        vector<sogbofa_expr> exprs;
        exprs.push_back(expr);
        return sogbofa_expr("EXP", exprs);
    } else if (StringUtils::startsWith(desc, "Bernoulli(")) {
        desc = desc.substr(10, desc.length() - 11);
        sogbofa_expr expr = sogbofa_expr::createFromString(
            desc, actions_indices, statefluents_indices);

        vector<sogbofa_expr> exprs;
        exprs.push_back(expr);
        return sogbofa_expr("BERN", exprs);

    } else if (StringUtils::startsWith(desc, "Discrete(")) {
        desc = desc.substr(9, desc.length() - 10);

        // Extract the value-probability pairs
        vector<string> tokens;
        StringUtils::tokenize(desc, '(', ')', tokens);

        vector<sogbofa_expr> values;
        vector<sogbofa_expr> probabilities;

        for (unsigned int index = 0; index < tokens.size(); ++index) {
            pair<sogbofa_expr, sogbofa_expr> valProbPair = splitExpressionPair(
                tokens[index], actions_indices, statefluents_indices);

            values.push_back(valProbPair.first);
            probabilities.push_back(valProbPair.second);
        }
        return sogbofa_expr("DISCRETE_DISTRIBUTION", values, probabilities);

    } else if (StringUtils::startsWith(desc, "switch(")) {
        desc = desc.substr(7, desc.length() - 8);

        // Extract the condition-effect pairs
        vector<string> tokens;
        StringUtils::tokenize(desc, '(', ')', tokens);
        assert(tokens.size() > 1);

        vector<sogbofa_expr> conditions;
        vector<sogbofa_expr> effects;

        for (unsigned int index = 0; index < tokens.size(); ++index) {
            pair<sogbofa_expr, sogbofa_expr> condEffPair = splitExpressionPair(
                tokens[index], actions_indices, statefluents_indices);

            conditions.push_back(condEffPair.first);
            effects.push_back(condEffPair.second);
        }
        return sogbofa_expr("MULTI_CONDITION_CHECKER", conditions, effects,
                            "MULTI_CONDITION_CHECKER");
    }

    SystemUtils::abort("Failed to create sogbofa expression from string:" +
                       desc);
    return sogbofa_expr();
}

vector<sogbofa_expr> sogbofa_expr::createExpressions(
    string& desc, map<int, string>& actions_indices,
    map<int, string>& statefluents_indices) {
    // desc must be a string of the form keyword(expressions).
    vector<sogbofa_expr> result;

    // Remove keyword and parentheses and create an expression from each
    // token.
    size_t cutPos = desc.find("(") + 1;
    desc = desc.substr(cutPos, desc.length() - cutPos - 1);

    // Extract the expression descriptions (careful, this only works since
    // all expressions end with a closing parenthesis!)
    vector<string> tokens;
    StringUtils::tokenize(desc, '(', ')', tokens);

    // Create the logical expressions
    for (unsigned int index = 0; index < tokens.size(); ++index) {
        result.push_back(sogbofa_expr::createFromString(
            tokens[index], actions_indices, statefluents_indices));
    }

    return result;
}

sogbofa_expr_Pair sogbofa_expr::splitExpressionPair(
    string& desc, map<int, string>& actions_indices,
    map<int, string>& statefluents_indices) {
    // Each pair must be in parentheses
    StringUtils::removeFirstAndLastCharacter(desc);

    vector<string> tokens;
    StringUtils::tokenize(desc, '(', ')', tokens);
    assert(tokens.size() == 2);

    tokens[1] = tokens[1].substr(1);
    StringUtils::trim(tokens[1]);

    sogbofa_expr first =
        createFromString(tokens[0], actions_indices, statefluents_indices);
    sogbofa_expr second =
        createFromString(tokens[1], actions_indices, statefluents_indices);

    return make_pair(first, second);
}

void sogbofa_expr::print() {
    if (fluent_full_name != "") {
        cout << this->fluent_full_name;

    } else if (name == "CONSTANT") {
        cout << to_string(this->constant_value);

    } else if (sub_exprs.size() != 0) {
        cout << this->name;
        cout << "(";
        for (size_t i = 0; i < this->sub_exprs.size(); i++) {
            sub_exprs[i].print();
            if (i < this->sub_exprs.size() - 1) {
                cout << ",";
            }
        }
        cout << ")";
    } else if (this->values.size() != 0) {
        cout << "PROB(";
        assert(values.size() == probabilities.size());
        for (size_t i = 0; i < values.size(); i++) {
            cout << "(";
            values[i].print();
            cout << ":";
            probabilities[i].print();
            cout << ")";
        }
        cout << ")";
    } else if (this->conditions.size() != 0) {
        assert(conditions.size() == effects.size());
        cout << "switch(";
        for (size_t i = 0; i < conditions.size(); i++) {
            cout << "(";
            conditions[i].print();
            cout << ":";
            effects[i].print();
            cout << ")";
        }
        cout << ")";
    } else {
        cout << "shouldn't be here: " << name;
        SystemUtils::abort("should not occur" + name);
    }
}