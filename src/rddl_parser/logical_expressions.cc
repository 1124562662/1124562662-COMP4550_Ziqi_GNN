#include "logical_expressions.h"

#include "instantiator.h"
#include "probability_distribution.h"

#include "rddl.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"

#include <iostream>

using namespace std;

/*****************************************************************
                        Constructors
*****************************************************************/

Object::Object(std::string _name, Type* _type)
    : Parameter(_name, _type), types() {
    Type* tmpType = type;
    while (tmpType) {
        types.push_back(tmpType);
        value = tmpType->objects.size();
        tmpType->objects.push_back(this);
        tmpType = tmpType->superType;
    }
}

// This constructor is used for instantiation
ParametrizedVariable::ParametrizedVariable(ParametrizedVariable const& source,
                                           std::vector<Parameter*> _params)
    : LogicalExpression(),
      variableName(source.variableName),
      fullName(source.variableName),
      params(_params),
      variableType(source.variableType),
      valueType(source.valueType),
      initialValue(source.initialValue) {
    assert(params.size() == source.params.size());
    for (unsigned int i = 0; i < params.size(); ++i) {
        if (!params[i]->type) {
            params[i]->type = source.params[i]->type;
        } else {
            assert(params[i]->type->isSubtypeOf(source.params[i]->type));
        }
    }

    if (!params.empty()) {
        fullName += "(";
        for (unsigned int i = 0; i < params.size(); ++i) {
            fullName += params[i]->name;
            if (i != params.size() - 1) {
                fullName += ", ";
            }
        }
        fullName += ")";
    }
}

ParametrizedVariable::ParametrizedVariable(ParametrizedVariable const& source,
                                           std::vector<Parameter*> _params,
                                           double _initialValue)
    : ParametrizedVariable(source, _params) {
    initialValue = _initialValue;
}

bool Type::isSubtypeOf(Type* const& other) const {
    Type const* comp = this;
    while (comp) {
        if (other == comp) {
            return true;
        }
        comp = comp->superType;
    }
    return false;
}

#include "logical_expressions_includes/calculate_domain.cc"
#include "logical_expressions_includes/calculate_domain_as_interval.cc"
#include "logical_expressions_includes/classify_action_fluents.cc"
#include "logical_expressions_includes/collect_initial_info.cc"
#include "logical_expressions_includes/determine_bounds.cc"
#include "logical_expressions_includes/determinization.cc"
#include "logical_expressions_includes/evaluate.cc"
#include "logical_expressions_includes/evaluate_to_kleene.cc"
#include "logical_expressions_includes/evaluate_to_pd.cc"
#include "logical_expressions_includes/instantiate.cc"
#include "logical_expressions_includes/print.cc"
#include "logical_expressions_includes/replace_quantifier.cc"
#include "logical_expressions_includes/simplify.cc"

// void Quantifier::write_in_file(std::string quantifier_name, std::string str,
//                                ParameterList* _paramList,
//                                LogicalExpression* expr) {
//     // To do : fix bug if no () exists -------
//     std::ifstream infile("CDFs_file_");
//     std::ofstream taskFile;
//     std::string taskFileName = "CDFs_file_";
//     if (!infile.good()) {
//         taskFile.open(taskFileName.c_str());
//     } else {
//         taskFile.open(taskFileName.c_str(), std::ofstream::app);
//     }
//     taskFile << quantifier_name << std::endl;
//     _paramList->print(taskFile);
//     taskFile << "" << std::endl;

//     std::stringstream os;
//     expr->print(os);
//     std::string expression__ = os.str();
//     std::string last = "";
//     std::string last2 = "";
//     bool last_symbol = false;
//     bool last_symbol2 = false;           //[]
//     bool last_symbol3 = false;           //{}
//     bool contains_question_mark = false; // ?

//     for (std::string::size_type i = 0; i < expression__.size(); ++i) {
//         if (expression__[i] == '(') {
//             last_symbol = true;
//             last2 = last;
//             last = "";
//         } else if (expression__[i] == ')') {
//             if (last_symbol && contains_question_mark && (!last_symbol2) &&
//                 (!last_symbol3)) {
//                 if (last2 != "$c") {
//                     taskFile << str << std::endl;
//                     taskFile << last2 << "(" << last << ")" << std::endl;
//                 }
//             }
//             last2 = last;
//             last = "";
//             last_symbol = false;
//             contains_question_mark = false;
//         } else if (expression__[i] == ' ') {
//         } else if (expression__[i] == '[') {
//             last_symbol2 = true;
//             last2 = last;
//             last = "";
//         } else if (expression__[i] == ']') {
//             last_symbol2 = false;
//             last2 = last;
//             last = "";
//         } else if (expression__[i] == '{') {
//             last_symbol3 = true;
//             last2 = last;
//             last = "";
//         } else if (expression__[i] == '}') {
//             last_symbol3 = false;
//             last2 = last;
//             last = "";
//         } else {
//             if (expression__[i] == '?') {
//                 contains_question_mark = true;
//             }
//             last += expression__[i];
//         }
//     }
//     taskFile << "end" << std::endl;
//     taskFile.close();
// }
