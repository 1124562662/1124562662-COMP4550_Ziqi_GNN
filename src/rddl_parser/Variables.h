#ifndef VARIABLES_H
#define VARIABLES_H
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
using namespace std;

class Variable {
public:
    Variable(){};
    Variable(std::string name, std::string variable_type)
        : name(name), variable_type(variable_type){};

    std::string name;
    // STATE_FLUENT, ACTION_FLUENT, NON_FLUENT
    std::string variable_type;
    virtual void myread(std::ifstream& taskFile) {
        string s;
        std::getline(taskFile,s);
        cout<<s;
    };
    virtual void mywrite(std::ofstream& taskFile) {
        taskFile << "";
    };
};

class AbstractVariable : public Variable {
public:
    AbstractVariable(){};
    AbstractVariable(std::ifstream& taskFile) {
        this->myread(taskFile);
    };
    AbstractVariable(std::string name, std::string variable_type,
                     std::vector<std::string> patameters_types2)
        : Variable(name, variable_type), patameters_types(patameters_types2){};

    // AbstractVariable(InstantiatedVariable insvar)
    //     : Variable(insvar.name, insvar.variable_type),
    //       patameters_types(insvar.patameters_types){};

    virtual void myread(std::ifstream& taskFile);
    virtual void mywrite(std::ofstream& taskFile);
    void mywrite_helper(std::ofstream& taskFile);
    void myread_helper(std::ifstream& taskFile);

    friend bool operator==(const AbstractVariable& v1,
                           const AbstractVariable& v2) {
        bool res = true;
        if (v1.name != v2.name) {
            res = false;
        }
        if (v1.patameters_types != v2.patameters_types) {
            res = false;
        }
        if (v1.variable_type != v2.variable_type) {
            res = false;
        }
        return res;
    }

    std::vector<std::string> patameters_types;
};

class InstantiatedVariable : public AbstractVariable {
public:
    InstantiatedVariable(){};
    InstantiatedVariable(std::string name, std::string fullname,
                         std::string variable_type,
                         std::vector<std::string> patameters_types,
                         std::vector<std::string> instantiated_patameters)
        : AbstractVariable(name, variable_type, patameters_types),
          fullname(fullname),
          instantiated_patameters(instantiated_patameters){};

    InstantiatedVariable(AbstractVariable av, double value, string fullname,
                         vector<string> instantiated_patameters)
        : AbstractVariable(av),
          value(value),
          fullname(fullname),
          instantiated_patameters(instantiated_patameters){};

    InstantiatedVariable(ifstream& taskFile) {
        this->myread(taskFile);
    };

    void myread(std::ifstream& taskFile);
    void mywrite(std::ofstream& taskFile);

    double value;
    std::string fullname;
    std::vector<std::string> instantiated_patameters;
};

class AbstractRestrictedVariable : public AbstractVariable {
public:
    AbstractRestrictedVariable(){};
    AbstractRestrictedVariable(AbstractVariable absVar,
                               vector<int>& ObjectsRestriction_loc,
                               vector<string>& ObjectsRestriction, string ID);
    vector<string> ObjectsRestriction;
    vector<int> ObjectsRestriction_loc;
    string ID;
};

#endif
