#ifndef HELPEREXP_H
#define HELPEREXP_H

#include "Variables.h"
#include "utils/system_utils.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

class helper_exp {
public:
    helper_exp(){};
    helper_exp(int exp_type) : exp_type(exp_type){};

    int exp_type = -1;

    virtual void write(std::ofstream& taskFile) {
        SystemUtils::abort("virtual  write called");
        taskFile << "";
    };
    virtual void read(std::ifstream& taskFile) {
        SystemUtils::abort("virtual  read called");
        string line;
        std::getline(taskFile, line);
        cout << line;
    };
};

class fluent_exp : public helper_exp {
public:
    fluent_exp(){};
    fluent_exp(AbstractVariable absv, vector<string> OBJ_names)
        : helper_exp(3), absv(absv), OBJ_names(OBJ_names){};

    AbstractVariable absv;

    // the name is like ?f, ?cur...
    // size( OBJ_names)== size(parameter types)
    vector<string> OBJ_names;

    void write(std::ofstream& taskFile);
    void read(std::ifstream& taskFile);
};

class quantifier_exp;
class head_exp : public helper_exp {
public:
    head_exp(){};
    head_exp(AbstractVariable headVar, vector<string> head_OBJ_names,
             vector<fluent_exp> f_desendents,
             vector<quantifier_exp> q_desendents)
        : helper_exp(1),
          headVar(headVar),
          head_OBJ_names(head_OBJ_names),
          f_desendents(f_desendents),
          q_desendents(q_desendents){};

    AbstractVariable headVar;

    // the name is like ?f, ?cur...
    // size( OBJ_names)== size(parameter types)
    vector<string> head_OBJ_names;

    vector<fluent_exp> f_desendents;
    vector<quantifier_exp> q_desendents;

    void write(std::ofstream& taskFile);
    void read(std::ifstream& taskFile);
};

class quantifier_exp : public helper_exp {
public:
    quantifier_exp(){};
    quantifier_exp(string OBJ_type, string OBJ_name,
                   vector<fluent_exp> f_desendents,
                   vector<quantifier_exp> q_desendents)
        : helper_exp(2),
          OBJ_type(OBJ_type),
          OBJ_name(OBJ_name),
          f_desendents(f_desendents),
          q_desendents(q_desendents){};

    string OBJ_type;
    string OBJ_name;

    vector<fluent_exp> f_desendents;
    vector<quantifier_exp> q_desendents;

    void write(std::ofstream& taskFile);
    void read(std::ifstream& taskFile);
};

// class all_exp {
// public:
//     all_exp(){};

//     // head_exp
//     all_exp(AbstractVariable headVar, vector<string> head_OBJ_names,
//             vector<all_exp> desendents)
//         : exp_type(1),
//           headVar(headVar),
//           OBJ_names(head_OBJ_names),
//           desendents(desendents){};

//     // quantifier_exp
//     all_exp(string OBJ_type, string OBJ_name, vector<all_exp> desendents)
//         : exp_type(2),
//           quantifier_OBJ_type(OBJ_type),
//           quantifier_OBJ_name(OBJ_name),
//           desendents(desendents){};

//     // fluent_exp
//     all_exp(AbstractVariable absv, vector<string> OBJ_names)
//         : exp_type(3), fluent_exp_absv(absv), OBJ_names(OBJ_names){};

//     // 1.common
//     int exp_type = -1;

//     // 2.head_exp and quantifier_exp
//     vector<all_exp> desendents;

//     // 3.head_exp and fluent_exp
//     // the name is like ?f, ?cur...
//     // size( OBJ_names)== size(parameter types)
//     vector<string> OBJ_names;

//     // 4.head_exp
//     AbstractVariable headVar;

//     // 5.quantifier_exp
//     string quantifier_OBJ_type;
//     string quantifier_OBJ_name;

//     // 6.fluent_exp
//     AbstractVariable fluent_exp_absv;

//     void write(std::ofstream& taskFile);
//     static all_exp read(std::ifstream& taskFile);

//     //
//     void instantiate(map<string, string> subsitude_obj);

// };

#endif
