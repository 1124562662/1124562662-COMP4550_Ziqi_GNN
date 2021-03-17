#include "helper_exp.h"
#include "assert.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <iostream>
using namespace std;

void head_exp::write(std::ofstream& taskFile) {
    taskFile << "begin" << endl;

    taskFile << "exp_type" << endl;
    taskFile << this->exp_type << endl;

    taskFile << "headVar" << endl;
    this->headVar.mywrite(taskFile);

    taskFile << "headOBJ_names" << endl;
    for (size_t i = 0; i < this->head_OBJ_names.size(); i++) {
        taskFile << this->head_OBJ_names[i] << endl;
    }
    taskFile << "headOBJ_names_end" << endl;

    taskFile << "f_desendents" << endl;
    for (size_t i = 0; i < this->f_desendents.size(); i++) {
        taskFile << "f_desendent_" << endl;
        this->f_desendents[i].write(taskFile);
    }
    taskFile << "f_desendents_end" << endl;

    taskFile << "q_desendents" << endl;
    for (size_t i = 0; i < this->q_desendents.size(); i++) {
        taskFile << "q_desendent_" << endl;
        this->q_desendents[i].write(taskFile);
    }
    taskFile << "q_desendents_end" << endl;

    taskFile << "end" << endl;
}

void head_exp::read(std::ifstream& taskFile) {
    if (taskFile.is_open()) {
        string line;
        std::getline(taskFile, line);
        assert(line == "begin");

        std::getline(taskFile, line);
        assert(line == "exp_type");

        std::getline(taskFile, line);
        this->exp_type = stoi(line);
        assert(this->exp_type != 0);
        cout << "exp_type_" << to_string(this->exp_type) << endl;

        std::getline(taskFile, line);
        assert(line == "headVar");
        this->headVar.myread(taskFile);

        std::getline(taskFile, line);
        assert(line == "headOBJ_names");
        std::getline(taskFile, line);
        while (line != "headOBJ_names_end") {
            this->head_OBJ_names.push_back(line);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "f_desendents");
        std::getline(taskFile, line);
        while (line != "f_desendents_end") {
            assert(line == "f_desendent_");

            fluent_exp fe = fluent_exp();
            fe.read(taskFile);
            this->f_desendents.push_back(fe);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "q_desendents");
        std::getline(taskFile, line);
        while (line != "q_desendents_end") {
            assert(line == "q_desendent_");

            quantifier_exp qe = quantifier_exp();
            qe.read(taskFile);
            this->q_desendents.push_back(qe);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "end");
    }
}

void quantifier_exp::write(std::ofstream& taskFile) {
    taskFile << "begin" << endl;

    taskFile << "exp_type" << endl;
    taskFile << this->exp_type << endl;

    taskFile << "quantifier_OBJ_type" << endl;
    taskFile << this->OBJ_type << endl;

    taskFile << "quantifier_OBJ_name" << endl;
    taskFile << this->OBJ_name << endl;

    taskFile << "f_desendents" << endl;
    for (size_t i = 0; i < this->f_desendents.size(); i++) {
        taskFile << "f_desendent_" << endl;
        this->f_desendents[i].write(taskFile);
    }
    taskFile << "f_desendents_end" << endl;

    taskFile << "q_desendents" << endl;
    for (size_t i = 0; i < this->q_desendents.size(); i++) {
        taskFile << "q_desendent_" << endl;
        this->q_desendents[i].write(taskFile);
    }
    taskFile << "q_desendents_end" << endl;

    taskFile << "end" << endl;
}

void quantifier_exp::read(std::ifstream& taskFile) {
    if (taskFile.is_open()) {
        string line;
        std::getline(taskFile, line);
        assert(line == "begin");

        std::getline(taskFile, line);
        assert(line == "exp_type");

        std::getline(taskFile, line);
        this->exp_type = stoi(line);
        assert(this->exp_type != 0);
        cout << "exp_type_" << to_string(this->exp_type) << endl;

        std::getline(taskFile, line);
        assert(line == "quantifier_OBJ_type");
        std::getline(taskFile, line);
        this->OBJ_type = line;

        std::getline(taskFile, line);
        assert(line == "quantifier_OBJ_name");
        std::getline(taskFile, line);
        this->OBJ_name = line;

        std::getline(taskFile, line);
        assert(line == "f_desendents");
        std::getline(taskFile, line);
        while (line != "f_desendents_end") {
            assert(line == "f_desendent_");

            fluent_exp fe = fluent_exp();
            fe.read(taskFile);
            this->f_desendents.push_back(fe);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "q_desendents");
        std::getline(taskFile, line);
        while (line != "q_desendents_end") {
            assert(line == "q_desendent_");

            quantifier_exp qe = quantifier_exp();
            qe.read(taskFile);
            this->q_desendents.push_back(qe);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "end");
    }
}

void fluent_exp::write(std::ofstream& taskFile) {
    taskFile << "begin" << endl;

    taskFile << "exp_type" << endl;
    taskFile << this->exp_type << endl;

    taskFile << "absv" << endl;
    this->absv.mywrite(taskFile);

    taskFile << "OBJ_names" << endl;
    for (size_t i = 0; i < this->OBJ_names.size(); i++) {
        taskFile << this->OBJ_names[i] << endl;
    }
    taskFile << "OBJ_names_end" << endl;

    taskFile << "end" << endl;
}

void fluent_exp::read(std::ifstream& taskFile) {
    if (taskFile.is_open()) {
        string line;
        std::getline(taskFile, line);
        assert(line == "begin");

        std::getline(taskFile, line);
        this->exp_type = stoi(line);
        assert(this->exp_type != 0);
        cout << "exp_type_" << to_string(this->exp_type) << endl;

        std::getline(taskFile, line);
        assert(line == "absv");
        this->absv.myread(taskFile);

        std::getline(taskFile, line);
        assert(line == "OBJ_names");
        std::getline(taskFile, line);
        while (line != "OBJ_names_end") {
            this->OBJ_names.push_back(line);
            std::getline(taskFile, line);
        }

        std::getline(taskFile, line);
        assert(line == "end");
    }
}

// void all_exp::write(std::ofstream& taskFile) {
//     //"transition_info"

//     taskFile << "begin" << endl;

//     taskFile << "exp_type" << endl;
//     taskFile << this->exp_type << endl;

//     taskFile << "desendents" << endl;

//     for (size_t i = 0; i < this->desendents.size(); i++) {
//         taskFile << "has_exp" << endl;
//         this->desendents[i].write(taskFile);
//     }
//     taskFile << "desendents_end" << endl;

//     taskFile << "OBJ_names" << endl;
//     for (size_t i = 0; i < this->OBJ_names.size(); i++) {
//         taskFile << this->OBJ_names[i] << endl;
//     }
//     taskFile << "OBJ_names_end" << endl;

//     taskFile << "headVar" << endl;
//     this->headVar.mywrite(taskFile);

//     taskFile << "quantifier_OBJ_type" << endl;
//     taskFile << this->quantifier_OBJ_type << endl;

//     taskFile << "quantifier_OBJ_name" << endl;
//     taskFile << this->quantifier_OBJ_name << endl;

//     taskFile << "fluent_exp_absv" << endl;
//     this->fluent_exp_absv.mywrite(taskFile);

//     taskFile << "end" << endl;

//     taskFile.close();
// }

// all_exp all_exp::read(std::ifstream& taskFile) {
//     //"transition_info"
//     all_exp res = all_exp();

//     if (taskFile.is_open()) {
//         string line;
//         getline(taskFile, line);
//         assert(line == "begin");

//         getline(taskFile, line);
//         assert(line == "exp_type");

//         getline(taskFile, line);
//         res.exp_type = stoi(line);
//         assert(res.exp_type != 0);
//         cout << "exp_type_" << to_string(res.exp_type) << endl;

//         getline(taskFile, line);
//         assert(line == "desendents");

//         getline(taskFile, line);
//         while (line != "desendents_end") {
//             assert(line == "has_exp");
//             all_exp he_i;
//             he_i = all_exp::read(taskFile);
//             res.desendents.push_back(he_i);
//             getline(taskFile, line);
//         }

//         getline(taskFile, line);
//         assert(line == "OBJ_names");
//         getline(taskFile, line);
//         while (line != "OBJ_names_end") {
//             res.OBJ_names.push_back(line);
//             getline(taskFile, line);
//         }

//         getline(taskFile, line);
//         assert(line == "headVar");
//         res.headVar.myread(taskFile);

//         getline(taskFile, line);
//         assert(line == "quantifier_OBJ_type");
//         getline(taskFile, line);
//         res.quantifier_OBJ_type = line;

//         getline(taskFile, line);
//         assert(line == "quantifier_OBJ_name");
//         getline(taskFile, line);
//         res.quantifier_OBJ_name = line;

//         getline(taskFile, line);
//         assert(line == "fluent_exp_absv");
//         res.fluent_exp_absv.myread(taskFile);

//         getline(taskFile, line);
//         assert(line == "end");

//         taskFile.close();
//     }

//     return res;
// }