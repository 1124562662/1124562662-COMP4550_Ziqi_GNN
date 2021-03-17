#include "sogbofa_parser.h"

#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;
sog_Parser::sog_Parser(string filename) : filename(filename) {
    map<int, string> cpfs_desc;
    string reward_s;

    std::ifstream taskFile;
    taskFile.open(filename + "sogbofa");

    if (taskFile.is_open()) {
        string line;
        while (std::getline(taskFile, line)) {
            if (line == "action") {
                std::getline(taskFile, line);
                string name = line;
                std::getline(taskFile, line);
                int index = stoi(line);

                this->actions_indices[index] = erasechar(name);
                std::getline(taskFile, line);
                assert(line == "action_end");

            } else if (line == "cpf") {
                std::getline(taskFile, line);
                string name = line;

                std::getline(taskFile, line);
                assert(line == "formula");

                std::getline(taskFile, line);
                string formula = line;

                std::getline(taskFile, line);
                assert(line == "formula_end");

                std::getline(taskFile, line);
                assert(line == "index");

                std::getline(taskFile, line);
                int index = stoi(line);

                this->statefluents_indices[index] = erasechar(name);
                cpfs_desc[index] = formula;

            } else if (line == "rewardfunction") {
                std::getline(taskFile, line);
                reward_s = line;
                std::getline(taskFile, line);
                assert(line == "rewardfunction_end");
            } else {
                SystemUtils::abort("sog parser" + line);
            };
        };

        taskFile.close();
    }

    map<int, string>::iterator it;
    for (it = cpfs_desc.begin(); it != cpfs_desc.end(); it++) {
        string desc = it->second;
        this->CPFs[it->first] = sogbofa_expr::createFromString(
            desc, this->actions_indices, this->statefluents_indices);
    }
    this->reward_function = sogbofa_expr::createFromString(
        reward_s, this->actions_indices, this->statefluents_indices);

    // remove("sogbofa");
}