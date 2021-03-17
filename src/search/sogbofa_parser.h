#ifndef SPARSER_H
#define SPARSER_H

#include "sogbofa_expr.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

class sog_Parser {
public:
    sog_Parser(){};
    sog_Parser(string filename);

    map<int, string> actions_indices;
    map<int, string> statefluents_indices;
    map<int, sogbofa_expr> CPFs;
    sogbofa_expr reward_function;

private:
    string filename;
    
    string erasechar(string str) {
        std::string::iterator end_pos =
            std::remove(str.begin(), str.end(), ' ');
        str.erase(end_pos, str.end());
        return str;
    }
};

#endif
