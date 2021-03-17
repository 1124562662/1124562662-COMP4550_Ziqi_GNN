#include "Variables.h"

#include "utils/system_utils.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace std;

AbstractRestrictedVariable::AbstractRestrictedVariable(
    AbstractVariable absVar, vector<int>& ObjectsRestriction_loc,
    vector<string>& ObjectsRestriction,string ID)
    : AbstractVariable(absVar),
      ObjectsRestriction(ObjectsRestriction),
      ObjectsRestriction_loc(ObjectsRestriction_loc),ID(ID) {
   
}

 

void InstantiatedVariable::myread(ifstream& taskFile) {
    if (taskFile.is_open()) {
        string line;
        getline(taskFile, line);
        assert(line == "InstantiatedVariable");

        getline(taskFile, line);
        assert(line == "fullname:");
        getline(taskFile, line);
        this->fullname = line;

        getline(taskFile, line);
        assert(line == "value");
        getline(taskFile, line);
        this->value = atof(line.c_str());

        cout<<"fluent:"<<this->fullname<<".         has value:"<< to_string( this->value )<<"."<<endl;
        this->myread_helper(taskFile);

        getline(taskFile, line);
        assert(line == "instantiated_patameters:");

        while (getline(taskFile, line)) {
            if (line == "end") {
                break;
            } else {
                this->instantiated_patameters.push_back(line);
            }
        }
    }
     
}

void AbstractVariable::myread(ifstream& taskFile) {
    if (taskFile.is_open()) {
        string line;
        getline(taskFile, line);
        assert(line == "AbstractVariable");
        this->myread_helper(taskFile);
    }
}

void InstantiatedVariable::mywrite(ofstream& taskFile) {
    taskFile << "InstantiatedVariable" << endl;

    taskFile << "fullname:" << endl;
    taskFile << this->fullname << endl;

    taskFile << "value" << endl;
    taskFile << this->value << endl;

    this->mywrite_helper(taskFile);

    taskFile << "instantiated_patameters:" << endl;
    for (int i = 0; i < this->instantiated_patameters.size(); i++) {
        taskFile << (this->instantiated_patameters)[i] << endl;
    }
    taskFile << "end" << endl;
}


void AbstractVariable::mywrite(ofstream& taskFile) {
    taskFile << "AbstractVariable" << endl;
    this->mywrite_helper(taskFile);
}

void AbstractVariable::mywrite_helper(ofstream& taskFile) {
    taskFile << "Name:" << endl;
    taskFile << (this->name) << endl;
    taskFile << "type:" << endl;
    taskFile << this->variable_type << endl;
    taskFile << "patameters_types:" << endl;
    for (int i = 0; i < this->patameters_types.size(); i++) {
        taskFile << (this->patameters_types)[i] << endl;
    }
    taskFile << "end" << endl;
}
void AbstractVariable::myread_helper(ifstream& taskFile) {
    if (taskFile.is_open()) {
        std::string line;
        getline(taskFile, line);
        assert(line == "Name:");

        getline(taskFile, line);
        this->name = line;

        getline(taskFile, line);
        assert(line == "type:");

        getline(taskFile, line);
        this->variable_type = line;

        getline(taskFile, line);
        assert(line == "patameters_types:");

        while (getline(taskFile, line)) {
            if (line == "end") {
                break;
            } else {
                this->patameters_types.push_back(line);
            }
        }
    }
}