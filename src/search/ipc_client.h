#ifndef RDDL_CLIENT_H
#define RDDL_CLIENT_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include "evaluatables.h"

class ProstPlanner;
class XMLNode;


class IPCClient {
public:
    IPCClient(std::string _hostName, unsigned short _port,
              std::string parserOptions);
    ~IPCClient();

    void run(std::string const& instanceName, std::string& plannerDesc);


private:
    void initConnection();
    int connectToServer();
    void closeConnection();

    void initSession(std::string const& instanceName, std::string& plannerDesc);
    void finishSession();

    void initRound(std::vector<double>& initialState, double& immediateReward);
    void finishRound(XMLNode const* node, double& immediateReward);

    bool submitAction(std::vector<std::string>& action,
                      std::vector<double>& nextState, double& immediateReward);

    void readState(XMLNode const* node, std::vector<double>& nextState,
                   double& immediateReward);
    void readVariable(XMLNode const* node,
                      std::map<std::string, std::string>& result);

    // If the client call did not contain a task file, we have to read the task
    // description from the server and run the external parser to create a task
    // in PROST format.
    void executeParser(std::string const& taskDesc,std::vector<ActionFluent*>* action_f_z);

    std::unique_ptr<ProstPlanner> planner;
    std::string hostName;
    unsigned short port;
    int socket;

    std::string parserOptions;

    int numberOfRounds;

    long remainingTime;

    std::map<std::string, int> stateVariableIndices;
    std::vector<std::vector<std::string>> stateVariableValues;

    // 4550

    
    // key : string of non-fluents(obj_1,...,onj_k)
    // value : double , the value of the nonfluent
    std::map<std::string, double> NonfluentsToValues;

    
    // key : string of state fluents with parameters
    // value : a vector of fixed-size dependent state fluents with parametetrs
    std::map<std::string, std::set<std::string>> sf2FixedSize_sfs;

    
    // key : string of state fluents with parameters -> quantifiers
    // -> vector of parameters and dependent state fluents
    std::map<std::string, std::map<std::string, std::set<std::string>>>  sf2VariedSize_sfs;

    
    // state fluent string -> {[object,type]}
    std::map<std::string, std::set<std::vector<std::string>>> sf_parameters_type_z;

    
    // string of type -> string of objects
    std::map<std::string, std::set<std::string>>  type_2_objs;
    

    std::vector<ActionFluent*> action_f_z;

    // action indices
    std::map<std::string, int> action_indices;

    bool saving=false ;

    std::string erasechar(std::string in );
    
};

#endif
