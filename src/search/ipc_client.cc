#include "ipc_client.h"

#include "RelationNet.h"
#include "parser.h"
#include "prost_planner.h"
#include "utils/base64.h"
#include "utils/string_utils.h"
#include "utils/strxml.h"
#include "utils/system_utils.h"

#include "sogbofa.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <logger.h>
#include <netdb.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using namespace std;

IPCClient::IPCClient(string _hostName, unsigned short _port,
                     string _parserOptions)
    : hostName(_hostName),
      port(_port),
      socket(-1),
      parserOptions(_parserOptions),
      numberOfRounds(-1),
      remainingTime(0) {}

// This destructor is required here to allow forward declaration of
// ProstPlanner in header because of usage of unique_ptr<ProstPlanner>
IPCClient::~IPCClient() = default;

void IPCClient::run(string const& instanceName, string& plannerDesc) {
    // Reset static members from possible earlier runs
    ProstPlanner::resetStaticMembers();
    // Init connection to the rddlsim server
    initConnection();
    // Request round
    initSession(instanceName, plannerDesc);

    std::string parameter_domian = "";
    for (std::string::size_type i = 0; i < instanceName.size(); ++i) {
        if (instanceName[i] != '_') {
            parameter_domian += instanceName[i];
        } else {
            if (parameter_domian == "game") {
                parameter_domian = "game_of_life";
            }
            break;
        }
    }

    parameter_domian += "/finalResult/";
    vector<double> nextState(stateVariableIndices.size());

    // 4550 tmp

    planner->rn = relation_net(parameter_domian, this->stateVariableIndices,   this->action_indices);

    planner->rn.set_state_pointer(&nextState);
   // relation_net::trainnet_(); SystemUtils::abort("training ends");

    double immediateReward = 0.0;
    planner->rn.helper_double = -100.0;

    map<int, string> qqq;
    map<string, int>::iterator it2;
    for (it2 = action_indices.begin(); it2 != action_indices.end(); it2++) {
        qqq[it2->second] = it2->first;
        cout << it2->first << ": " << to_string(it2->second) << endl;
    }

    // Main loop
    std::map<string, int> useddd;

    for (unsigned int currentRound = 0; currentRound < numberOfRounds;
         ++currentRound) {
        initRound(nextState, immediateReward);

        while (true) {
            planner->initStep(nextState, remainingTime);
            int action_result;
            vector<string> nextActions = planner->plan(action_result);

            // if (action_result != 0) {
            //     string state_s = "";
            //     for (unsigned int index = 0; index < nextState.size();
            //          ++index) {
            //         state_s += std::to_string((int)nextState[index]);
            //     }
            //     if (useddd[state_s] != 9) {
            //         useddd[state_s] = 9;
            //         std::ofstream ofs;
            //         ofs.open("data.txt", std::ofstream::app);
            //         ofs << state_s << "\n";
            //         ofs << action_result << "\n";
            //         ofs.close();
            //     }
            // }

            // //  if (rand() % (currentRound % 20 + 1) == 0) {
            // if (rand() %4 ==0 ) {
            //     int ead = rand() % (this->action_indices.size());
            //     ead += 1;
            //     nextActions.clear();
            //     nextActions.push_back(qqq[ead]);
            //     cout << "kkkk:" << to_string(ead) << qqq[ead] << endl;
            // }

            if (!submitAction(nextActions, nextState, immediateReward)) {
                break;
            }

            // cout << "round:" << to_string(currentRound) << endl;

            planner->finishStep(immediateReward);
        }
    }

    // Get end of session message and print total result
    finishSession();

    // Close connection to the rddlsim server
    closeConnection();
}

/******************************************************************************
                               Server Communication
******************************************************************************/

void IPCClient::initConnection() {
    assert(socket == -1);
    try {
        socket = connectToServer();
        if (socket <= 0) {
            SystemUtils::abort("Error: couldn't connect to server.");
        }
    } catch (const exception& e) {
        SystemUtils::abort("Error: couldn't connect to server.");
    } catch (...) {
        SystemUtils::abort("Error: couldn't connect to server.");
    }
}

int IPCClient::connectToServer() {
    struct hostent* host = ::gethostbyname(hostName.c_str());
    if (!host) {
        return -1;
    }

    int res = ::socket(PF_INET, SOCK_STREAM, 0);
    if (res == -1) {
        return -1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr = *((struct in_addr*)host->h_addr);
    memset(&(addr.sin_zero), '\0', 8);

    if (::connect(res, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        return -1;
    }
    return res;
}

void IPCClient::closeConnection() {
    if (socket == -1) {
        SystemUtils::abort("Error: couldn't disconnect from server.");
    }
    close(socket);
}

/******************************************************************************
                     Session and rounds management
******************************************************************************/

void IPCClient::initSession(string const& instanceName, string& plannerDesc) {
    stringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
       << "<session-request>"
       << "<problem-name>" << instanceName << "</problem-name>"
       << "<client-name>"
       << "prost"
       << "</client-name>"
       << "<input-language>rddl</input-language>"
       << "<no-header/>"
       << "</session-request>" << '\0';
    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        SystemUtils::abort("Error: writing to socket failed.");
    }

    const XMLNode* serverResponse = XMLNode::readNode(socket);

    if (!serverResponse) {
        SystemUtils::abort("Error: initializing session failed.");
    }

    string s;
    // If the task was not initialized, we have to read it from the server and
    // run the parser
    assert(SearchEngine::taskName.empty());
    if (!serverResponse->dissect("task", s)) {
        SystemUtils::abort(
            "Error: server response does not contain task description.");
    }

    s = decodeBase64(s);
    executeParser(s, &(this->action_f_z));
    if (!serverResponse->dissect("num-rounds", s)) {
        SystemUtils::abort("Error: server response insufficient.");
    }
    numberOfRounds = atoi(s.c_str());

    if (!serverResponse->dissect("time-allowed", s)) {
        SystemUtils::abort("Error: server response insufficient.");
    }
    remainingTime = atoi(s.c_str());

    delete serverResponse;
    // in c++ 14 we would use make_unique<ProstPlanner>

    planner = std::unique_ptr<ProstPlanner>(new ProstPlanner(plannerDesc));

    planner->initSession(numberOfRounds, remainingTime);
}

void IPCClient::finishSession() {
    XMLNode const* sessionEndResponse = XMLNode::readNode(socket);

    if (sessionEndResponse->getName() != "session-end") {
        SystemUtils::abort("Error: session end message insufficient.");
    }
    string s;
    if (!sessionEndResponse->dissect("total-reward", s)) {
        SystemUtils::abort("Error: session end message insufficient.");
    }
    double totalReward = atof(s.c_str());

    delete sessionEndResponse;

    planner->finishSession(totalReward);
}

void IPCClient::initRound(vector<double>& initialState,
                          double& immediateReward) {
    stringstream os;
    os.str("");
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
       << "<round-request> <execute-policy>yes</execute-policy> "
          "</round-request>"
       << '\0';

    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        SystemUtils::abort("Error: writing to socket failed.");
    }

    XMLNode const* serverResponse = XMLNode::readNode(socket);

    if (!serverResponse || serverResponse->getName() != "round-init") {
        SystemUtils::abort("Error: round-request response insufficient.");
    }

    string s;
    if (!serverResponse->dissect("time-left", s)) {
        SystemUtils::abort("Error: round-request response insufficient.");
    }
    remainingTime = atoi(s.c_str());

    delete serverResponse;

    serverResponse = XMLNode::readNode(socket);

    readState(serverResponse, initialState, immediateReward);

    assert(MathUtils::doubleIsEqual(immediateReward, 0.0));

    delete serverResponse;

    planner->initRound(remainingTime);
}

void IPCClient::finishRound(XMLNode const* node, double& immediateReward) {
    // TODO: Move immediate rewards
    string s;
    if (!node->dissect("immediate-reward", s)) {
        SystemUtils::abort("Error: round end message insufficient.");
    }
    immediateReward = atof(s.c_str());

    if (!node->dissect("round-reward", s)) {
        SystemUtils::abort("Error: server communication failed.");
    }

    double roundReward = atof(s.c_str());

    // zzq
    std::ofstream ofs;
    ofs.open("roundRewards.txt", std::ofstream::app);
    ofs << roundReward << "\n";
    ofs.close();
    // zzq

    planner->finishStep(immediateReward);
    planner->finishRound(roundReward);
}

/******************************************************************************
                         Submission of actions
******************************************************************************/

bool IPCClient::submitAction(vector<string>& actions, vector<double>& nextState,
                             double& immediateReward) {
    stringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
       << "<actions>";

    for (unsigned int i = 0; i < actions.size(); ++i) {
        size_t cutPos = actions[i].find("(");
        if (cutPos == string::npos) {
            os << "<action><action-name>" << actions[i] << "</action-name>";
        } else {
            string actionName = actions[i].substr(0, cutPos);
            os << "<action><action-name>" << actionName << "</action-name>";
            string allParams = actions[i].substr(cutPos + 1);
            assert(allParams[allParams.length() - 1] == ')');
            allParams = allParams.substr(0, allParams.length() - 1);
            vector<string> params;
            StringUtils::split(allParams, params, ",");
            for (unsigned int j = 0; j < params.size(); ++j) {
                StringUtils::trim(params[j]);
                os << "<action-arg>" << params[j] << "</action-arg>";
            }
        }
        os << "<action-value>true</action-value></action>";
    }
    os << "</actions>" << '\0';
    if (write(socket, os.str().c_str(), os.str().length()) == -1) {
        return false;
    }
    XMLNode const* serverResponse = XMLNode::readNode(socket);

    bool roundContinues = true;
    if (serverResponse->getName() == "round-end") {
        finishRound(serverResponse, immediateReward);
        roundContinues = false;
    } else {
        readState(serverResponse, nextState, immediateReward);
        // to do
    }

    delete serverResponse;
    return roundContinues;
}

/******************************************************************************
                             Receiving of states
******************************************************************************/

void IPCClient::readState(XMLNode const* node, vector<double>& nextState,
                          double& immediateReward) {
    assert(node);
    assert(node->getName() == "turn");

    if (node->size() == 2 &&
        node->getChild(1)->getName() == "no-observed-fluents") {
        assert(false);
    }

    map<string, string> newValues;

    string s;
    if (!node->dissect("time-left", s)) {
        SystemUtils::abort("Error: turn response message insufficient.");
    }
    remainingTime = atoi(s.c_str());

    if (!node->dissect("immediate-reward", s)) {
        SystemUtils::abort("Error: turn response message insufficient.");
    }
    immediateReward = atof(s.c_str());

    for (int i = 0; i < node->size(); i++) {
        XMLNode const* child = node->getChild(i);
        if (child->getName() == "observed-fluent") {
            // 4550 todo

            readVariable(child, newValues);
        }
    }

    for (map<string, string>::iterator it = newValues.begin();
         it != newValues.end(); ++it) {
        string varName = it->first;
        string value = it->second;

        // If the variable has no parameters, its name is different from the one
        // that is used by PROST internally where no parents are used (afaik,
        // this changed at some point in rddlsim, and I am not sure if it will
        // change back which is why this hacky solution is fine for the moment).
        if (varName[varName.length() - 2] == '(') {
            varName = varName.substr(0, varName.length() - 2);
        }

        if (stateVariableIndices.find(varName) != stateVariableIndices.end()) {
            if (stateVariableValues[stateVariableIndices[varName]].empty()) {
                // TODO: This should be a numerical variable without
                // value->index mapping, but it can also be a boolean one atm.
                if (value == "true") {
                    nextState[stateVariableIndices[varName]] = 1.0;
                } else if (value == "false") {
                    nextState[stateVariableIndices[varName]] = 0.0;
                } else {
                    nextState[stateVariableIndices[varName]] =
                        atof(value.c_str());
                }
            } else {
                for (unsigned int i = 0;
                     i <
                     stateVariableValues[stateVariableIndices[varName]].size();
                     ++i) {
                    if (stateVariableValues[stateVariableIndices[varName]][i] ==
                        value) {
                        nextState[stateVariableIndices[varName]] = i;
                        break;
                    }
                }
            }
        }
    }
}

void IPCClient::readVariable(XMLNode const* node, map<string, string>& result) {
    string name;
    if (node->getName() != "observed-fluent") {
        assert(false);
    }

    if (!node->dissect("fluent-name", name)) {
        assert(false);
    }
    name = name.substr(0, name.length() - 1);

    vector<string> params;
    string value;
    string fluentName;

    for (int i = 0; i < node->size(); i++) {
        XMLNode const* paramNode = node->getChild(i);
        if (!paramNode) {
            assert(false);
            continue;
        } else if (paramNode->getName() == "fluent-arg") {
            string param = paramNode->getText();
            params.push_back(param.substr(0, param.length() - 1));
        } else if (paramNode->getName() == "fluent-value") {
            value = paramNode->getText();
            value = value.substr(0, value.length() - 1);
        }
    }
    name += "(";
    for (unsigned int i = 0; i < params.size(); ++i) {
        name += params[i];
        if (i != params.size() - 1) {
            name += ", ";
        }
    }
    name += ")";
    assert(result.find(name) == result.end());
    result[name] = value;
}

/******************************************************************************
                             Parser Interaction
******************************************************************************/

void IPCClient::executeParser(string const& taskDesc,
                              std::vector<ActionFluent*>* action_f_z) {
#ifdef NDEBUG
    std::string parserExec = "./rddl-parser-release ";
#else
    std::string parserExec = "./rddl-parser-debug ";
#endif
    Logger::logLine("Running RDDL parser at " + parserExec, Verbosity::VERBOSE);
    // Generate temporary input file for parser
    std::ofstream taskFile;
    stringstream taskFileNameStream;
    taskFileNameStream << "./parser_in_" << ::getpid() << ".rddl";
    string taskFileName = taskFileNameStream.str();
    taskFile.open(taskFileName.c_str());

    taskFile << taskDesc << endl;
    taskFile.close();

    stringstream parserOutStream;
    parserOutStream << "parser_out_" << ::getpid();
    string parserOut = parserOutStream.str();
    stringstream callString;
    callString << parserExec << taskFileName << " ./" << parserOut << " "
               << parserOptions;
    int result = std::system(callString.str().c_str());
    if (result != 0) {
        SystemUtils::abort("Error: " + parserExec + " had an error");
    }

    Parser parser(parserOut, action_f_z);
    parser.parseTask(stateVariableIndices, stateVariableValues,
                     &(this->action_indices));

    remove("objects_file_z");

    // Remove temporary files

    if ((remove(taskFileName.c_str()) != 0) ||
        (remove(parserOut.c_str()) != 0)) {
        SystemUtils::abort("Error: deleting temporary file failed");
    }
}

std::string IPCClient::erasechar(std::string str) {
    std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    return str;
}
