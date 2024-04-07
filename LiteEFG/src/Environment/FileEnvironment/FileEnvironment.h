#ifndef FILEENVIRONMENT_H_
#define FILEENVIRONMENT_H_

#include "Environment/Environment.h"

#include "Basic/FileReader.h"

#include <vector>
#include <string>
#include <unordered_map>

class FileNode : public Node {
public:
    std::vector<double> utility;
    std::vector<std::string> actions;
    std::string name;

    FileNode(const int& player_, const int& infoset_, const int& player_num_);
    double GetUtility(const int& player) override;
};

class FileEnvironment : public Environment {
public:
    FileReader file_reader;
    std::vector<FileNode> file_nodes;

    int node_num;
    std::unordered_map<std::string, int> node_map;
    std::vector<int> infoset_num;

    FileEnvironment(const std::string& file_name_, const std::string& traverse_);

};

#endif