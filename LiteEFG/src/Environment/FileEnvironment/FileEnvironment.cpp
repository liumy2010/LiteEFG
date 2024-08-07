#include "Environment/FileEnvironment/FileEnvironment.h"

#include "Basic/FileReader.h"
#include "Basic/BasicFunction.h"

#include <string>

FileNode::FileNode(const int& player_, const int& infoset_, const int& player_num_) : Node(player_, infoset_, player_num_) {
    utility = std::vector<double>(player_num_+1, 0.0);
    actions.clear();
    is_root = true;
}

double FileNode::GetUtility(const int& player){
    return utility[player];
}

void FileEnvironment::GetNextNode(const bool& is_openspiel){
    std::string name;
    if(!is_openspiel){
        for(auto& node : file_nodes){
            node.next_node.clear();
            for(int i=0; i<node.actions.size(); ++i){
                name = (node.name.back() == '/') ? node.name : node.name + '/';
                name = (node.player == 0) ? name + "C:" : name + 'P' + std::to_string(node.player) + ':';
                name += node.actions[i];
                node.next_node.push_back(node_map[name]);
            }
        }
    } else{
        for(auto& node : file_nodes){
            node.next_node.clear();
            for(int i=0; i<node.actions.size(); ++i){
                node.next_node.push_back(node_map[node.actions[i]]);
            }
        }
    }
}

FileEnvironment::FileEnvironment(const std::string& file_name_, const std::string& traverse_)
                                : Environment(0, traverse_), file_reader(file_name_), node_num(0), infoset_num(0){
    // Constructor. Constructor of Environment will be re-called at the end of this constructor
    // Read the game tree from file
    // The file should be in the format of the game tree

    file_nodes.clear();
    node_map.clear();
    infoset_map.clear();
    
    std::string name, player_name, temp_name;
    char c;
    bool is_openspiel = false;
    
    player_num = -1;
    while((c=file_reader.Read()) == '#'){ // Jump the descriptions of the game
        file_reader.NextWord();
        char cc = file_reader.GetWord(name);
        if(name == "num_players:" || name == "players:"){
            cc = file_reader.GetWord(name);
            name = name.substr(0, name.size()-1);
            player_num = std::stoi(name);
        } else if(name == "openspiel,"){
            is_openspiel = true;
        } else if(name == "}"){
            break;
        }
        if(cc != '\n') file_reader.NextLine();
    }
    while((c=file_reader.Read()) == '#')
        file_reader.NextLine();

    if(player_num == -1) throw std::runtime_error("Please specify the number of players at the beginning of the file");
    infoset_names.resize(player_num+1);
    infoset_names_backup.resize(player_num+1);
    infoset_num = std::vector<int>(player_num+1, 0);
    
    while(true){
        if(c == 'n'){ // node description
            file_reader.NextWord();
            file_reader.GetWord(name); // get the name of the node

            node_map[name] = node_num++;
            file_nodes.push_back(FileNode(0, 0, player_num));
            FileNode& node = file_nodes.back();
            node.name = name;

            char cc = file_reader.Read();
            file_reader.NextWord();
            if(cc == 'c') { // chance node
                /*
                    Example (Kuhn Poker): 
                    node / chance actions 12=0.16666667 13=0.16666667 21=0.16666667 23=0.16666667 31=0.16666667 32=0.16666667
                */
                node.player = 0;
                file_reader.NextWord(); // Skip "actions"
                char ccc;
                while(true){
                    ccc=file_reader.GetWord(name);
                    node.actions.push_back(Basic::GetSlice(name, 0, '='));
                    std::string& action = node.actions.back();
                    double prob = std::stod(Basic::GetSlice(name, action.size()+1, '\n'));
                    node.chance.push_back(prob);
                    if(ccc == '\n') break;
                }
                node.chance.Div(node.chance.Sum());
            } else if(cc == 'p') { // player node
                /*
                    Example (Kuhn Poker):
                    node /C:12 player 1 actions k b
                */
                file_reader.GetWord(player_name);
                node.player = std::stoi(player_name);

                file_reader.NextWord(); // Skip "actions"

                char ccc;
                while(true){
                    ccc = file_reader.GetWord(name);
                    node.actions.push_back(name);
                    if(ccc == '\n') break;
                }
            } else if(cc == 'l') { // terminal node
                /*
                    Example (Kuhn Poker):
                    node /C:12/P1:k/P2:k leaf payoffs 1=-1 2=1
                */
                file_reader.NextWord();
                int player = 0;

                char ccc;
                while(true){
                    ccc = file_reader.GetWord(name);
                    player_name = Basic::GetSlice(name, 0, '=');
                    player = std::stoi(player_name);
                    node.utility[player] = std::stod(Basic::GetSlice(name, player_name.size() + 1, '\n'));
                    if(ccc == '\n') break;
                }
            } else{
                throw std::runtime_error("Unknown node type, only supports [\"chance\", \"player\", \"leaf\"]");
            }
        } 
        else if(c == 'i'){ // infoset description
            /*
                Example (Kuhn Poker):
                infoset pl1_0__1?/ nodes /C:12 /C:13 
            */
            file_reader.NextWord();
            file_reader.GetWord(temp_name);
            file_reader.NextWord();

            char ccc;
            int player;
            while(true){
                ccc = file_reader.GetWord(name);
                auto it = node_map.find(name);
                if(it == node_map.end()) throw std::runtime_error("Please specify all nodes before defining infosets: Infoset: " + temp_name + " Node: " + name);
                player = file_nodes[it -> second].player;
                file_nodes[it -> second].infoset = infoset_num[player] + 1;
                if(ccc == '\n' || ccc == EOF) {++infoset_num[player]; break;}
            }
            infoset_names_backup[player].push_back(temp_name);
        }
        else break;
        c = file_reader.Read();
    }

    for(auto& node : file_nodes){
        if(node.player !=0 && node.infoset == 0) {
            node.infoset = ++infoset_num[node.player];
            infoset_names_backup[node.player].push_back("pl"+std::to_string(node.player)+"_"+std::to_string(infoset_num[node.player])+"__singleton");
        }
        // sometimes singleton infosets are not mentioned in the file, so we need to add them manually
    }
    GetNextNode(is_openspiel);

    for(auto& node : file_nodes){
        for(int i=0; i<node.next_node.size(); ++i){
            file_nodes[node.next_node[i]].is_root = false;
        }
    }
    int root=0;
    for(root=0; root<file_nodes.size(); ++root){
        if(file_nodes[root].is_root) break;
    }
    if(root == file_nodes.size()) throw std::runtime_error("No root node found");
    
    nodes.push_back(&file_nodes[root]);
    node_map[file_nodes[root].name] = 0;
    for(int i=0; i<nodes.size(); ++i){
        for(int j=0; j<nodes[i]->next_node.size(); ++j){
            nodes.push_back(&file_nodes[nodes[i]->next_node[j]]);
            node_map[file_nodes[nodes[i]->next_node[j]].name] = nodes.size()-1;
        }
    }
    if(nodes.size() != node_num)
        throw std::runtime_error("Some nodes are not connected to the tree");

    GetNextNode(is_openspiel);
    for(int i=0; i<=player_num; ++i) infoset_num[i] = 0;
    for(int i=0; i<nodes.size(); ++i) if(nodes[i]->player != 0){
        auto it = infoset_map.find(std::make_pair(nodes[i]->player, nodes[i]->infoset));
        if(it == infoset_map.end()){
            infoset_map[std::make_pair(nodes[i]->player, nodes[i]->infoset)] = ++infoset_num[nodes[i]->player];
            it = infoset_map.find(std::make_pair(nodes[i]->player, nodes[i]->infoset));
            infoset_names[nodes[i]->player].push_back(infoset_names_backup[nodes[i]->player][nodes[i]->infoset-1]);
        }
        nodes[i]->infoset = it -> second;
    }
}
