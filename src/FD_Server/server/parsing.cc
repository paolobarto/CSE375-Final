#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/socket.h>

#include "../common/contextmanager.h"
#include "../common/protocol.h"

#include "parsing.h"
#include "responses.h"

using namespace std;


/// When a new client connection is accepted, this code will run to figure out
/// what the client is requesting, and to dispatch to the right function for
/// satisfying the request.
///
/// @param sd      The socket on which communication with the client takes place
/// @param network The Neural Network of the server
///
/// @return true if the server should halt immediately, false otherwise
//bool parse_request(int sd, NeuralNetwork *network, atomic<int> *trainingThreads)

bool parse_request(int sd, NeuralNetwork *network, NeuralNetwork *sumNetwork,thread_pool *pool)
{
    cout<<"Parsing request\n";
    // get RKBlock off the socket
    vector<uint8_t> msg(8+LEN_DATABLOCK);

    // read from socket of size 8 + LEN_DATABLOCKß
    int dataConsumed = 0;
    // while(dataConsumed < 8+LEN_DATABLOCK){
    //     int bytesRead = read(sd, msg.data() + dataConsumed, 8+LEN_DATABLOCK - dataConsumed);
    //     if(bytesRead == -1){
    //         cout<<"Error reading from socket\n";
    //         return false;
    //     }
    //     dataConsumed += bytesRead;
    // }
    // cout<<"Read from socket\n";
    int BUFFER_SIZE = 1024;
    uint8_t buffer[BUFFER_SIZE];

    int bytesRead;
    while ((bytesRead = recv(sd, buffer, BUFFER_SIZE, 0)) > 0)
    {
        for (int i = 0; i < bytesRead; ++i){
            msg[dataConsumed++] = buffer[i];
        }
        if (dataConsumed >= 8 + LEN_DATABLOCK)
            break;
    }
    cout<<"Size of message Recieved: "<<dataConsumed<<endl;
    string cmd(msg.begin(), msg.begin() + 8);
    cout<<"\nCommand: "<<cmd<<endl;


    // Iterate through possible commands, pick the right one, run it
    vector<string> s = {REQ_TRAIN, REQ_UPDT};
    decltype(handle_train) *cmds[] = {handle_train, handle_update};
    for (size_t i = 0; i < s.size(); ++i)
        if (cmd == s[i])
            return cmds[i](sd, network,sumNetwork, msg, pool);

    return false;
}