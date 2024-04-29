#include <cassert>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <mutex>

#include "responses.h"

using namespace std;

// I technically dont need to pass the msg with weights becuase this request is just to get the weights
//bool handle_train(int sd, NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads)
bool handle_train(int sd, NeuralNetwork *network, NeuralNetwork *sumNetwork, vector<uint8_t> msg, thread_pool *pool)
{
    assert(msg.size() != 0);
    assert(sumNetwork != nullptr);
    cout<<"Network requested by client.\n";

    pool->modelLock.lock_shared();
    vector<uint8_t> resp = network->ExtractWeights(RESP_NETWORK);
    pool->modelLock.unlock_shared();

    cout<<"Size of response: "<<resp.size()<<endl;

    // Address of training threads is broken
    cout<<"Amount of running threads: "<<pool->trainingTreads->load()<<endl;

    pool->trainingTreads->fetch_add(1);
    cout<<"Training threads: "<<pool->trainingTreads->load()<<endl;
    // Verify that server is aware of how many training connections are there
    cout<<"Sending response to client of size: "<<resp.size()<<"\n";

    bool success = send(sd, resp.data(), LEN_DATABLOCK + 8, 0);
    if(success)
    {
        cout<<"Response sent to client.\n";
    }
    else
    {
        cout<<"Error sending response to client.\n";
    }

    return !success;
}

//bool handle_update(int sd, NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads)

bool handle_update(int sd, NeuralNetwork *network, NeuralNetwork *sumNetwork, vector<uint8_t> msg, thread_pool *pool)
{
    cout<<"Network update requested by client.\n";
    pool->modelLock.lock();
    sumNetwork->AddWeights(msg);
    pool->modelLock.unlock();
    cout<<"Weights added to sum network.\n";


    pool->respondedThreads->fetch_add(1);
    cout<<"Responded Threads: "<<pool->respondedThreads->load()<<endl;

    // Send the updated weights to the server update network

    unique_lock<mutex> lk(pool->updateLock);
    pool->updateCon.wait(lk, [&](){return pool->respondedThreads->load() == 0;});

    vector<uint8_t> resp = network->ExtractWeights(RESP_UPDATE);
    cout<<"Size of response: "<<resp.size()<<endl;

    bool success = send(sd, resp.data(), LEN_DATABLOCK + 8, 0);
    if (success)
    {
        cout << "Response sent to client.\n";
    }
    else
    {
        cout << "Error sending response to client.\n";
    }

    return !success;
}

// This function returns a NN object that I can use to more conveniently access the weights
NeuralNetwork CreateWeightNetwork(vector<uint8_t> msg)
{
    vector<vector<float>> inputWeights;
    vector<vector<float>> hiddenWeights;
    vector<vector<float>> outputWeights;

    int byteIndex = 8;
    
    for(int i=0; i<LEN_INPUTNODES; i++)
    {
        vector<float> inputWeightNode;
        for(int j=0; j<LEN_HIDDENNODES; j++)
        {
            float weight = 0;
            memcpy(&weight, msg.data() + byteIndex, sizeof(float));
            byteIndex += sizeof(float);
            inputWeightNode.push_back(weight);
        }
    }

    for(int i=0; i<NUM_HIDDENLAYERS-1; i++)
    {
        for(int j=0; j<LEN_HIDDENNODES; j++)
        {
            vector<float> hiddenWeightNode;
            for(int k=0; k<LEN_HIDDENNODES; k++)
            {
                float weight = 0;
                memcpy(&weight, msg.data() + byteIndex, sizeof(float));
                byteIndex += sizeof(float);
                hiddenWeightNode.push_back(weight);
            }
            hiddenWeights.push_back(hiddenWeightNode);
        }
    }

    for(int i=0; i<LEN_HIDDENNODES; i++)
    {
        vector<float> outputWeightNode;
        for(int j=0; j<LEN_OUTPUTNODES; j++)
        {
            float weight = 0;
            memcpy(&weight, msg.data() + byteIndex, sizeof(float));
            byteIndex += sizeof(float);
            outputWeightNode.push_back(weight);
        }
        outputWeights.push_back(outputWeightNode);
    }

    NeuralNetwork network(LEN_INPUTNODES, LEN_HIDDENNODES, LEN_OUTPUTNODES, NUM_HIDDENLAYERS, inputWeights, hiddenWeights, outputWeights);
    return network;
}