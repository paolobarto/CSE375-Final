#include <cassert>
#include <iostream>
#include <string>

#include "responses.h"

using namespace std;

// I technically dont need to pass the msg with weights becuase this request is just to get the weights
bool handle_train(NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads)
{
    assert(msg.size() != 0);
    cout<<"Network requested by client.\n";
    vector<uint8_t> resp = network->ExtractWeights(RESP_NETWORK);
    cout<<"Size of response: "<<resp.size()<<endl;

    trainingThreads->fetch_add(1);
    cout<<"Training threads: "<<trainingThreads->load()<<endl;
    // Verify that server is aware of how many training connections are there
   return true;
}


bool handle_update(NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads)
{
    cout<<"Network update requested by client.\n";
    assert(trainingThreads != nullptr);
    assert(network != nullptr);
    assert(msg.size() != 0);

    return true;
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