#pragma once

//#include <openssl/pem.h>
#include <vector>
#include <atomic>
#include "../common/protocol.h"
#include "../common/NeuralNetwork.h"
#include "../common/pool.h"

// Moving pool object
//bool handle_train(int sd, NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads);
bool handle_train(int sd, NeuralNetwork *network, NeuralNetwork *sumNetwork,vector<uint8_t> msg, thread_pool *pool);


//bool handle_update(int sd, NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads);
bool handle_update(int sd, NeuralNetwork *network, NeuralNetwork *sumNetwork, vector<uint8_t> msg, thread_pool *pool);
