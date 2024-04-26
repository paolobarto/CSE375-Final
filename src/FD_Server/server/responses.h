#pragma once

//#include <openssl/pem.h>
#include <vector>
#include <atomic>
#include "../common/protocol.h"
#include "../common/NeuralNetwork.h"

bool handle_train(NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads);

bool handle_update(NeuralNetwork *network, vector<uint8_t> msg, atomic<int> *trainingThreads);


