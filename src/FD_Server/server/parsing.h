#pragma once

//#include <openssl/pem.h>
#include <atomic>
#include <vector>
#include "../common/NeuralNetwork.h"

/// When a new client connection is accepted, this code will run to figure out
/// what the client is requesting, and to dispatch to the right function for
/// satisfying the request.
///
/// @param sd      The socket on which communication with the client takes place
/// @param network The NeuralNetwork we are training with
///
/// @return true if the server should halt immediately, false otherwise
bool parse_request(int sd, NeuralNetwork *network, atomic<int> *trainingTreads);