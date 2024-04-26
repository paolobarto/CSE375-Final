#pragma once

#include <string>
#include <cmath>
/// protocol.h defines the messages that a client may send, and the responses a
/// server will provide.  Note that the entire request is a single byte stream,
/// as is the entire response.  The entire communication between client and
/// server should consist of just two messages.  First, the client sends a
/// message (the request), and then the server sends a reply message (the
/// response).
///
//
// Constants
//

const int LEN_INPUTNODES = 10;
const int LEN_HIDDENNODES = 16;
const int NUM_HIDDENLAYERS = 2;
const int LEN_OUTPUTNODES = 10;



// 10 nodes * 16 weights each = 160 Weights * (4 byte floats) = 640 bytes
const int LEN_INPUTBLOCKWEIGHTS = LEN_INPUTNODES * LEN_HIDDENNODES * sizeof(float);

// (HiddenLayer 1) 16 Nodes * 16 Weights each = 256 Weights * (4 byte floats) = 1024 bytes
const int LEN_HIDDENLAYERWEIGHTS = pow(LEN_HIDDENNODES, NUM_HIDDENLAYERS) * sizeof(float);

// (HiddenLayer 2) 16 Nodes * 10 Weights each = 160 Weights * (4 byte floats) = 640 bytes
const int LEN_OUTPUTLAYERWEIGHTS = LEN_HIDDENNODES * LEN_OUTPUTNODES * sizeof(float);

/// Length of an rblock or kblock
const int LEN_DATABLOCK = LEN_INPUTBLOCKWEIGHTS + LEN_HIDDENLAYERWEIGHTS + LEN_OUTPUTLAYERWEIGHTS;
//
// Request Messages

/// Request the server to return the neural network with the current data
const std::string REQ_TRAIN = "TRAIN___";

// Request the server to update the neural network with the given data
const std::string REQ_UPDT = "UPDATE__";

const std::string RESP_NETWORK = "NETWORK_";

const std::string RESP_UPDATE = "UPDATE__";

/// Request the server's public key (@pubkey), to use for subsequent interaction
/// with the server by the client
///
/// @kblock   pad0("PUB_KEY_")
/// @response @pubkey.<EOF>
/// @errors   None
//const std::string REQ_KEY = "PUB_KEY_";