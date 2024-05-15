#pragma once
#ifndef NeuralNetwork_H
#define NeuralNetwork_H

#include "Layer.h"
#include "MNIST_Image.h"
#include <math.h>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <cstring>

class NeuralNetwork
{
private:
    float LEARNING_RATE = 0.3;
    Layer input;
    Layer output;
    vector<Layer *> layers;
public:
    float previousTrainValue = 0;
    float maxIndex = 0;
    NeuralNetwork();
    NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize, int hiddenLayers, vector<vector<float>> inputWeights, vector<vector<float>> hiddenWeights, vector<vector<float>> outputWeights);
    NeuralNetwork(int inputSize, int hiddenLayerSize, int outputSize, int hiddenLayers);
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    int ForwardPropagateImage(MNIST_Image img);

    float FeedForward(MNIST_Image img);
    void Train(MNIST_Image img);
    vector<vector<float>> dotProduct(vector<vector<float>> a, vector<vector<float>> b);
    vector<vector<float>> transpose(vector<vector<float>> a);
    void PrintLoss(MNIST_Image img);
    int Predict(MNIST_Image img);
    int ArgMax(vector<float> vec);

    // int ForwardPropagateSequential(MNIST_Image img);
    // float BackPropagateImage(MNIST_Image img);
    // float BackPropagateRecursive(Node *target, float error, float runningInflunce, float runningChangeTotal);
    // void BackPropagateRecursiveArchive(Node *target, int error, float runningInflunce);
    // float BackPropagateLogging(Node *target, float error, float runningInflunce);
    void ResetValues();
    void AddLayer(Layer& layer);
    void UpdateLastHidden();
    void PrintLayerAverage();
    float Sigmoid(float x);
    float DerivativeSigmoid(float x);
    float DerivativeRELU(float x);
    float GetRandomWeight();

    vector<vector<float>> GetDistributedWeights(int inputSize, int outputSize);
    vector<float> GetDistributedBias(int size);
    vector<uint8_t> ExtractWeights(string message);
    void AddWeights(vector<uint8_t> msg);
    void AverageWeights(NeuralNetwork *sumNetwork, int numClients);
    void UpdateWeights(vector<uint8_t> msg);
    void ResetWeights();

    ~NeuralNetwork();
};


#endif