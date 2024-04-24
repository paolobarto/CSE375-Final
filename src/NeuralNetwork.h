#pragma once
#ifndef NeuralNetwork_H
#define NeuralNetwork_H

#include "Layer.h"
#include "MNIST_Image.h"
#include <math.h>
#include <iostream>
#include <random>
#include <cmath>


class NeuralNetwork
{
private:
    float LEARNING_RATE = 0.1;
    Layer input;
    Layer output;
    vector<Layer *> layers;
public:
    float previousTrainValue = 0;
    float maxIndex = 0;
    NeuralNetwork();
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    int ForwardPropagateImage(MNIST_Image img);
    int ForwardPropagateSequential(MNIST_Image img);
    float BackPropagateImage(MNIST_Image img);
    float BackPropagateRecursive(Node *target, float error, float runningInflunce, float runningChangeTotal);
    void BackPropagateRecursiveArchive(Node *target, int error, float runningInflunce);
    float BackPropagateLogging(Node *target, float error, float runningInflunce);
    void ResetValues();
    void AddLayer(Layer& layer);
    void UpdateLastHidden();
    void PrintLayerAverage();
    float Sigmoid(float x);
    float DerivativeSigmoid(float x);
    float DerivativeRELU(float x);
    float GetRandomWeight();
    vector<vector<float>> GetDistributedWeights(int inputSize, int outputSize);
    ~NeuralNetwork();
};


#endif