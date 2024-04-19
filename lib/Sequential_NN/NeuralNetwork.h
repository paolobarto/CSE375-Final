#pragma once
#ifndef NeuralNetwork_H
#define NeuralNetwork_H

#include "Layer.h"
#include "../MNIST/MNIST_Image.h"
#include <math.h>
#include <iostream>


class NeuralNetwork
{
private:
    float LEARNING_RATE = 0.01;
    Layer input;
    Layer output;
    vector<Layer *> layers;
public:
    NeuralNetwork();
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    int ForwardPropagateImage(MNIST_Image img);
    void BackPropagateImage(MNIST_Image img);
    void BackPropagateRecursive(Node *target, int error, float runningInflunce);
    void AddLayer(Layer& layer);
    void UpdateLastHidden();
    float Sigmoid(float x);
    float DerivativeSigmoid(float x);
    float DerivativeRELU(float x);
    float GetRandomWeight();
    ~NeuralNetwork();
};


#endif