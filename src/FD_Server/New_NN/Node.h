#pragma once
#ifndef Node_H
#define Node_H

#include <vector>
using namespace std;

class Node
{
private:
    vector<Node> nextNodes;
public:
    int id;
    int layer;
    vector<Node *> prevNodes;
    vector<float> *weight;
    float value = 0;
    float trainingTotal = 0;
    float backPropIntermediateValue = 0;
    float priorActivationValue = 0;
    Node();
    Node(int id, int layer);
    Node(int id, int layer, vector<float> *weight);
    Node(int id, int layer, vector<float> *weight, vector<Node *> prevNodes);
    float CalculateOutput(float input, int nextNodeIndex, float (*activationFunction)(float));
    void IncrementValue(float input);
    void ApplyRELU(float (*activationFunction)(float));
    void ApplySigmoid(float (*activationFunction)(float));
    void ResetValue();
    ~Node();
};


#endif