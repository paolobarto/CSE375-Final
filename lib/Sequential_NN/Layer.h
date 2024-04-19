#pragma once
#ifndef Layer_H
#define Layer_H

#include "Node.h"

class Layer
{
private:
    // Since we can assume will be fully connected, we can just store the nodes of the previous layer
    vector<Node> prevLayerNodes;
public:
    vector<Node> nodes;
    Layer(vector<Node> previousNodes = vector<Node>());
    void addNode(Node node);
    void appendLayer(Layer layer);
    ~Layer();
};

#endif