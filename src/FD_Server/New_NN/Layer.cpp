#include "Layer.h"

Layer::Layer(vector<Node *> previousNodes)
{
    nodes = vector<Node *>();
    weightMatrix = vector<vector<float> *>();
    bias = vector<float>();
    prevLayerNodes = previousNodes;
}

void Layer::ResetNodes()
{
    for(size_t i=0;i<this->nodes.size();i++)
        nodes[i]->ResetValue();
}

void Layer::addNode(Node *node)
{
    nodes.push_back(node);
    weightMatrix.push_back(node->weight);
}

Layer::~Layer()
{
}