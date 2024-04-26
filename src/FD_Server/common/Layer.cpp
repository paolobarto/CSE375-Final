#include "Layer.h"

Layer::Layer(vector<Node> previousNodes)
{
    nodes = vector<Node>();
    prevLayerNodes = previousNodes;
}

void Layer::ResetNodes()
{
    for(size_t i=0;i<this->nodes.size();i++)
        this->nodes[i].ResetValue();
}

void Layer::addNode(Node node)
{
    nodes.push_back(node);
}

Layer::~Layer()
{
}