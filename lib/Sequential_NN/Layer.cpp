#include "Layer.h"

Layer::Layer(vector<Node> previousNodes)
{
    nodes = vector<Node>();
    prevLayerNodes = previousNodes;
}

void Layer::addNode(Node node)
{
    nodes.push_back(node);
}

Layer::~Layer()
{
}