#include <vector>
#include "Node.h"
using namespace std;

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
