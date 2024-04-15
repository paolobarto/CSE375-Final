#include <vector>
#include "Node.h"
using namespace std;

class Layer
{
private:
    vector<Node> nodes;
public:
    Layer();
    Layer(vector<Node> nodes);
    void addNode(Node node);
    vector<Node> getNodes() { return nodes; }
    void appendLayer(Layer layer);
    ~Layer();
};

Layer::Layer()
{
    nodes = vector<Node>();
}

Layer::Layer(vector<Node> nodes)
{
    this->nodes = nodes;
}

void Layer::addNode(Node node)
{
    nodes.push_back(node);
}

Layer::~Layer()
{
}
