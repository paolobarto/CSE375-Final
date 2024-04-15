#include <vector>
using namespace std;

class Node
{
private:
    int id;
    int layer;
    vector<Node> nextNodes;
    vector<Node> prevNodes;
public:
    float weight;
    float value = 0;
    Node();
    Node(int id, int layer, int weight);
    Node(int id, int layer, int weight, vector<Node> prevNodes);
    float calculateOutput(float input, float (*activationFunction)(float));
    ~Node();
};

Node::Node()
{
    id = 0;
    layer = 0;
    weight = 0;
    prevNodes = vector<Node>();
    nextNodes = vector<Node>();
}

Node::Node(int id, int layer, int weight)
{
    this->id = id;
    this->layer = layer;
    this->weight = weight;
    prevNodes = vector<Node>();
    nextNodes = vector<Node>();
}

Node::Node(int id, int layer, int weight, vector<Node> prevNodes)
{
    this->id = id;
    this->layer = layer;
    this->weight = weight;
    this->prevNodes = prevNodes;
    this->nextNodes = vector<Node>();
}


float Node::calculateOutput(float input, float (*activationFunction)(float))
{
    this->value =  activationFunction(input * weight);
    return this->value;
}

Node::~Node()
{
}
