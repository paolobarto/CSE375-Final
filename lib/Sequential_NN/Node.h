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
    vector<float> weight;
    float value = 0;
    Node();
    Node(int id, int layer);
    Node(int id, int layer, vector<float> weight);
    Node(int id, int layer, vector<float> weight, vector<Node> prevNodes);
    float CalculateOutput(float input, int nextNodeIndex, float (*activationFunction)(float));
    void IncrementValue(float input);
    void ApplyRELU(float (*activationFunction)(float));
    void ResetValue();
    ~Node();
};

Node::Node()
{
    id = 0;
    layer = 0;
    weight = vector<float>();
    prevNodes = vector<Node>();
    nextNodes = vector<Node>();
}

Node::Node(int id, int layer)
{
    this->id = id;
    this->layer = layer;
    weight = vector<float>();
    prevNodes = vector<Node>();
    nextNodes = vector<Node>();
}

Node::Node(int id, int layer, vector<float> weight)
{
    this->id = id;
    this->layer = layer;
    this->weight = weight;
    prevNodes = vector<Node>();
    nextNodes = vector<Node>();
}

Node::Node(int id, int layer, vector<float> weight, vector<Node> prevNodes)
{
    this->id = id;
    this->layer = layer;
    this->weight = weight;
    this->prevNodes = prevNodes;
    this->nextNodes = vector<Node>();
}


float Node::CalculateOutput(float input, int nextNodeIndex, float (*activationFunction)(float))
{
    this->value =  activationFunction(input * weight[nextNodeIndex]);
   return this->value;
}

void Node::IncrementValue(float input)
{
    this->value+=input;
}

void Node::ApplyRELU(float (*activationFunction)(float))
{
    this->value = activationFunction(this->value);
}

void Node::ResetValue() 
{
    this->value = 0;
}

Node::~Node()
{
}
