#include "Layer.h"
using namespace std;

class NeuralNetwork
{
private:
    vector<float> input;
    vector<float> output;
    vector<Layer> layers;
public:
    NeuralNetwork();
    NeuralNetwork(vector<float> input, vector<float> output, vector<Layer> layers);
    
    void addLayer(Layer layer);
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork()
{
    input = vector<float>();
    output = vector<float>();
    layers = vector<Layer>();
}

NeuralNetwork::NeuralNetwork(vector<float> input, vector<float> output, vector<Layer> layers)
{
    this->input = input;
    this->output = output;
    this->layers = layers;
}

void NeuralNetwork::addLayer(Layer layer)
{
    layers.push_back(layer);
}

NeuralNetwork::~NeuralNetwork()
{
}
