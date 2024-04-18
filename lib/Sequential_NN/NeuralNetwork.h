#include "Layer.h"
#include "../MNIST/MNIST_Image.h"
#include <math.h>
#include <iostream>

using namespace std;
float LEARNING_RATE = 0.01;

class NeuralNetwork
{
private:
    Layer input;
    Layer output;
    vector<Layer *> layers;
public:
    NeuralNetwork();
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    int ForwardPropagateImage(MNIST_Image img);
    void BackPropagateImage(MNIST_Image img);
    void BackPropagateRecursive(Node *target, int error, float runningInflunce);
    void AddLayer(Layer& layer);
    void UpdateLastHidden();
    float Sigmoid(float x);
    float DerivativeSigmoid(float x);
    float DerivativeRELU(float x);
    float GetRandomWeight();
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork()
{
    input = Layer();
    output = Layer();
    layers = vector<Layer *>();
}


void NeuralNetwork::CreateNetwork(int inputSize, int hiddenLayerSize, int outputSize, int hiddenLayers)
{
    // Create input layer
    Layer inputLayer = Layer();
    for (int i = 0; i < inputSize; i++)
    {
        Node n = Node(i, 0);
        inputLayer.addNode(n);
    }
    this->input = inputLayer;


    // Create hidden layers
    for (int i = 0; i < hiddenLayers; i++)
    {
        Layer *previousLayer = i==0 ? &this->input : layers[i-1];

        Layer *hiddenLayer = new Layer((*previousLayer).nodes);
        for (int j = 0; j < hiddenLayerSize; j++)
        {
            Node n = Node(j, i + 1);
            hiddenLayer->addNode(n);
            
        }
        this->AddLayer(*hiddenLayer);
    }

    // Create output layer
    Layer outputLayer = Layer();
    for (int i = 0; i < outputSize; i++)
    {
        Node n = Node(i, layers.size()+1);
        outputLayer.addNode(n);
    }
    this->output = outputLayer;
    this->UpdateLastHidden();
}


int NeuralNetwork::ForwardPropagateImage(MNIST_Image img)
{
    // layer 1 in hidden layer inputs come from input layer
    for(int i=0; i<img.size;i++)
    {
        this->input.nodes[i].value = img.pixels[i];
    }

    //Layers
    for(int i=0;i<this->layers.size() + 1;i++)
    {
        Layer pastLayer = Layer();
        Layer currLayer = Layer();
        
        if(i==0){
            pastLayer = this->input;
            currLayer = *this->layers[i];
        }
        else if(i==this->layers.size()){
            pastLayer =  *this->layers[i-1];
            currLayer = this->output;
        }
        else {
            pastLayer = *this->layers[i-1];
            currLayer = *this->layers[i];
        }
        

        for(int j=0; j<pastLayer.nodes.size();j++)
            for(int k=0; k<currLayer.nodes.size(); k++)
                currLayer.nodes[k].IncrementValue(pastLayer.nodes[j].value * pastLayer.nodes[j].weight[i]);
            

        if(i!=this->layers.size())
            for(int j=0; j<currLayer.nodes.size();j++)
                currLayer.nodes[j].ApplyRELU([](float x){return x>0?x:0;});
        else
             for(int j=0; j<currLayer.nodes.size();j++)
                currLayer.nodes[j].ApplyRELU([](float x){return 1/(1+exp(-x));});
    }

    return img.label;

}

void NeuralNetwork::BackPropagateImage(MNIST_Image img)
{
    
    // Error impact
    int vsTarget[10];

    for(int i=0; i<10; i++)
    {
        if(i == img.label)
        {
            if(this->output.nodes[i].value == 1)
                vsTarget[i]==0;
            else  
                vsTarget[i]=-1;
        }
        else
        {
            if(this->output.nodes[i].value == 0)
                vsTarget[i] = 0;
            else
                vsTarget[i] = 1;
        }
    }

    for(int i=0; i<this->output.nodes.size(); i++)
    {
        if(vsTarget[i] != 0)
            this->BackPropagateRecursive(&this->output.nodes[i], vsTarget[i], 1);
    }
}

void NeuralNetwork::BackPropagateRecursive(Node *target, int error, float runningInfluence)
{
    for(int i=0; i<target->prevNodes.size(); i++)
    {
        // Calculate the partial derivative of the error with respect to the weight
        float partialDeriv = error;
        float influence = runningInfluence;
        //Given a node, calculate contributing influence then repeat for previous nodes
        if(target->layer == this->layers.size()+1)
        {
            // This is the output layer
            // Calculate the derivative of the sigmoid
            partialDeriv *= this->DerivativeSigmoid(target->prevNodes[i]->value);
            // This is used in all subsequent calculations
            influence = this->DerivativeSigmoid(target->prevNodes[i]->weight[target->id]);
        }
        else
        {
            // This is a hidden layer
            // Calculate the derivative of the RELU
            partialDeriv *= this->DerivativeRELU(target->prevNodes[i]->value);
            // This is used in all subsequent calculations
            influence *= this->DerivativeRELU(target->prevNodes[i]->weight[target->id]);
        }

        // Update the weight
        target->prevNodes[i]->weight[target->id] -= LEARNING_RATE * partialDeriv * influence;
        // Recurse
        this->BackPropagateRecursive(target->prevNodes[i], partialDeriv, influence);
    }
}

void NeuralNetwork::AddLayer(Layer& layer)
{

    if(layers.size() == 0)
    {
        // This is the first layer after the input layer
        // Connect the input layer to this layer
        for(int i=0; i<this->input.nodes.size();i++) {
            for(int j=0;j<layer.nodes.size();j++){
                this->input.nodes[i].weight.push_back(this->GetRandomWeight());
                //this->input.nodes[i].prevNodes.push_back(layer.nodes[j]);
                layer.nodes[j].prevNodes.push_back(&this->input.nodes[i]);
            }
        }
    }
    else {
        Layer *previousLayer = layers[layers.size()-1];

        for (int i = 0; i < previousLayer->nodes.size(); i++)
        {
            for (int j = 0; j < layer.nodes.size(); j++)
            {
                previousLayer->nodes[i].weight.push_back(this->GetRandomWeight());
                //this->input.nodes[i].prevNodes.push_back(layer.nodes[j]);
                layer.nodes[j].prevNodes.push_back(&previousLayer->nodes[i]);
            }
        }
        //layers[layers.size()-1] = &previousLayer;
    }


    layers.push_back(&layer);
}

void NeuralNetwork::UpdateLastHidden()
{
    Layer *previousLayer = this->layers[layers.size() - 1];

    for (int i = 0; i < previousLayer->nodes.size(); i++)
    {
        for (int j = 0; j < this->output.nodes.size(); j++)
        {
            previousLayer->nodes[i].weight.push_back(this->GetRandomWeight());
            this->output.nodes[j].prevNodes.push_back(&previousLayer->nodes[i]);
        }
    }
    //layers[layers.size() - 1] = previousLayer;
}

float NeuralNetwork::GetRandomWeight()
{
    return -1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5-(-3))));
}


float NeuralNetwork::Sigmoid(float x)
{
    return 1/(1+exp(-x));
}

float NeuralNetwork::DerivativeSigmoid(float x)
{
    return Sigmoid(x) * (1 - Sigmoid(x));
}

float NeuralNetwork::DerivativeRELU(float x)
{
    return x>0?1:0;
}



NeuralNetwork::~NeuralNetwork()
{
}
