#include "Layer.h"
#include "../MNIST/MNIST_Image.h"
#include <math.h>
#include <iostream>

using namespace std;

class NeuralNetwork
{
private:
    Layer input;
    Layer output;
    vector<Layer> layers;
public:
    NeuralNetwork();
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    void ForwardPropagateImage(MNIST_Image img);
    void BackPropagateImage(MNIST_Image img);
    void AddLayer(Layer& layer);
    void UpdateLastHidden();
    float Sigmoid(float x);
    float DerivativeSigmoid(float x);
    float GetRandomWeight();
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork()
{
    input = Layer();
    output = Layer();
    layers = vector<Layer>();
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
        Layer previousLayer = i==0 ? this->input : layers[i-1];

        Layer hiddenLayer = Layer(previousLayer.nodes);
        for (int j = 0; j < hiddenLayerSize; j++)
        {
            Node n = Node(j, i + 1);
            hiddenLayer.addNode(n);
        }
        this->AddLayer(hiddenLayer);
    }

    // Create output layer
    Layer outputLayer = Layer();
    for (int i = 0; i < outputSize; i++)
    {
        Node n = Node(i, layers.size());
        outputLayer.addNode(n);
    }
    this->output = outputLayer;
    this->UpdateLastHidden();
}


void NeuralNetwork::ForwardPropagateImage(MNIST_Image img)
{
    // layer 1 in hidden layer inputs come from input layer

    //Layers
    for(int i=0;i<this->layers.size() + 1;i++)
    {
        Layer pastLayer = Layer();
        Layer currLayer = Layer();
        
        if(i==0){
            pastLayer = this->input;
            currLayer = this->layers[i];
        }
        else if(i==this->layers.size()){
            pastLayer =  this->layers[i-1];
            currLayer = this->output;
        }
        else {
            pastLayer = this->layers[i-1];
            currLayer = this->layers[i];
        }
        

            for(int j=0; j<pastLayer.nodes.size();j++)
            {
                for(int k=0; k<currLayer.nodes.size(); k++)
                {
                    currLayer.nodes[k].IncrementValue(pastLayer.nodes[j].value * pastLayer.nodes[j].weight[i]);
                }
            }

        if(i!=this->layers.size())
            for(int j=0; j<currLayer.nodes.size();j++)
                currLayer.nodes[j].ApplyRELU([](float x){return x>0?x:0;});
        else
             for(int j=0; j<currLayer.nodes.size();j++)
                currLayer.nodes[j].ApplyRELU([](float x){return 1/(1+exp(-x));});
    }

    


}

void NeuralNetwork::BackPropagateImage(MNIST_Image img)
{
    // Calculate error from output
    // formula = .5 * (target - output)^2
    float error = 0;
    for(int i=0;i<this->output.nodes.size();i++)
    {
        error += .5 * pow((img.label == i ? 1 : 0) - this->output.nodes[i].value, 2);
    }
    cout<<"Total Error: "<<error<<endl;

    //calculate error for node in hidden layer
    // partial derivative of error with respect to weight = 



    

    for(int i=0; i<output.nodes.size(); i++)
    {
        float partialEofY = this->output.nodes[i].value - (img.label == i ? 1 : 0);

        // if current layer is desired, return input, else return output of previous layer
        if(partialEofY==0)
            continue;

        for(int j=this->layers.size()-1; j>0; j--)
        {
            for(int k=0; k < this->layers[j].nodes.size(); k++){


                // amount to change weight of node
                Node currNode = this->layers[k].nodes[k];

                // calculating partial derivative of error with respect to weight
                if(j==this->layers.size()-1)
                {
                    // if current layer is output layer
                    float partialEofW = partialEofY * DerivativeSigmoid(currNode.value);
                    cout<<"Partial E of W: "<<partialEofW<<endl;
                }
                else
                {
                    // if current layer is hidden layer 
                    //float partialEofW = partialEofY * DerivativeSigmoid(currNode.weight);
                    // This covers the first part of the chain rule

                    int currentLayer = j;
                    // this is a layer before what we have covered
                    while(currentLayer<(this->layers.size()-1))
                    {

                    }




                    //cout<<"Partial E of W: "<<partialEofW<<endl;
                }
                
                // TODO apply change in weight
                // TODO change representation of weights to display relationship
            }
        }

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
            }
        }
    }
    else {
        Layer previousLayer = layers[layers.size()-1];

        for (int i = 0; i < previousLayer.nodes.size(); i++)
        {
            for (int j = 0; j < layer.nodes.size(); j++)
            {
                previousLayer.nodes[i].weight.push_back(this->GetRandomWeight());
            }
        }
        layers[layers.size()-1] = previousLayer;
    }


    layers.push_back(layer);
}

void NeuralNetwork::UpdateLastHidden()
{
    Layer previousLayer = layers[layers.size() - 1];

    for (int i = 0; i < previousLayer.nodes.size(); i++)
    {
        for (int j = 0; j < this->output.nodes.size(); j++)
        {
            previousLayer.nodes[i].weight.push_back(this->GetRandomWeight());
        }
    }
    layers[layers.size() - 1] = previousLayer;
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




NeuralNetwork::~NeuralNetwork()
{
}
