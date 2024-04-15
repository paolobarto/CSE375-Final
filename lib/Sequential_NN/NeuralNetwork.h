#include "Layer.h"
#include "../MNIST/MNIST_Image.h"
#include <math.h>

using namespace std;

class NeuralNetwork
{
private:
    Layer input;
    Layer output;
    vector<Layer> layers;
public:
    NeuralNetwork();
    void addLayer(Layer layer);
    void CreateNetwork(int inputSize,int hiddenLayerSize, int outputSize, int hiddenLayers);
    void ForwardPropagateImage(MNIST_Image img);
    void BackPropagateImage(MNIST_Image img);
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork()
{
    input = Layer();
    output = Layer();
    layers = vector<Layer>();
}


void NeuralNetwork::addLayer(Layer layer)
{
    layers.push_back(layer);
}

void NeuralNetwork::CreateNetwork(int inputSize, int hiddenLayerSize, int outputSize, int hiddenLayers)
{
    // Create input layer
    Layer inputLayer = Layer();
    for (int i = 0; i < inputSize; i++)
    {
        Node n = Node(i, 0, -3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5-(-3)))));
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
            Node n = Node(j, i + 1, -3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5-(-3)))));
            hiddenLayer.addNode(n);
        }
        layers.push_back(hiddenLayer);
    }

    // Create output layer
    Layer outputLayer = Layer();
    for (int i = 0; i < outputSize; i++)
    {
        Node n = Node(i, layers.size(), 1);
        outputLayer.addNode(n);
    }
    this->output = outputLayer;
}


void NeuralNetwork::ForwardPropagateImage(MNIST_Image img)
{
    float inputLayerTotal = 0;
    for(int i=0;i<img.size;i++)
    {
        inputLayerTotal += this->input.nodes[i].calculateOutput(img.pixels[i]/255, [](float x){return x;});
    }

    // iterating through layers
        float prevLayerTotal = 0;
        float currLayerTotal = 0;
    for(int i=0;i<this->layers.size();i++)
    {
        //need to calculate output for each layer
        // Since will be full connected NN, I can store the sum of the entire previous layer
        if(i==0)
            prevLayerTotal = inputLayerTotal;
        
        for(int j=0; j<layers[i].nodes.size(); j++)
        {
            // j = node index
            currLayerTotal+= layers[i].nodes[j].calculateOutput(prevLayerTotal, [](float x){return x>0 ? x : 0;});
        }
        prevLayerTotal = currLayerTotal;
    }

    // Calculate output with sigmoid
    for(int i=0;i<this->output.nodes.size();i++)
    {
        this->output.nodes[i].calculateOutput(prevLayerTotal, [](float x){return 1/(1+exp(-x));});
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
                    float partialEofW = partialEofY * derivativeSigmoid(currNode.value);
                    cout<<"Partial E of W: "<<partialEofW<<endl;
                }
                else
                {
                    // if current layer is hidden layer 
                    float partialEofW = partialEofY * derivativeSigmoid(currNode.weight);
                    // This covers the first part of the chain rule

                    int currentLayer = j;
                    // this is a layer before what we have covered
                    while(currentLayer<(this->layers.size()-1))
                    {

                    }




                    cout<<"Partial E of W: "<<partialEofW<<endl;
                }
                
                // TODO apply change in weight
                // TODO change representation of weights to display relationship
            }
        }

    }








}

float derivativeSigmoid(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

float sigmoid(float x)
{
    return 1/(1+exp(-x));
}



NeuralNetwork::~NeuralNetwork()
{
}
