#include "NeuralNetwork.h"


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


int NeuralNetwork::ForwardPropagateSequential(MNIST_Image img)
{
    // Reset values in input layer
    this->input.ResetNodes();

    // Set pixel values in input layer nodes
    for (int i = 0; i < img.size; i++)
    {
        input.nodes[i].value = img.pixels[i];
    }

    // Forward propagate through hidden layers
    for (int i = 0; i < layers.size(); i++)
    {
        // Iterate over nodes in current layer
        for (int j = 0; j < layers[i]->nodes.size(); j++)
        {
            Node &node = layers[i]->nodes[j];
            float sum = 0;

            // Iterate over nodes in previous layer
            for (int k = 0; k < node.prevNodes.size(); k++)
            {
                // Calculate weighted sum
                sum += node.weight[k] * node.prevNodes[k]->value;
            }

            // Apply activation function
            node.value = Sigmoid(sum);
        }
    }

    // Forward propagate through output layer
    for (int i = 0; i < output.nodes.size(); i++)
    {
        Node &node = output.nodes[i];
        float sum = 0;

        // Iterate over nodes in previous layer
        for (int j = 0; j < node.prevNodes.size(); j++)
        {
            // Calculate weighted sum
            sum += node.prevNodes[j]->weight[i] * node.prevNodes[j]->value;
        }

        // Apply activation function
        node.value = Sigmoid(sum);
    }

    // Find index of node with highest value in output layer (classification)
    int maxIndex = 0;
    float maxValue = output.nodes[0].value;
    for (int i = 1; i < output.nodes.size(); i++)
    {
        if (output.nodes[i].value > maxValue)
        {
            maxIndex = i;
            maxValue = output.nodes[i].value;
        }
    }

    return maxIndex;
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
        Layer *pastLayer = new Layer();
        Layer *currLayer = new Layer();

        if(i==0){
            pastLayer = &this->input;
            currLayer = this->layers[i];
        }
        else if(i==this->layers.size()){
            pastLayer =  this->layers[i-1];
            currLayer = &this->output;
        }
        else {
            pastLayer = this->layers[i-1];
            currLayer = this->layers[i];
        }


        for(int j=0; j<pastLayer->nodes.size();j++)
            for(int k=0; k<currLayer->nodes.size(); k++){
                float sum = pastLayer->nodes[j].value * pastLayer->nodes[j].weight[k];
                if(isnan(sum))
                    cout<<"NAN"<<endl;
                currLayer->nodes[k].IncrementValue(sum);
                //cout<<pastLayer->nodes[j].value<<"*"<<pastLayer->nodes[j].weight[k]<<" = "<<currLayer->nodes[k].value<<endl;
            }


        if(i!=this->layers.size())
            for(int j=0; j<currLayer->nodes.size();j++){
                //currLayer->nodes[j].ApplyRELU([](float x){return x>0?x:0;});
                currLayer->nodes[j].ApplyRELU([](float x){return x>0?x:(float)(0.1*x);}); // leakyRelu
                //cout<<"RELU Before: "<<currLayer->nodes[j].priorActivationValue<<" After: "<<currLayer->nodes[j].value<<endl;
            }
        else
             for(int j=0; j<currLayer->nodes.size();j++){
                currLayer->nodes[j].ApplySigmoid([](float x){return (1/(1+exp(-x)));});
                //cout<<"Sigmoid Before: "<<currLayer->nodes[j].priorActivationValue<<" After: "<<currLayer->nodes[j].value<<endl;
             }
    }

    for(int i=0;i<10;i++)
    {
        if(this->output.nodes[i].value == 1 && i!=img.label)
            return -1;
        else if(this->output.nodes[i].value == 0 && i==img.label)
            return -1;
    }
    
    //argmax
    float max = .5;
    int index = -1;
    for(int i=0; i<10; i++)
    {
        if(this->output.nodes[i].value>max)
        {
            max = this->output.nodes[i].value;
            index = i;
        }
    }

    return index;

}

float NeuralNetwork::BackPropagateImage(MNIST_Image img)
{

    // Error impact
    float vsTarget[10];
    float delta = .0000001;
    for(int i=0; i<10; i++)
    {
        float y = i==img.label?1:0;
        float y_hat = this->output.nodes[i].value;
        // float x = y * log(y_hat + delta);
        // float x2 = (1 - y);
        // float x3 = log((1 - y_hat) + delta);
        // vsTarget[i] = -(x + x2 * x3);
        vsTarget[i] = y - y_hat;
    }

    // for (int i = 0; i < 10; i++)
    // {
    //     if (i == img.label)
    //     {
    //         if (this->output.nodes[i].value == 1)
    //             vsTarget[i] = 0;
    //         else
    //             vsTarget[i] = -1;
    //     }
    //     else
    //     {
    //         if (this->output.nodes[i].value == 0)
    //             vsTarget[i] = 0;
    //         else
    //             vsTarget[i] = 1;
    //     }
    // }

    float runningChangeTotal = 0;
    // vector<vector<vector<float>>> debugPrevLayerWeights;
    // for(int i=0; i<this->layers.size(); i++)
    // {
    //     vector<vector<float>> layer;
    //     for(int j=0; j<this->layers[i]->nodes.size(); j++)
    //     {   
    //         vector<float> node;
    //         for(int k=0; k<this->layers[i]->nodes[j].weight.size(); k++)
    //         {
    //             node.push_back(this->layers[i]->nodes[j].weight[k]);
    //         }
    //         layer.push_back(node);
    //     }
    //     debugPrevLayerWeights.push_back(layer);
    // }

    for (int i = 0; i < this->output.nodes.size(); i++)
    {
        if(vsTarget[i] != 0){
            //runningChangeTotal += this->BackPropagateRecursive(&this->output.nodes[i], vsTarget[i], 1, 0);;
            //this->BackPropagateRecursiveArchive(&this->output.nodes[i], vsTarget[i], 1);
            runningChangeTotal += this->BackPropagateLogging(&this->output.nodes[i], vsTarget[i], 1);
        }
    }
    ////cout<<"Backprop changeTotal: "<<runningChangeTotal<<endl;
    // if(runningChangeTotal==this->previousTrainValue)
    //     this->PrintLayerAverage();
    // else
    //     this->previousTrainValue=runningChangeTotal;

    // for(int i=0; i<this->layers.size(); i++)
    // {
    //     for(int j=0; j<this->layers[i]->nodes.size(); j++)
    //     {
    //         for(int k=0; k<this->layers[i]->nodes[j].weight.size(); k++)
    //         {
    //             if(debugPrevLayerWeights[i][j][k] != this->layers[i]->nodes[j].weight[k])
    //             {
    //                 return runningChangeTotal;
    //             }
    //         }
    //     }
    // }


    // cout<<"REDUNDANT TRAINING"<<endl;
    // this->PrintLayerAverage();
    // for (int i = 0; i < this->output.nodes.size(); i++)
    // {
    //     if (vsTarget[i] != 0)
    //     {
    //         this->BackPropagateLogging(&this->output.nodes[i], vsTarget[i], 1);
            
    //         // this->BackPropagateRecursiveArchive(&this->output.nodes[i], vsTarget[i], 1);
    //     }
    // }

    return runningChangeTotal;
}


float NeuralNetwork::BackPropagateLogging(Node *target, float error, float runningInfluence)
{

    // Base case: if the target node has no previous nodes, return the error
    if (target->prevNodes.empty())
    {
        return error;
    }

    // Calculate the derivative of the activation function at the target node's value
    float derivative;
    if (target->layer != this->layers.size() + 1)
    {
        // Hidden layer: ReLU activation
        derivative = DerivativeRELU(target->value);
    }
    else
    {
        // Last hidden layer: Sigmoid activation
        derivative = DerivativeSigmoid(target->value);
    }

    // Calculate the change in the target node's value
    float delta = error * derivative;
    if(delta==0)
        return 0;

    // Update weights and propagate error to previous layer nodes
    for (int i = 0; i < target->prevNodes.size(); i++)
    {
        // Calculate the change in weight
        float weightChange = delta * target->prevNodes[i]->value * LEARNING_RATE;

        // Update the weight
        target->prevNodes[i]->weight[target->id] += weightChange;
        if(target->layer==1)
        {
            continue;
        }

        // Recursively propagate the error to the previous node
        //runningInfluence = runningInfluence * target->prevNodes[i]->weight[target->id];
        float prevError = delta * target->prevNodes[i]->weight[target->id] * runningInfluence;
        BackPropagateLogging(target->prevNodes[i], prevError, runningInfluence * target->prevNodes[i]->weight[target->id]);
    }

    // Return the error propagated to the previous layer
    return delta * runningInfluence;
}

float NeuralNetwork::BackPropagateRecursive(Node *target, float error, float runningInfluence, float runningChangeTotal)
{

    for(int i=0; i<target->prevNodes.size(); i++)
    {
        // Calculate the partial derivative of the error with respect to the weight
        float partialDeriv = error;
        float influence = runningInfluence;
        float changeTotal = 0;
        //Given a node, calculate contributing influence then repeat for previous nodes
        if(target->layer == this->layers.size()+1)
        {
            // This is the output layer
            // Calculate the derivative of the sigmoid
            partialDeriv *= this->DerivativeSigmoid(target->prevNodes[i]->value);
            // same thing as 
            //float samePartial = target->value * (1 - target->value);


            // This is used in all subsequent calculations
            influence = this->DerivativeSigmoid(target->prevNodes[i]->weight[target->id]);

            changeTotal = this->LEARNING_RATE * partialDeriv;
            target->prevNodes[i]->weight[target->id] -= changeTotal;
            // if(influence==0) // partialDerive may be zero
            //     return runningChangeTotal + changeTotal;
        }
        else
        {
            // This is a hidden layer
            // Calculate the derivative of the RELU
            partialDeriv *= this->DerivativeRELU(target->prevNodes[i]->value);
            //partialDeriv *= this->DerivativeSigmoid(target->prevNodes[i]->value);
            // This is used in all subsequent calculations

            changeTotal = this->LEARNING_RATE * partialDeriv * runningInfluence;
            // float prevWeight = target->prevNodes[i]->weight[target->id];
            // if(prevWeight-changeTotal<=0)
            //     target->prevNodes[i]->weight[target->id]=.00000001; // this is floor value, if set to zero will not train previous layer
            // else
                target->prevNodes[i]->weight[target->id] -= changeTotal;

            // Woukd setting a floor be an option?
            // if(target->prevNodes[i]->weight[target->id]<0)
            //     cout<<"here"<<endl;


            influence *= this->DerivativeRELU(target->prevNodes[i]->weight[target->id]);
            //influence = this->DerivativeSigmoid(target->prevNodes[i]->weight[target->id]);
        }
        if(influence==0) // partialDerive may be zero
            //return runningChangeTotal + changeTotal;
            {
                runningChangeTotal += changeTotal;
                continue;
            }

        if(target->prevNodes[i]->layer==0)
            continue;

        // Update the weight
        // Recurse
        runningChangeTotal += this->BackPropagateRecursive(target->prevNodes[i], error, influence, changeTotal+runningChangeTotal);
    }
    return runningChangeTotal;
}

void NeuralNetwork::BackPropagateRecursiveArchive(Node *target, int error, float runningInfluence)
{
    for (int i = 0; i < target->prevNodes.size(); i++)
    {
        // Calculate the partial derivative of the error with respect to the weight
        float partialDeriv = error;
        float influence = runningInfluence;
        // Given a node, calculate contributing influence then repeat for previous nodes
        if (target->layer == this->layers.size() + 1)
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
        target->prevNodes[i]->weight[target->id] -= this->LEARNING_RATE * partialDeriv * runningInfluence; //DIFERENT FROM ARCHIVED
         //target->prevNodes[i]->weight[target->id] -= this->LEARNING_RATE * partialDeriv * runningInfluence;

        if (influence == 0 || target->prevNodes[i]->layer==0){

            continue;
        } // This is literally all I needed for my code to work

        // Update the weight

        // Recurse
        //this->BackPropagateRecursiveArchive(target->prevNodes[i], partialDeriv, influence);
        this->BackPropagateRecursiveArchive(target->prevNodes[i], error, influence);
    }
}
        

void NeuralNetwork::ResetValues()
{
    this->input.ResetNodes();

    for(int i=0; i<this->layers.size(); i++)
        this->layers[i]->ResetNodes();

    this->output.ResetNodes();
}


void NeuralNetwork::AddLayer(Layer& layer)
{

    if(layers.size() == 0)
    {
        // This is the first layer after the input layer
        // Connect the input layer to this layer
        vector<vector<float>> distrubutedWeights = this->GetDistributedWeights(this->input.nodes.size(), layer.nodes.size());

        for(int i=0; i<this->input.nodes.size();i++) {
            for(int j=0;j<layer.nodes.size();j++){
                this->input.nodes[i].weight.push_back(distrubutedWeights[i][j]);
                //this->input.nodes[i].prevNodes.push_back(layer.nodes[j]);
                layer.nodes[j].prevNodes.push_back(&this->input.nodes[i]);
            }
        }
    }
    else {
        Layer *previousLayer = layers[layers.size()-1];

        vector<vector<float>> distrubutedWeights = this->GetDistributedWeights(previousLayer->nodes.size(), layer.nodes.size());


        for (int i = 0; i < previousLayer->nodes.size(); i++)
        {
            for (int j = 0; j < layer.nodes.size(); j++)
            {
                previousLayer->nodes[i].weight.push_back(distrubutedWeights[i][j]);
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

    vector<vector<float>> distrubutedWeights = this->GetDistributedWeights(previousLayer->nodes.size(), this->output.nodes.size());


    for (int i = 0; i < previousLayer->nodes.size(); i++)
    {
        for (int j = 0; j < this->output.nodes.size(); j++)
        {
            previousLayer->nodes[i].weight.push_back(distrubutedWeights[i][j]);
            this->output.nodes[j].prevNodes.push_back(&previousLayer->nodes[i]);
        }
    }
    //layers[layers.size() - 1] = previousLayer;
}

float NeuralNetwork::GetRandomWeight()
{
    // min = 0, max = 1
    return 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1-(-1))));
}


vector<vector<float>> NeuralNetwork::GetDistributedWeights(int inputSize, int outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0, sqrt(2.0 / (inputSize + outputSize)));

    std::vector<std::vector<float>> weights(inputSize, std::vector<float>(outputSize));

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            
            weights[i][j] = distribution(gen);
        }
    }

    return weights;

}

void NeuralNetwork::PrintLayerAverage()
{
    float sum = 0;
    float weightSum = 0;
    for(int i=0; i<this->input.nodes.size();i++)
    {
        sum+=this->input.nodes[i].value;
        for(int j=0; j<this->input.nodes[i].weight.size();j++)
        {
            weightSum+=this->input.nodes[i].weight[j];
        }
    }
    cout<<"Input Layer Value Average: "<<sum/this->input.nodes.size()<<endl;
    cout<<"Input Layer Weight Average: "<<weightSum/(this->input.nodes.size()*this->input.nodes[0].weight.size())<<endl;

    for(int i=0; i<this->layers.size();i++)
    {
        sum = 0;
        for(int j=0; j<this->layers[i]->nodes.size();j++)
        {
            sum+=this->layers[i]->nodes[j].value;
            cout<<"("<<this->layers[i]->nodes[j].value<<","<<this->layers[i]->nodes[j].priorActivationValue<<"),";
            for(int k=0; k<this->layers[i]->nodes[j].weight.size();k++)
            {
                weightSum+=this->layers[i]->nodes[j].weight[k];
            }
            
        }
        cout<<endl;
        cout<<"Hidden Layer "<<i+1<<" Average: "<<sum/this->layers[i]->nodes.size()<<endl;
        cout<<"Hidden Layer "<<i+1<<" Weight Average: "<<weightSum/(this->layers[i]->nodes.size()*this->layers[i]->nodes[0].weight.size())<<endl;
    }

    sum = 0;
    for(int i=0; i<this->output.nodes.size();i++)
    {
        sum+=this->output.nodes[i].priorActivationValue;
    }
    cout<<"Output Layer Average: "<<sum/this->output.nodes.size()<<endl;
    cout<<"-------------------------"<<endl<<endl;
}


float NeuralNetwork::Sigmoid(float x)
{
    return (1/(1+exp(-x)));
}

float NeuralNetwork::DerivativeSigmoid(float x)
{
    return Sigmoid(x) * (1 - Sigmoid(x));
}

float NeuralNetwork::DerivativeRELU(float x)
{
    //return x>0?1:0;
    return x>0?1:(float)(0.1);
}





NeuralNetwork::~NeuralNetwork()
{
}