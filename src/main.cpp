#include "../lib/Sequential_NN/NeuralNetwork.h"
#include "../lib/MNIST/image.h"

#include <fstream>
#include <iostream>
using namespace std;
int main()
{
    //read csv
    fstream testFile("../data/MNIST_CSV/mnist_test.csv", ios::in);
    if (!testFile.is_open())
    {
        cout << "Error: File not found" << endl;
        return 1;
    }
    fstream trainFile("../data/MNIST_CSV/mnist_train.csv", ios::in);
    if (!trainFile.is_open())
    {
        cout << "Error: File not found" << endl;
        return 1;
    }


    // vector<image> testImages;
    // vector<image> trainImages;

    // string line;
    // while(getline(testFile,line)) {
    //     testImages.push_back(image(line));
    // }
    // while(getline(trainFile,line)) {
    //     trainImages.push_back(image(line));
    // }
    // testFile.close();
    // cout<<testImages.size()<<endl;   
    // cout<<trainImages.size()<<endl;
    NeuralNetwork nn = NeuralNetwork();

    // 5 hidden layers of 784 nodes, relu activation, fully connected

    Layer inputLayer = Layer();
    for (int i = 0; i < 784; i++)
    {
        Node n = Node(i, 0, 0);
        inputLayer.addNode(n);
    }
    nn.addLayer(inputLayer);

    Layer hiddenLayer1 = Layer();
    for (int i = 0; i < 784; i++)
    {
        Node n = Node(i, 1, 0, inputLayer.getNodes());
        hiddenLayer1.addNode(n);
    }
    nn.addLayer(hiddenLayer1);


    Layer hiddenLayer2 = Layer();
    for (int i = 0; i < 784; i++)
    {
        Node n = Node(i, 2, 0, hiddenLayer1.getNodes());
        hiddenLayer2.addNode(n);
    }
    nn.addLayer(hiddenLayer2);

    Layer outputLayer = Layer();
    for (int i = 0; i < 10; i++)
    {
        Node n = Node(i, 3, 0, hiddenLayer2.getNodes());
        outputLayer.addNode(n);
    }
    nn.addLayer(outputLayer);

    return 0;
}