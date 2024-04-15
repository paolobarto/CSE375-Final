#include "../lib/Sequential_NN/NeuralNetwork.h"


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
    // fstream trainFile("../data/MNIST_CSV/mnist_train.csv", ios::in);
    // if (!trainFile.is_open())
    // {
    //     cout << "Error: File not found" << endl;
    //     return 1;
    // }


     vector<MNIST_Image> testImages;
    // vector<image> trainImages;

    string line;
    while(getline(testFile,line)) {
        testImages.push_back(MNIST_Image(line));
    }
    // while(getline(trainFile,line)) {
    //     trainImages.push_back(image(line));
    // }
    // trainFile.close();
    testFile.close();
    // cout<<testImages.size()<<endl;   
    // cout<<trainImages.size()<<endl;
    NeuralNetwork nn = NeuralNetwork();

    // 5 hidden layers of 784 nodes, relu activation, fully connected

    nn.CreateNetwork(784, 16, 10, 2);

    nn.ForwardPropagateImage(testImages[0]);



    return 0;
}