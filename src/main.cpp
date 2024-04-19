#include "../lib/Sequential_NN/NeuralNetwork.h"


#include <fstream>


int main(int argc, char *argv[])
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


    vector<MNIST_Image> testImages;
    vector<MNIST_Image> trainImages;

    string line;
    while(getline(testFile,line)) {
        testImages.push_back(MNIST_Image(line));
    }
    while(getline(trainFile,line)) {
        trainImages.push_back(MNIST_Image(line));
    }
    trainFile.close();
    testFile.close();
    cout<<testImages.size()<<endl;   
    cout<<trainImages.size()<<endl;
    NeuralNetwork nn = NeuralNetwork();

    // 5 hidden layers of 784 nodes, relu activation, fully connected

    nn.CreateNetwork(784, 16, 10, 2);

    for(int i=0;i<trainImages.size();i++)
    {
        nn.ForwardPropagateImage(trainImages[i]);
        nn.BackPropagateImage(trainImages[i]);
    }

    int correctGuesses = 0;
    int falseGuesses = 0;

    for(int i=0; i<testImages.size(); i++)
    {
        int guess = nn.ForwardPropagateImage(testImages[i]);
        if(guess == testImages[i].label)
            correctGuesses++;
        else
            falseGuesses++;
    }

    cout<<"Correct Guesses: "<<correctGuesses<<endl;
    cout<<"False Guesses: "<<falseGuesses<<endl;


    return 0; 
}