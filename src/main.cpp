#include "../lib/Sequential_NN/NeuralNetwork.h"


#include <fstream>
#include <chrono>


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

    int epochs = 1;
    while(epochs<10){

        nn.CreateNetwork(784, 16, 10, 2);
        
        for(int i=(epochs - 1)*(trainImages.size()/10);i<epochs*(trainImages.size()/10);i++)
        {
            //auto started = std::chrono::high_resolution_clock::now();

            nn.ForwardPropagateImage(trainImages[i]);
            nn.BackPropagateImage(trainImages[i]);

            //time_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - started).count();
            //cout<<"Time for i:"<<i<<" "<<elapsed<<endl; 
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

        cout<<"Epoch: "<<epochs<<endl;
        cout<<"Correct Guesses: "<<correctGuesses<<endl;
        cout<<"False Guesses: "<<falseGuesses<<endl;
        epochs++;
    }


    return 0; 
}