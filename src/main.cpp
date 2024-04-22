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
    //int epochIndex = 1;
    nn.CreateNetwork(784, 128, 10, 2);
    while(epochs<1000){

        int trainCorrectGuesses = 0;
        int trainIncorrectGuesses = 0;
        // if(epochIndex %10==0)
        //     epochIndex = 1;
        float changeTotal = 0;


        auto started = std::chrono::high_resolution_clock::now();
        for(int i=0; i<trainImages.size(); i++)
        {

            int guess = nn.ForwardPropagateImage(trainImages[i]);
            if(guess == trainImages[i].label)
                trainCorrectGuesses++;
            else {
                trainIncorrectGuesses++;
                changeTotal += nn.BackPropagateImage(trainImages[i]);
            }
            nn.ResetValues();
        }
            time_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - started).count();
            cout<<"Epoch: "<<epochs<<endl;
            cout<<"Time for backprop: "<<elapsed<<endl;
            cout<<"Change Total: "<<changeTotal<<endl;
            cout<<"Train Correct Guesses: "<<trainCorrectGuesses<<endl;
            cout<<"Train Incorrect Guesses: "<<trainIncorrectGuesses<<endl;


            nn.PrintLayerAverage();
        if(trainCorrectGuesses==0){
            epochs++;
            continue;
        }

        int correctGuesses = 0;
        int falseGuesses = 0;

        for(int i=0; i<testImages.size(); i++)
        {
            int guess = nn.ForwardPropagateImage(testImages[i]);
            nn.ResetValues();
            if(guess == testImages[i].label)
                correctGuesses++;
            else
                falseGuesses++;
            
        }

        cout<<"Correct Guesses: "<<correctGuesses<<endl;
        cout<<"False Guesses: "<<falseGuesses<<endl;
        cout<<"-------------------------"<<endl<<endl;
        if(correctGuesses==testImages.size())
        {
            cout<<"Perfect Outcome Achieved"<<endl;
            return 0;
        }


        epochs++;
        // epochIndex++;
    }


    return 0; 
}