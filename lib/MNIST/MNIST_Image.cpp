#include "MNIST_Image.h"

MNIST_Image::MNIST_Image(string line)
{
    string delimiter = ",";
    size_t pos = 0;
    string token;
    int i = 0;
    while ((pos = line.find(delimiter)) != string::npos)
    {
        token = line.substr(0, pos);
        if (i == 0)
        {
            this->label = stoi(token);
        }
        else
        {
            this->pixels[i - 1] = stoi(token);
        }
        line.erase(0, pos + delimiter.length());
        i++;
    }
    this->pixels[783] = stoi(line);
}

MNIST_Image::~MNIST_Image()
{
}