#pragma once
#ifndef MNIST_Image_H
#define MNIST_Image_H

#include <cstdint>
#include <string>

using namespace std;

class MNIST_Image
{
public:
    int label;
    float pixels[784];
    uint16_t size = 784;
    MNIST_Image(string line);
    ~MNIST_Image();
};

#endif