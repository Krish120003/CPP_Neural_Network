#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "engine/DenseLayer.cpp"
#include "engine/ReLuLayer.cpp"
#include "engine/LeakyRelu.cpp"
#include "engine/SoftmaxLayer.cpp"
#include "engine/MeanSquaredErrorLoss.cpp"

using namespace std;

#define VD vector<double>

void reverse_bytes(char *bytes, int size)
{
    for (int i = 0; i < size / 2; i++)
    {
        char temp = bytes[i];
        bytes[i] = bytes[size - i - 1];
        bytes[size - i - 1] = temp;
    }
}

// void print_vector(vector<double> v)
// {
//     cout << "[";
//     for (int i = 0; i < v.size(); i++)
//     {
//         cout << v[i];
//         if (i != v.size() - 1)
//         {
//             cout << ", ";
//         }
//     }
//     cout << "]" << endl;
// }

bool load_data(vector<VD> *images, vector<int> *labels)
{
    // Load labels

    // train-labels-idx1-ubyte
    ifstream labels_file;
    labels_file.open("data/train-labels-idx1-ubyte", ios::binary | ios::in);
    if (!labels_file.is_open())
    {
        return false;
    }

    // seek to beginning of file
    labels_file.seekg(0, ios::beg);

    // first 32 bits are magic number
    // next 32 bits are number of items

    // read magic number
    int magic_number;
    labels_file.read((char *)&magic_number, sizeof(magic_number));
    reverse_bytes((char *)&magic_number, sizeof(magic_number));

    int number_of_items;
    labels_file.read((char *)&number_of_items, sizeof(number_of_items));
    reverse_bytes((char *)&number_of_items, sizeof(number_of_items));

    // read labels
    for (int i = 0; i < number_of_items; i++)
    {
        // each label is 1 byte, so we use a char
        char label;
        labels_file.read(&label, sizeof(label));
        labels->push_back((int)label);
    };

    labels_file.close();

    // Load images
    ifstream images_file;
    images_file.open("data/train-images-idx3-ubyte", ios::binary | ios::in);
    if (!images_file.is_open())
    {
        return false;
    }

    // seek to beginning of file
    images_file.seekg(0, ios::beg);

    // first 32 bits are magic number
    // next 32 bits are number of items
    // next 32 bits are number of rows
    // next 32 bits are number of columns

    // read magic number
    images_file.read((char *)&magic_number, sizeof(magic_number));
    reverse_bytes((char *)&magic_number, sizeof(magic_number));

    // read number of items
    images_file.read((char *)&number_of_items, sizeof(number_of_items));
    reverse_bytes((char *)&number_of_items, sizeof(number_of_items));

    // read number of rows
    int number_of_rows;
    images_file.read((char *)&number_of_rows, sizeof(number_of_rows));
    reverse_bytes((char *)&number_of_rows, sizeof(number_of_rows));

    // read number of columns
    int number_of_columns;
    images_file.read((char *)&number_of_columns, sizeof(number_of_columns));
    reverse_bytes((char *)&number_of_columns, sizeof(number_of_columns));

    // read images
    for (int i = 0; i < number_of_items; i++)
    {
        // each image is 28 * 28 = 784 bytes
        // so we use a char array
        char image[784];
        images_file.read(image, sizeof(image));

        // convert to vector of doubles
        VD image_vector = VD(784);
        for (int j = 0; j < 784; j++)
        {
            // we normalize the values to be between 0 and 1
            // by dividing by 255, the maximum value of a byte
            image_vector[j] = (double)image[j] / 255.0;
        }

        images->push_back(image_vector);
    };

    images_file.close();

    return true;
}

int main()
{
    int seed = time(NULL);
    cout << "Seed: " << seed << endl;
    srand(seed);

    vector<VD> images;
    vector<int> labels;

    bool loaded = load_data(&images, &labels);
    if (!loaded)
    {
        cout << "Failed to load data" << endl;
        return 1;
    }

    cout << "Data loaded" << endl;
    cout << "Images: " << images.size() << endl;
    cout << "Labels: " << labels.size() << endl;

    DenseLayer d1 = DenseLayer(28 * 28, 64);
    LeakyReLuLayer a1 = LeakyReLuLayer();
    DenseLayer d2 = DenseLayer(64, 64);
    LeakyReLuLayer a2 = LeakyReLuLayer();
    DenseLayer d3 = DenseLayer(64, 32);
    LeakyReLuLayer a3 = LeakyReLuLayer();
    DenseLayer d4 = DenseLayer(32, 1);
    LeakyReLuLayer a4 = LeakyReLuLayer();

    for (int epoch = 0; epoch < 100; epoch++)
    {
        double learning_rate = 0.001;
        if (epoch < 2)
        {
            learning_rate = 0.0001;
        }
        double mean_loss = 0.0;
        int i = 0;
        double train_correct = 0.0;
        for (; i < images.size() / 3; i++)
        {
            int index = i;
            VD image = images[index];
            int label = labels[index];

            // forward pass
            VD d1_output = d1.forward(image);
            VD a1_output = a1.forward(d1_output);
            VD d2_output = d2.forward(a1_output);
            VD a2_output = a2.forward(d2_output);
            VD d3_output = d3.forward(a2_output);
            VD a3_output = a3.forward(d3_output);
            VD d4_output = d4.forward(a3_output);
            VD a4_output = a4.forward(d4_output);

            int prediction = (int)round(a4_output[0]);
            if (prediction == label)
            {
                train_correct += 1.0;
            }

            MeanSquaredErrorLoss loss = MeanSquaredErrorLoss();
            double loss_output = loss.forward(a4_output, {(double)label});

            mean_loss += loss_output;
            if (i % 50 == 0)
                cout << setprecision(4) << "i:" << i << " | Mean Loss: " << (mean_loss / (i + 1)) << " | Last Output: " << a4_output[0] << " | Label: " << label << endl;

            // cout << "Layerwise outputs: \n";
            // cout << "Dense 1: ";
            // print_vector(d1_output);
            // cout << "Relu 1:  ";
            // print_vector(a1_output);
            // cout << "Dense 2: ";
            // print_vector(d2_output);
            // cout << "Relu 2:  ";
            // print_vector(a2_output);
            // cout << "Dense 3: ";
            // print_vector(d3_output);
            // cout << "Relu 3:  ";
            // print_vector(a3_output);
            // cout << "Dense 4: ";
            // print_vector(d4_output);
            // cout << "Relu 4:  ";
            // print_vector(a4_output);

            // backward pass
            // zero_grad everything
            d1.zero_grad();
            d2.zero_grad();
            d3.zero_grad();
            d4.zero_grad();

            loss.backward(1.0);
            a4.backward(loss.grad);
            d4.backward(a4.grad);
            a3.backward(d4);
            d3.backward(a3.grad);
            a2.backward(d3);
            d2.backward(a2.grad);
            a1.backward(d2);
            d1.backward(a1.grad);

            // update weights
            d1.descend(learning_rate);
            d2.descend(learning_rate);
            d3.descend(learning_rate);
            d4.descend(learning_rate);
        }

        // find accuracy
        double correct = 0;
        for (int j = 20000; j < 30000; j++)
        {
            int index = j;
            VD image = images[index];
            int label = labels[index];

            // forward pass
            VD d1_output = d1.forward(image);
            VD a1_output = a1.forward(d1_output);
            VD d2_output = d2.forward(a1_output);
            VD a2_output = a2.forward(d2_output);
            VD d3_output = d3.forward(a2_output);
            VD a3_output = a3.forward(d3_output);
            VD d4_output = d4.forward(a3_output);
            VD a4_output = a4.forward(d4_output);

            int prediction = (int)round(a4_output[0]);
            if (prediction == label)
            {
                correct += 1.0;
            }
        }

        double train_accuracy = (double)train_correct / 20000.0;
        double test_accuracy = (double)correct / 10000.0;

        cout << "Epoch: " << epoch << " | Loss: " << (mean_loss / i) << " | Train Accuracy: " << train_accuracy << " | Test Accuracy: " << test_accuracy << endl;
    }
    return 0;
}