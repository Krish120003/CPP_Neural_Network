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
#include "engine/SigmoidLayer.cpp"

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

bool load_data(vector<VD> *images_train, vector<int> *labels_train, vector<VD> *images_test, vector<int> *labels_test)
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
        labels_train->push_back((int)label);
    };

    labels_file.close();

    // Load images_train
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

    // read images_train
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

        images_train->push_back(image_vector);
    };

    images_file.close();

    // Load test labels
    ifstream test_labels_file;
    test_labels_file.open("data/t10k-labels-idx1-ubyte", ios::binary | ios::in);
    if (!test_labels_file.is_open())
    {
        return false;
    }

    // seek to beginning of file
    test_labels_file.seekg(0, ios::beg);

    // first 32 bits are magic number
    // next 32 bits are number of items

    // read magic number
    test_labels_file.read((char *)&magic_number, sizeof(magic_number));
    reverse_bytes((char *)&magic_number, sizeof(magic_number));

    test_labels_file.read((char *)&number_of_items, sizeof(number_of_items));
    reverse_bytes((char *)&number_of_items, sizeof(number_of_items));

    // read labels
    for (int i = 0; i < number_of_items; i++)
    {
        // each label is 1 byte, so we use a char
        char label;
        test_labels_file.read(&label, sizeof(label));
        labels_test->push_back((int)label);
    };

    test_labels_file.close();

    // Load images_test
    ifstream test_images_file;

    test_images_file.open("data/t10k-images-idx3-ubyte", ios::binary | ios::in);
    if (!test_images_file.is_open())
    {
        return false;
    }

    // seek to beginning of file
    test_images_file.seekg(0, ios::beg);

    // first 32 bits are magic number
    // next 32 bits are number of items
    // next 32 bits are number of rows
    // next 32 bits are number of columns

    // read magic number
    test_images_file.read((char *)&magic_number, sizeof(magic_number));
    reverse_bytes((char *)&magic_number, sizeof(magic_number));

    // read number of items
    test_images_file.read((char *)&number_of_items, sizeof(number_of_items));
    reverse_bytes((char *)&number_of_items, sizeof(number_of_items));

    // read number of rows
    test_images_file.read((char *)&number_of_rows, sizeof(number_of_rows));
    reverse_bytes((char *)&number_of_rows, sizeof(number_of_rows));

    // read number of columns
    test_images_file.read((char *)&number_of_columns, sizeof(number_of_columns));
    reverse_bytes((char *)&number_of_columns, sizeof(number_of_columns));

    // read images_test

    for (int i = 0; i < number_of_items; i++)
    {
        // each image is 28 * 28 = 784 bytes
        // so we use a char array
        char image[784];
        test_images_file.read(image, sizeof(image));

        // convert to vector of doubles
        VD image_vector = VD(784);
        for (int j = 0; j < 784; j++)
        {
            // we normalize the values to be between 0 and 1
            // by dividing by 255, the maximum value of a byte
            image_vector[j] = (double)image[j] / 255.0;
        }

        images_test->push_back(image_vector);
    };

    return true;
}

double accuracy(vector<int> predictions, vector<int> labels)
{
    double correct = 0.0;
    for (int i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == labels[i])
        {
            correct += 1.0;
        }
    }

    cout << "Correct: " << correct << endl;
    cout << "Total: " << predictions.size() << endl;

    double accuracy = correct / (double)predictions.size();

    return accuracy;
}

int main()
{
    int seed = time(NULL);
    cout << "Seed: " << seed << endl;
    srand(seed);

    vector<VD> images_train;
    vector<int> labels_train;

    vector<VD> images_test;
    vector<int> labels_test;

    bool loaded = load_data(&images_train, &labels_train, &images_test, &labels_test);
    if (!loaded)
    {
        cout << "Failed to load data" << endl;
        return 1;
    }

    cout << "Data loaded" << endl;
    cout << "Train Images: " << images_train.size() << endl;
    cout << "Train Labels: " << labels_train.size() << endl;
    cout << "Test Images: " << images_test.size() << endl;
    cout << "Test Labels: " << labels_test.size() << endl;

    DenseLayer d1 = DenseLayer(28 * 28, 10);
    SigmoidLayer a1 = SigmoidLayer();

    for (int epoch = 0; epoch < 1; epoch++)
    {
        double learning_rate = 0.01;

        double mean_loss = 0.0;
        int i = 0;
        double train_correct = 0.0;
        // for (; i < 100; i++)
        for (; i < images_train.size(); i++)
        {
            int index = i;
            VD image = images_train[index];
            int label = labels_train[index];

            // forward pass
            VD d1_output = d1.forward(image);
            VD a1_output = a1.forward(d1_output);
            VD label_vector = VD(10);

            label_vector[label] = 1.0;

            MeanSquaredErrorLoss loss = MeanSquaredErrorLoss();
            double loss_output = loss.forward(
                a1_output,
                label_vector);

            mean_loss += loss_output;
            if (i % 50 == 0)
            {
                cout << setprecision(4) << "i:" << i << " | Mean Loss: " << (mean_loss / (i + 1)) << " | Last Output: " << a1_output[0] << " | Label: " << label << "\r" << flush;
            }

            // backward pass
            // zero_grad everything
            d1.zero_grad();

            loss.backward(1.0);
            a1.backward(loss.grad);
            d1.backward(a1.grad);

            // update weights
            d1.descend(learning_rate);
        }

        cout << endl
             << "Epoch: " << epoch << " | Loss: " << (mean_loss / i) << endl;
    }

    // Evaluate the model by going through the test set

    vector<int> predictions = vector<int>();

    for (int i = 0; i < images_test.size(); i++)
    {
        int index = i;
        VD image = images_test[index];
        int label = labels_test[index];

        // forward pass
        VD d1_output = d1.forward(image);
        VD a1_output = a1.forward(d1_output);

        // the prediction is the index of the highest value
        int prediction = 0;

        for (int j = 0; j < a1_output.size(); j++)
        {
            if (a1_output[j] > a1_output[prediction])
            {
                prediction = j;
            }
        }

        predictions.push_back(prediction);
    }

    double final_acc = accuracy(predictions, labels_test);

    cout << "Final Accuracy: " << final_acc << endl;

    return 0;
}