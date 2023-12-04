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
            unsigned int temp = (unsigned int)((unsigned char)image[j]);
            // we normalize the values to be between 0 and 1
            // by dividing by 255, the maximum value of a byte

            image_vector[j] = (double)(temp) / 255.0;
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
            unsigned int temp = (unsigned int)((unsigned char)image[j]);

            // we normalize the values to be between 0 and 1
            // by dividing by 255, the maximum value of a byte

            image_vector[j] = (double)(temp) / 255.0;
        }

        images_test->push_back(image_vector);
    };

    return true;
}

void printAsciiImage(const std::vector<double> &values)
{
    // Check if the size of the input vector is correct
    cout << "values.size(): " << values.size() << endl;

    if (values.size() != 784)
    {
        std::cerr << "Error: Input vector size is not 784." << std::endl;
        return;
    }

    // Define characters to represent different intensity levels
    const char intensityChars[] = {' ', '.', ',', ':', 'o', 'O', 'X', '#', '$', '@'};

    // Calculate the range for each intensity level
    const double range = 1.0 / (sizeof(intensityChars) / sizeof(intensityChars[0]) - 1);

    // Iterate over the vector and print ASCII characters based on the values
    for (int i = 0; i < values.size(); ++i)
    {
        // Adjust the intensity to a character in the range of ASCII characters
        int intensityLevel = static_cast<int>(values[i] / range);
        intensityLevel = std::min(std::max(intensityLevel, 0), static_cast<int>(sizeof(intensityChars) - 1));

        char pixel = intensityChars[intensityLevel];

        // Print the ASCII character
        std::cout << pixel;

        // Insert a newline character after every 28 characters to create a 28x28 image
        if ((i + 1) % 28 == 0)
        {
            std::cout << std::endl;
        }
    }
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

    double accuracy = correct / (double)predictions.size();

    return accuracy;
}

int main()
{
    // int seed = time(NULL);
    int seed = 0;
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

    DenseLayer d1 = DenseLayer(28 * 28, 100);
    SigmoidLayer a1 = SigmoidLayer();
    DenseLayer d2 = DenseLayer(100, 10);
    SigmoidLayer a2 = SigmoidLayer();

    for (int epoch = 0; epoch < 15; epoch++)
    {
        double learning_rate = 0.025;
        double mean_loss = 0.0;
        int i = 0;

        vector<int>
            predictions = vector<int>();

        // for (; i < 500; i++)
        for (; i < images_train.size(); i++)
        {

            int index = i;
            VD image = images_train[index];
            int label = labels_train[index];

            // forward pass
            VD d1_output = d1.forward(image);
            VD a1_output = a1.forward(d1_output);
            VD d2_output = d2.forward(a1_output);
            VD a2_output = a2.forward(d2_output);

            VD label_vector = VD(10);

            label_vector[label] = 1.0;

            // the prediction is the index of the highest value
            int prediction = 0;
            for (int j = 0; j < a2_output.size(); j++)
            {
                if (a2_output[j] > a2_output[prediction])
                {
                    prediction = j;
                }
            }

            predictions.push_back(prediction);

            MeanSquaredErrorLoss loss = MeanSquaredErrorLoss();
            double loss_output = loss.forward(
                a2_output,
                label_vector);

            mean_loss += loss_output;
            if (i % 500 == 0)
            {
                cout << setprecision(4) << "i:" << i << " | Mean Loss: " << (mean_loss / (i + 1)) << "\r" << flush;
            }

            // backward pass
            // zero_grad everything

            d1.zero_grad();
            d2.zero_grad();

            loss.backward(1.0);

            a2.backward(loss.grad);
            d2.backward(a2.grad);
            a1.backward(d2);
            d1.backward(a1.grad);

            // update weights
            d1.descend(learning_rate);
            d2.descend(learning_rate);
        }

        // lets get the accuracy on the training set

        cout << "                                           \r"
             << "Epoch: " << epoch << " | Loss: " << (mean_loss / i) << " | Train Accuracy: " << accuracy(predictions, labels_train) << endl;
    }

    // Evaluate the model by going through the test set

    vector<int>
        predictions = vector<int>();

    for (int i = 0; i < images_test.size(); i++)
    {
        int index = i;
        VD image = images_test[index];
        int label = labels_test[index];

        // forward pass
        VD d1_output = d1.forward(image);
        VD a1_output = a1.forward(d1_output);
        VD d2_output = d2.forward(a1_output);
        VD a2_output = a2.forward(d2_output);

        // the prediction is the index of the highest value
        int prediction = 0;

        for (int j = 0; j < a2_output.size(); j++)
        {
            if (a2_output[j] > a2_output[prediction])
            {
                prediction = j;
            }
        }

        predictions.push_back(prediction);
    }

    double final_acc = accuracy(predictions, labels_test);

    cout << "Test Accuracy: " << final_acc << endl;

    // Let's run an example on a random image from the test set
    int index = rand() % images_test.size();
    VD image = images_test[index];
    int label = labels_test[index];

    // forward pass
    VD d1_output = d1.forward(image);
    VD a1_output = a1.forward(d1_output);
    VD d2_output = d2.forward(a1_output);
    VD a2_output = a2.forward(d2_output);

    // the prediction is the index of the highest value
    int prediction = 0;

    for (int j = 0; j < a2_output.size(); j++)
    {
        if (a2_output[j] > a2_output[prediction])
        {
            prediction = j;
        }
    }

    printAsciiImage(image);
    cout << "Prediction: " << prediction << " | Label: " << label << endl;
    cout << "Probabilities: ";
    print_vector(a2_output);

    return 0;
}