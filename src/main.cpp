#include <iostream>
#include <fstream>
#include <vector>

#include "engine/DenseLayer.cpp"
#include "engine/ReLuLayer.cpp"
#include "engine/SoftmaxLayer.cpp"

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

void print_vector(vector<double> v)
{
    cout << "[";
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i];
        if (i != v.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

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

    DenseLayer d1 = DenseLayer(2, 10);
    ReLuLayer a1 = ReLuLayer();
    DenseLayer d2 = DenseLayer(10, 2);
    SoftmaxLayer a2 = SoftmaxLayer();

    VD inputs = {1.0, 2.0};
    VD d1_output = d1.forward(inputs);
    VD a1_output = a1.forward(d1_output);
    VD d2_output = d2.forward(a1_output);
    VD a2_output = a2.forward(d2_output);

    print_vector(d1_output);
    print_vector(a1_output);
    print_vector(d2_output);
    print_vector(a2_output);

    return 0;
}