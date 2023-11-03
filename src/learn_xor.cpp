#include <iostream>
#include <fstream>
#include <vector>

#include "engine/DenseLayer.cpp"
#include "engine/ReLuLayer.cpp"
#include "engine/SoftmaxLayer.cpp"
#include "engine/MeanSquaredErrorLoss.cpp"

using namespace std;

#define VD vector<double>

int main()
{

    vector<VD> images;
    vector<double> labels;

    for (int i = 0; i < 10000; i++)
    {
        double j = i / 1000.0;
        double k = i / 100.0;

        images.push_back({j, k});
        labels.push_back(j + k * 2);
    }

    cout << "Data loaded" << endl;
    cout << "Images: " << images.size() << endl;
    cout << "Labels: " << labels.size() << endl;

    DenseLayer d1 = DenseLayer(2, 4);
    ReLuLayer a1 = ReLuLayer();
    DenseLayer d2 = DenseLayer(4, 1);
    ReLuLayer a2 = ReLuLayer();

    for (int epoch = 0; epoch < 25; epoch++)
    {
        double learning_rate = 0.001;
        double mean_loss = 0.0;

        for (int i = 0; i < 1000; i++)
        {
            int index = i;
            VD image = images[index];
            double label = labels[index];

            // forward pass
            VD d1_output = d1.forward(image);
            VD a1_output = a1.forward(d1_output);
            VD d2_output = d2.forward(a1_output);
            VD a2_output = a2.forward(d2_output);

            MeanSquaredErrorLoss loss = MeanSquaredErrorLoss();
            double loss_output = loss.forward(a2_output, {label});

            mean_loss += loss_output;
            cout << "i:" << i << " Loss: " << (mean_loss / (i + 1)) << "\r" << flush;

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
        cout << "Epoch: " << epoch << " | Loss: " << (mean_loss / images.size()) << endl;
    }
    return 0;
}