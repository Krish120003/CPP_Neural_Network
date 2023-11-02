#include <iostream>
#include <fstream>
#include <vector>

#include "engine/DenseLayer.cpp"
#include "engine/ReLuLayer.cpp"
#include "engine/SoftmaxLayer.cpp"
#include "engine/MeanSquaredErrorLoss.cpp"

using namespace std;

int main()
{

    DenseLayer d1 = DenseLayer(2, 2);
    ReLuLayer a1 = ReLuLayer();
    DenseLayer d2 = DenseLayer(2, 2);
    ReLuLayer a2 = ReLuLayer();
    MeanSquaredErrorLoss l = MeanSquaredErrorLoss();

    // layer 1 config
    for (int i = 0; i < d1.neurons.size(); i++)
    {
        cout << "d1.neurons[" << i << "].weights: ";
        print_vector(d1.neurons[i].weights);
        cout << "d1.neurons[" << i << "].bias: " << d1.neurons[i].bias << endl;
    }

    // layer 2 config
    for (int i = 0; i < d2.neurons.size(); i++)
    {
        cout << "d2.neurons[" << i << "].weights: ";
        print_vector(d2.neurons[i].weights);
        cout << "d2.neurons[" << i << "].bias: " << d2.neurons[i].bias << endl;
    }

    cout << "========== Doing forward pass...==========" << endl;

    auto o1 = d1.forward({-5, -6});
    auto r1 = a1.forward(o1);
    auto o2 = d2.forward(r1);
    auto r2 = a2.forward(o2);
    auto loss = l.forward(r2, {2, 2});

    print_vector(r2);
    cout << "Loss: " << loss << endl;

    d2.zero_grad();
    d1.zero_grad();

    l.backward(1);
    cout << "L GRAD: ";
    print_vector(l.grad);
    a2.backward(l.grad);
    d2.backward(a2.grad);
    a1.backward(d2);
    d1.backward(a1.grad);

    cout << "========== Gradients...==========" << endl;
    print_vector(a1.grad);

    // d2.neurons
    // loop over neurons
    for (int i = 0; i < d2.neurons.size(); i++)
    {
        cout << "d2.neurons[" << i << "].wgrad: ";
        print_vector(d2.neurons[i].wgrad);
        cout << "d2.neurons[" << i << "].bgrad: " << d2.neurons[i].bgrad << endl;
    }

    // // d1.neurons
    // loop over neurons
    for (int i = 0; i < d1.neurons.size(); i++)
    {
        cout << "d1.neurons[" << i << "].wgrad: ";
        print_vector(d1.neurons[i].wgrad);
        cout << "d1.neurons[" << i << "].bgrad: " << d1.neurons[i].bgrad << endl;
    }

    return 0;
}