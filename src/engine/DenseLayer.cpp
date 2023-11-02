#include <vector>
#include <memory>

#include "Neuron.cpp"

class DenseLayer
{
public:
    // constructor
    DenseLayer(int input_size, int output_size);
    // destructor
    ~DenseLayer();

    // methods
    void zero_grad();
    std::vector<double> forward(std::vector<double> inputs);
    void backward(std::vector<double> grad);
    void descend(double learning_rate);

    // data
    std::vector<Neuron> neurons;
    std::vector<double> last_input;
};

DenseLayer::~DenseLayer()
{
}

DenseLayer::DenseLayer(int input_size, int output_size)
{
    // initialize neurons
    this->neurons = std::vector<Neuron>();

    for (int i = 0; i < output_size; i++)
    {
        Neuron to_add = Neuron(input_size);
        this->neurons.emplace_back(std::move(to_add));
    }
}

void DenseLayer::zero_grad()
{
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->neurons[i].zero_grad();
    }
}

std::vector<double> DenseLayer::forward(std::vector<double> inputs)
{
    this->last_input = inputs;
    std::vector<double> outputs = std::vector<double>(this->neurons.size());

    for (int i = 0; i < this->neurons.size(); i++)
    {
        outputs[i] = this->neurons[i].forward(inputs);
    }

    return outputs;
}

void DenseLayer::backward(std::vector<double> grad)
{
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->neurons[i].backward(last_input, grad[i]);
    }
}

void DenseLayer::descend(double learning_rate)
{
    for (int i = 0; i < this->neurons.size(); i++)
    {
        this->neurons[i].descend(learning_rate);
    }
}