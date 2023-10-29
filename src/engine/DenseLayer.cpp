#include <vector>
#include "Neuron.cpp"
#include <memory>

class DenseLayer
{
public:
    // constructor
    DenseLayer(int input_size, int output_size);
    // destructor
    ~DenseLayer();

    // methods
    std::vector<double> forward(std::vector<double> inputs);
    std::vector<double> backward(std::vector<double> grad);

    // data
    std::vector<Neuron> neurons;
    std::vector<double> last_input;
    std::vector<double> wgrad; // the gradient of the weights
    std::vector<double> bgrad; // the gradient of the bias
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
