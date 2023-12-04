#include <vector>
#include <iostream>

class SigmoidLayer
{
public:
    // constructor
    SigmoidLayer();
    // destructor
    ~SigmoidLayer();

    // methods
    std::vector<double> forward(std::vector<double> inputs);
    void backward(std::vector<double> grad);
    void backward(DenseLayer &prev_layer);

    // data
    std::vector<double> last_input;
    std::vector<double> grad;
    std::vector<double> last_output;
};

SigmoidLayer::~SigmoidLayer()
{
}

SigmoidLayer::SigmoidLayer()
{
}

std::vector<double> SigmoidLayer::forward(std::vector<double> inputs)
{
    this->last_input = inputs;
    std::vector<double> outputs = std::vector<double>(inputs.size());

    for (int i = 0; i < inputs.size(); i++)
    {
        outputs[i] = 1 / (1 + exp(-inputs[i]));
    }

    this->last_output = outputs;

    return outputs;
}

void SigmoidLayer::backward(std::vector<double> chain_grad)
{
    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < this->last_input.size(); i++)
    {
        this->grad[i] = this->last_output[i] * (1 - this->last_output[i]) * chain_grad[i];
    }
}

void SigmoidLayer::backward(DenseLayer &prev_layer)
{
    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < prev_layer.last_input.size(); i++)
    {
        for (int n = 0; n < prev_layer.neurons.size(); n++)
        {
            double curr_grad = this->last_output[n] * (1 - this->last_output[n]);
            double chain_grad = prev_layer.neurons[n].weights[i] * prev_layer.neurons[n].wgrad[i] / prev_layer.last_input[i];

            this->grad[i] += curr_grad * chain_grad;
        }
    }
}
