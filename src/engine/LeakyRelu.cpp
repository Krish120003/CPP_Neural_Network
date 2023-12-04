#include <vector>

#define LEAK 0.1

class LeakyReLuLayer
{
public:
    // constructor
    LeakyReLuLayer();
    // destructor
    ~LeakyReLuLayer();

    // methods
    std::vector<double> forward(std::vector<double> inputs);
    void backward(std::vector<double> grad);
    void backward(DenseLayer &prev_layer);

    // data
    std::vector<double> last_input;
    std::vector<double> grad;
};

LeakyReLuLayer::~LeakyReLuLayer()
{
}

LeakyReLuLayer::LeakyReLuLayer()
{
}

std::vector<double> LeakyReLuLayer::forward(std::vector<double> inputs)
{
    this->last_input = inputs;
    std::vector<double> outputs = std::vector<double>(inputs.size());

    for (int i = 0; i < inputs.size(); i++)
    {
        outputs[i] = inputs[i] > 0 ? inputs[i] : inputs[i] * LEAK;
    }

    return outputs;
}

void LeakyReLuLayer::backward(std::vector<double> chain_grad)
{
    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < this->last_input.size(); i++)
    {
        this->grad[i] = this->last_input[i] > 0 ? chain_grad[i] : chain_grad[i] * LEAK;
    }
}

void LeakyReLuLayer::backward(DenseLayer &prev_layer)
{
    this->grad = std::vector<double>(this->last_input.size());
    // cout << "Building ReLuLayer grad..." << endl;

    // cout << "ReLu Last Input: ";
    // print_vector(this->last_input);

    // cout << "Previous Layer's Input: ";
    // print_vector(prev_layer.last_input);
    // cout << "Previous Layer's Weights: \n";
    // print_vector(prev_layer.neurons[0].weights);
    // print_vector(prev_layer.neurons[1].weights);
    // cout << "Previous Layer's Grads: \n";
    // print_vector(prev_layer.neurons[0].wgrad);
    // print_vector(prev_layer.neurons[1].wgrad);

    // cout << "Number of neurons:" << prev_layer.neurons.size() << endl;

    for (int i = 0; i < prev_layer.last_input.size(); i++)
    {

        for (int n = 0; n < prev_layer.neurons.size(); n++)
        {
            double old_grad = prev_layer.neurons[n].wgrad[i] / prev_layer.last_input[i];
            this->grad[i] += prev_layer.neurons[n].weights[i] * old_grad;
        }
    }

    for (int i = 0; i < this->last_input.size(); i++)
    {
        this->grad[i] = this->last_input[i] > 0 ? this->grad[i] : this->grad[i] * LEAK;
    }

    // cout << "ReLuLayer grad built." << endl;

    // print_vector(this->grad);
}
