#include <vector>
#include "../util/random.cpp"

class Neuron
{
private:
public:
    // data
    std::vector<double> weights;
    double bias;

    // constructor
    Neuron(int input_size);
    // destructor
    ~Neuron();
    // move constructor
    Neuron(Neuron &&other)
    {
        this->weights = std::move(other.weights);
        this->bias = other.bias;
    }

    // methods
    double forward(std::vector<double> inputs);
};

Neuron::Neuron(int input_size)
{
    this->weights = std::vector<double>(input_size);
    this->bias = get_random();

    for (int i = 0; i < input_size; i++)
    {
        this->weights[i] = get_random();
    }
}
Neuron::~Neuron()
{
}

double Neuron::forward(std::vector<double> inputs)
{
    double total = this->bias;

    for (int i = 0; i < inputs.size(); i++)
    {
        total += inputs[i] * this->weights[i];
    }

    return total;
}