#include <vector>
#include "../util/random.cpp"
#include "../util/print_vector.cpp"
class Neuron
{
private:
public:
    // data
    std::vector<double> weights;
    std::vector<double> wgrad;
    double bias;
    double bgrad;

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
    void backward(std::vector<double> last_input, double grad);
    void zero_grad();
    void descend(double learning_rate);
};

Neuron::Neuron(int input_size)
{
    this->weights = std::vector<double>(input_size);
    this->bias = 0.1 * get_random();

    for (int i = 0; i < input_size; i++)
    {
        this->weights[i] = get_random();
    }
}
Neuron::~Neuron()
{
}

void Neuron::zero_grad()
{
    this->wgrad = std::vector<double>(this->weights.size());
    this->bgrad = 0.0;
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

void Neuron::backward(std::vector<double> last_input, double grad)
{
    this->bgrad += grad;
    for (int i = 0; i < this->wgrad.size(); i++)
    {

        this->wgrad.at(i) = this->wgrad.at(i) + grad * last_input.at(i);
    }
}

void Neuron::descend(double learning_rate)
{
    this->bias -= this->bgrad * learning_rate;
    for (int i = 0; i < this->weights.size(); i++)
    {
        this->weights.at(i) -= this->wgrad.at(i) * learning_rate;
    }
}