#include <vector>

class ReLuLayer
{
public:
    // constructor
    ReLuLayer();
    // destructor
    ~ReLuLayer();

    // methods
    std::vector<double> forward(std::vector<double> inputs);
    void backward(std::vector<double> grad);
    void backward(DenseLayer &prev_layer);

    // data
    std::vector<double> last_input;
    std::vector<double> grad;
};

ReLuLayer::~ReLuLayer()
{
}

ReLuLayer::ReLuLayer()
{
}

std::vector<double> ReLuLayer::forward(std::vector<double> inputs)
{
    this->last_input = inputs;
    std::vector<double> outputs = std::vector<double>(inputs.size());

    for (int i = 0; i < inputs.size(); i++)
    {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }

    return outputs;
}

void ReLuLayer::backward(std::vector<double> chain_grad)
{
    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < this->last_input.size(); i++)
    {
        this->grad[i] = this->last_input[i] > 0 ? chain_grad[i] : 0;
    }
}

void ReLuLayer::backward(DenseLayer &prev_layer)
{
    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < prev_layer.last_input.size(); i++)
    {
        for (int n = 0; n < prev_layer.neurons.size(); n++)
        {
            cout << "INNER LOOP" << endl;
            cout << prev_layer.neurons[n].wgrad[i] << endl;
            cout << prev_layer.last_input[i] << endl;
            double old_grad = prev_layer.neurons[n].wgrad[i] / prev_layer.last_input[i];
            cout << "old_grad: " << old_grad << endl;
            this->grad[i] += prev_layer.neurons[n].weights[i] * old_grad;

            if (this->last_input[i] < 0)
            {
                this->grad[i] = 0;
            }
        }
    }

    // cout << "ReLuLayer grad built." << endl;

    // print_vector(this->grad);
}
