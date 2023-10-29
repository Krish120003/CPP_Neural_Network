#include <vector>

class SoftmaxLayer
{
public:
    // constructor
    SoftmaxLayer();
    // destructor
    ~SoftmaxLayer();

    // methods
    std::vector<double> forward(std::vector<double> inputs);
    std::vector<double> backward(std::vector<double> grad);

    // data
    std::vector<double> last_input;
    std::vector<double> grad;
};

SoftmaxLayer::~SoftmaxLayer()
{
}

SoftmaxLayer::SoftmaxLayer()
{
}

std::vector<double> SoftmaxLayer::forward(std::vector<double> inputs)
{
    this->last_input = inputs;
    std::vector<double> outputs = std::vector<double>(inputs.size());

    double total = 0.0;
    for (int i = 0; i < inputs.size(); i++)
    {
        outputs[i] = exp(inputs[i]);
        total += outputs[i];
    }

    for (int i = 0; i < inputs.size(); i++)
    {
        outputs[i] /= total;
    }

    return outputs;
}