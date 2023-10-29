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
    std::vector<double> backward(std::vector<double> grad);

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