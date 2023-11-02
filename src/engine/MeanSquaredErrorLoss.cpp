#include <vector>

class MeanSquaredErrorLoss
{
public:
    MeanSquaredErrorLoss();
    ~MeanSquaredErrorLoss();

    double forward(std::vector<double> inputs, std::vector<double> targets);
    void backward(double grad);

    // data
    std::vector<double> last_input;
    std::vector<double> last_target;
    std::vector<double> grad;
};

MeanSquaredErrorLoss::MeanSquaredErrorLoss()
{
}

MeanSquaredErrorLoss::~MeanSquaredErrorLoss()
{
}

double MeanSquaredErrorLoss::forward(std::vector<double> inputs, std::vector<double> targets)
{
    // we only need to calculate the loss for the target class
    this->last_input = inputs;
    this->last_target = targets;

    double total = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        total += pow(inputs[i] - targets[i], 2);
    }

    double loss = total;
    return loss;
}

void MeanSquaredErrorLoss::backward(double grad)
{

    this->grad = std::vector<double>(this->last_input.size());

    for (int i = 0; i < this->last_input.size(); i++)
    {
        this->grad.at(i) = 2 * (this->last_input[i] - this->last_target[i]);
        this->grad.at(i) *= grad;
    }
}