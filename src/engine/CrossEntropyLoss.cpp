/*
WARNING: This file is UNTESTED!
This code may be incorrect or non-functional.
*/

#include <vector>

class CrossEntropyLoss
{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();
    double forward(std::vector<double> inputs, int target);
    void backward(double grad);

    // data
    std::vector<double> last_input;
    int last_target;
    std::vector<double> grad;
};

CrossEntropyLoss::CrossEntropyLoss()
{
}

CrossEntropyLoss::~CrossEntropyLoss()
{
}

double CrossEntropyLoss::forward(std::vector<double> inputs, int target)
{
    // we only need to calculate the loss for the target class
    this->last_input = inputs;
    this->last_target = target;

    double loss = -log(inputs.at(target));
    return loss;
}

void CrossEntropyLoss::backward(double grad)
{
    // we only need to calculate the gradient for the target class
    this->grad = std::vector<double>(this->last_input.size());
    this->grad[this->last_target] = -1.0 / this->last_input[this->last_target];
}