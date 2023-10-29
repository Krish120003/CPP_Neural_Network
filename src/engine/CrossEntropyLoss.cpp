#include <vector>

class CrossEntropyLoss
{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();
    double forward(std::vector<double> inputs, int target);
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
    double loss = -log(inputs.at(target));
    return loss;
}