#include <functional>
#include <string>
#include <vector>

class Value
{
private:
public:
    double data;
    double grad;
    std::vector<Value *> children;  // children of this node in the computational graph
    std::function<void()> backprop; // per-value backpropagation function
    std::string op;

    Value(double data);
    Value(double data, std::vector<Value *> children);
    ~Value();

    Value add(Value &v); // addition
    Value mul(Value &v); // multiplication
    Value sub(Value &v); // subtraction
    Value div(Value &v); // division
    Value pow(int v);    // power
    Value pow(double v); // power
    Value neg();         // negation
    Value exp();         // exponential
    void backward();     // backpropagation, on the full computational graph

    std::string to_string();
};
