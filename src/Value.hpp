#include <functional>
#include <string>
#include <vector>
#include <memory>

#define SV std::shared_ptr<Value>

// https://en.cppreference.com/w/cpp/memory/enable_shared_from_this
class Value : std::enable_shared_from_this<Value>
{
private:
public:
    double data;
    double grad;
    std::vector<SV> children;       // children of this node in the computational graph
    std::function<void()> backprop; // per-value backpropagation function
    std::string op;

    Value(double data);
    Value(double data, std::vector<SV> children);
    ~Value();

    SV add(SV v);     // addition
    SV mul(SV v);     // multiplication
    SV sub(SV v);     // subtraction
    SV div(SV v);     // division
    SV pow(int v);    // power
    SV pow(double v); // power
    SV neg();         // negation
    SV exp();         // exponential
    void backward();  // backpropagation, on the full computational graph

    std::string to_string();
};
