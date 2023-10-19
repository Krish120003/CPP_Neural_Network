#include "Value.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <iostream>

#include <iostream>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <vector>

Value::Value(double data)
{
    this->data = data;
    this->grad = 0;
    this->backprop = []() {};
    this->op = "none";
}

Value::Value(double data, std::vector<Value *> children)
{
    this->data = data;
    this->grad = 0;
    this->backprop = []() {};
    this->children = children;
    this->op = "none";
}

Value::~Value()
{
}

Value Value::add(Value &v)
{
    Value out = Value(this->data + v.data, {this, &v});

    out.backprop = [this, &out, &v]()
    {
        this->grad += out.grad;
        v.grad += out.grad;
    };

    out.op = "add";

    return out;
}

Value Value::mul(Value &v)
{
    Value out = Value(this->data * v.data, {this, &v});

    out.backprop = [this, &out, &v]()
    {
        this->grad += v.data * out.grad;
        v.grad += this->data * out.grad;
    };

    out.op = "mul";

    return out;
}

Value Value::pow(double p)
{
    Value out = Value(std::pow(this->data, p), {this});

    out.backprop = [this, &out, &p]()
    {
        this->grad += p * std::pow(this->data, p - 1) * out.grad;
    };

    out.op = "pow";

    return out;
}

Value Value::exp()
{
    Value out = Value(std::exp(this->data), {this});
    out.backprop = [this, &out]()
    {
        std::cout << "backprop exp" << std::endl;
        this->grad += std::exp(this->data) * out.grad;
    };
    out.op = "exp";

    return out;
}

Value Value::neg()
{
    Value negative_one = Value(-1.0);
    return this->mul(negative_one);
}

Value Value::sub(Value &v)
{
    Value negative = v.neg();
    return this->add(negative);
}

Value Value::div(Value &v)
{
    Value negative_one = Value(-1.0);
    Value inverse = v.pow(-1.0);
    return this->mul(inverse);
}

std::string Value::to_string()
{
    return "<Value data=" + std::to_string(this->data) + ", grad=" + std::to_string(this->grad) + " op=" + this->op + ">";
}

void Value::backward()
{
    // we need to call the backpropogation in order
    // of the computational graph, so we use a queue
    // built from a topological sort of the graph
    std::vector<Value *> queue;
    std::set<Value *> visited;
    std::vector<Value *> topo;

    queue.push_back(this);

    std::cout << "topological sort" << std::endl;
    while (queue.size() > 0)
    {
        Value *current = queue.back();
        queue.pop_back();

        if (visited.count(current) < 1)
        {
            visited.insert(current);

            std::cout << "trying backprop on: " << current->to_string() << std::endl;

            if (current->backprop != nullptr)
            {
                current->backprop();
            }
            else
            {
                std::cout << "backprop is null" << std::endl;
            }
            std::cout << "backprop done" << std::endl;

            for (Value *child : current->children)
            {
                queue.push_back(child);
            }
        }
    }
}
