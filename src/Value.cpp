#include "Value.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <cmath>

#define SV std::shared_ptr<Value>

Value::Value(double data)
{
    this->data = data;
    this->grad = 0;
    this->backprop = []() {};
    this->op = "none";
}

Value::Value(double data, std::vector<SV> children)
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

SV Value::add(SV v)
{
    // https://cplusplus.com/reference/memory/make_shared/
    SV out(new Value(this->data + v->data, {shared_from_this(), v}));

    out->backprop = [this, v, &out]()
    {
        this->grad += out->grad;
        v->grad += out->grad;
    };

    out->op = "add";

    return out;
}

SV Value::mul(SV v)
{
    SV out(new Value(this->data * v->data, {shared_from_this(), v}));

    out->backprop = [this, &out, &v]()
    {
        this->grad += v->data * out->grad;
        v->grad += this->data * out->grad;
    };

    out->op = "mul";

    return out;
}

SV Value::pow(double p)
{
    SV out(new Value(std::pow(this->data, p), {shared_from_this()}));

    out->backprop = [this, &out, &p]()
    {
        this->grad += p * std::pow(this->data, p - 1) * out->grad;
    };

    out->op = "pow";

    return out;
}

SV Value::exp()
{
    SV out(new Value(std::exp(this->data), {shared_from_this()}));
    out->backprop = [this, &out]()
    {
        std::cout << "backprop exp" << std::endl;
        this->grad += std::exp(this->data) * out->grad;
    };
    out->op = "exp";

    return out;
}

SV Value::neg()
{
    SV negative_one(new Value(-1.0));
    return this->mul(negative_one);
}

SV Value::sub(SV v)
{
    return this->add(v->neg());
}

SV Value::div(SV v)
{
    return this->mul(v->pow(-1.0));
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
    std::vector<SV> queue;
    std::set<SV> visited;
    std::vector<SV> topo;

    queue.push_back(shared_from_this());

    std::cout << "topological sort" << std::endl;
    while (queue.size() > 0)
    {
        SV current = queue.back();
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

            for (SV child : current->children)
            {
                queue.push_back(child);
            }
        }
    }
}
