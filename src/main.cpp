#include "Value.hpp"
#include <iostream>

using namespace std;

#define SV std::shared_ptr<Value>

int main()
{

    SV x(new Value(1));
    SV y(new Value(2));
    SV z = x->add(y)->exp();

    // z.grad = 1;

    // z.backward();

    cout << x->to_string() << endl;
    cout << y->to_string() << endl;
    cout << z->to_string() << endl;

    return 0;
}