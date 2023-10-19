#include "Value.hpp"
#include <iostream>

using namespace std;

int main()
{

    Value x = Value(1);
    Value y = Value(1);
    Value z = x.mul(y).exp();
    z.grad = 1;

    z.backward();

    cout << x.to_string() << endl;
    cout << y.to_string() << endl;
    cout << z.to_string() << endl;

    return 0;
}