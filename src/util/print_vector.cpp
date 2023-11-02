
#include <iostream>
#include <vector>

using namespace std;

void print_vector(vector<double> v)
{
    cout << "[";
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i];
        if (i != v.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}