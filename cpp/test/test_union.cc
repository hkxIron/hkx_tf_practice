#include <iostream>
using namespace std;

union test
{
     char mark;
     long num;
     float score;
     int num_int;
     char byte_array[4]; //这样可以获取int的byte_array字节数组
}a;

int main()
{
     // cout<<a<<endl; // wrong
     a.mark = 'b';
     cout<<a.mark<<endl; // 输出'b'
     cout<<a.num<<endl; // 98 字符'b'的ACSII值
     cout<<a.score<<endl; // 输出错误值

     cout<<"---------------"<<endl;
     a.num = 10;
     cout<<a.mark<<endl; // 输出换行 非常感谢suxin同学的指正
     cout<<a.num<<endl; // 输出10
     cout<<a.num_int<<endl; // 输出10
     cout<<a.score<<endl; // 输出错误值

     cout<<"---------------"<<endl;
     a.num = 10;
     a.score = 10.0;
     cout<<a.mark<<endl; // 输出空
     cout<<a.num<<endl; // 输出错误值
     cout<<a.score<<endl; // 输出10

     cout<<"---------------"<<endl;
     a.num = 1234324320;
     cout<<a.mark<<endl; // 输出换行 非常感谢suxin同学的指正
     cout<<a.num<<endl; // 输出10
     cout<<a.num_int<<endl; // 输出10
     cout<<a.byte_array<<endl; // `K▒I
     cout<<a.score<<endl; // 输出错误值

     return 0;
}