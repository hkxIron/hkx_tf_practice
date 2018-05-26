/*
blog:https://github.com/hkxIron/algorithm/blob/master/sword_offer/src/064.cpp
	[包含min函数的栈]
    [题目]
	定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
    [解析]
    - 解法 1:
        两个栈，一个存储值(stackVal)，一个存储当前的最小值(stackMin)，一一对应。
        空间复杂度，如果有 n 个 int 类型的数，则 O(2\*n\*sizeof(int))
        - push 一个值 val 时:
            if(val < stackMin.top): stackMin.push(val)
            else : stackMin.push(stackMin.top()) // 将之前的最小值再push 一次

        - pop, stackVal.pop, stackMin.pop()
        例如：
        data_stack:     3  2  4  5 -1 6
        min_data_stack: 3  2  2  2 -1 -1


    - 解法 2 （见下面的第一个代码段）：
        值使用 vector 存储以便于获取下标，minIndexs 存储最小值对应的下标（与 1 不同的是，相同的最小值不需要重复入栈）。
        最坏的空间复杂度还是：O(2\*n\*sizeof(int))。

    - 解法 3 (见下面的第二个代码段)：
        - stackVal 存储 x = value - minVal，即当前值和最小值的差值
        - 注意类型：long，因为当 minVal 为 int 范围内最小的数时，只要 value>0， x 就会超过 int 可以表示的范围。
        - 空间复杂度，由于堆栈使用的是 long 型的，sizeof(long) = 2*sizeof(int)，因此空间复杂度较第二种情况并没有减小。
*/

#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <stack>
#include <cassert>

using namespace std;

class Solution{
public:
    void push(int value){
        values.push_back(value);
        if(minIndexs.empty() || value < values[minIndexs.top()]){
            minIndexs.push(values.size()-1); //若当前入栈的元素小于栈顶下标所对应的值，将下标入栈
        }
    }

    void pop(){
        if(values.empty()) return;

        int val = values.back(); // 返回容器中最后一个元素。c.back();
        if(values.size()-1 == minIndexs.top()) // 如果将要出栈的元素下标等于 最小栈的栈顶下标对应的元素 ，那么将元素下标出栈
            minIndexs.pop();

        values.pop_back(); // 移除最后一个元素
    }

    int top(){
        return values.back(); // Calling this function on an empty container causes undefined behavior.
    }

    int min(){

        //assert( !minIndexs.empty() );
        return values[minIndexs.top()];
    }

private:
    vector<int> values; // 存储元素值的栈（值使用vector存储以便获取下标）
    stack<int> minIndexs;// 用来记录最小元素对应的下标
};

class Solution2{
public:
    void push(int value){
        if(stackVal.empty()) // 0 will be push into stackVal
            minVal = value;

        int x = value - minVal;
        if(x < 0){
            // val < minVal, update the minVal
            minVal = value;
        }

        stackVal.push(x);
    }

    void pop(){
        if(stackVal.empty())
            return;

        int val = stackVal.top();
        if(val <= 0){
            // update min value
            // in this case
            // minValNew - minValOld = x, if x < 0 then minValNew < minValOld,
            // so the minValNew will become the new minVal ==> minVal = minValNew
            // minValOld = minVal - x
            minVal = minVal - val;
        }

        stackVal.pop();
    }

    int top(){
        // assert stackVal is not empty
        return stackVal.top() + minVal;
    }

    int min(){
        return minVal;
    }
private:
    long minVal;
    stack<long> stackVal; // 存储的是当前值与最小值的差值
};


int main()
{
    {
        int a[]={10,3,-3,2,8,9,-4,1};
        int len=sizeof(a)/sizeof(int);
        Solution s;
        for(int x:a) s.push(x);
        std::copy(a,a+len,ostream_iterator<int>(std::cout," "));
        cout<<"min:"<<s.min()<<endl;
        s.pop();
        cout<<"min:"<<s.min()<<endl;
        s.pop();
        cout<<"min:"<<s.min()<<endl;
    }
    return 0;
}
