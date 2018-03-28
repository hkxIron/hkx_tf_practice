//copyright@ mark allen weiss && July && yansha
//July，yansha、updated，2011.05.08.

//本程序，后经飞羽找出错误，已经修正。
//随机选取枢纽元，寻找最小的第k个数
#include <iostream>
#include <stdlib.h>
using namespace std;

int my_rand(int low, int high)
{
    int size = high - low + 1;
    return  low + rand() % size;
}

//q_select places the kth smallest element in a[k]
int q_select(int a[], int k, int left, int right)
{
    if(k > right || k < left)
    {
        //为了处理当k大于数组中元素个数的异常情况
        return false;
    }

    //真正的三数中值作为枢纽元方法，关键代码就是下述六行
    int midIndex = (left + right) / 2;
    if(a[left] < a[midIndex])
        swap(a[left], a[midIndex]);
    if(a[right] < a[midIndex])
        swap(a[right], a[midIndex]);
    if(a[right] < a[left])
        swap(a[right], a[left]);
    swap(a[left], a[right]);

    // 申请两个移动指针并初始化
    int pivot = a[right];
    int i = left;
    int j = right-1;
    // 根据枢纽元素的值对数组进行一次划分
    while( i < j ){
        while(i<j&&a[i] <= pivot) ++i; // 从左边找出违反顺序的数（大于pivot）
        while(i<j&&a[j] >= pivot) --j; // 从右边找出违反顺序的数（小于pivot）
        swap(a[i], a[j]);   //a[i] <= a[j]
    }
    // 此时 i=j
    swap(a[i], a[right]);
    //swap(a[i], a[left]);

    /* 对三种情况进行处理
    1、如果i=k，即返回的主元即为我们要找的第k小的元素，那么直接返回主元a[i]即可;
    2、如果i>k，那么接下来要到低区间A[0....m-1]中寻找，丢掉高区间;
    3、如果i<k，那么接下来要到高区间A[m+1...n-1]中寻找，丢掉低区间。
    */
    if (i == k)
        return true;
    else if (i > k)
        return q_select(a, k, left, i-1);
    else return q_select(a, k, i+1, right);  // 注意此处k并没有减，而是相对于整个数组a的
}

int main()
{
    int a[] = {7, 8, 13, 24 , 11, 19, 9, 54, 6, 4, 11, 1, 2, 33}; // 1, 2, 4, 6, 7, 8, 9, 11, ...
    int len = sizeof(a) / sizeof(int);
    q_select(a, 4, 0, len - 1);
    //std::cout<< a[3] <<std::endl; // 第4小的数在a[3] = 6
    for(int i = 0; i< len; i ++)
        std::cout<< a[i] <<std::endl; // 第4小的数在a[3] = 6
    return 0;
}