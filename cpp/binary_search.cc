/*
  g++ -std=c++11 -o  binary_search binary_search.cc &&./binary_search
*/
// 插入排序
// 被插入表的已经是有序，待插入元素每次需要查找插入位置，同时需要移动元素
#include<iostream>
#include<iterator>
#include<algorithm>
using namespace std;

int binary_search(int a[],int n,int key){
    int low = 0;
    int high = n-1;
    while(low<=high){ // 注意，这里需要相等
        int mid = (low+high)/2;
        if(key<a[mid]) high=mid-1; // 因为值已经不相等了，所以要-1
        else if(key>a[mid]) low = mid +1;
        else return mid;
    }
    // 最后如果未找到,就是 low = high +1
    return -1; // 未找到，返回-1
}

/**
头条面试:
给一个非降序数组,要求找出数组中最后一个小于x的数, 如果没有就返回-1

*/
int binary_search_last_less(int a[], int n, int key) {
    int low = 0;
    int high = n-1;
    while (low<=high) { // 注意，这里需要相等
        int mid = (low+high)/2;
        if(key<=a[mid]) high=mid-1; // 因为值已经不相等了，所以要-1
        else low = mid +1;
        //else if (key>a[mid]) low = mid +1;
    }
    // 最后如果未找到,就是 low = high +1
    /**
    if(low==0) return -1;
    else if(high ==n-1) return n-1;
    else {
        return low-1;
    }
    */
    return low-1;
    //return -1; // 未找到，返回-1
}

int main(){
    {
        //       0  1   2 3 4 5
        int a[]={-3,-2,-1,4,5,9};
        int n = sizeof(a)/sizeof(int);
        std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
        //
        std::cout<<endl;
        std::cout<<binary_search(a,n,45)<<endl;
        std::cout<<binary_search(a,n,10)<<endl;
        std::cout<<binary_search(a,n,4)<<endl;
        std::cout<<binary_search(a,n,9)<<endl;
        std::cout<<binary_search(a,n,-3)<<endl;
    }
    {
        cout << " 找最小的数: " <<endl;
        int a[] = {1,2,3,5,6};
        int n = sizeof(a)/sizeof(int);
        cout<< binary_search_last_less(a, n, 4) << endl;
        cout<< binary_search_last_less(a, n, 6) << endl;
        cout<< binary_search_last_less(a, n, -1) << endl;
        cout<< binary_search_last_less(a, n, 10) << endl;
    }
}