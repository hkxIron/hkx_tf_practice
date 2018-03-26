// 插入排序
// 被插入表的已经是有序，待插入元素每次需要查找插入位置，同时需要移动元素
//  g++ -std=c++11 -o  insert_sort insert_sort.cc &&./insert_sort.exe
#include<iostream>
#include<iterator>
#include<algorithm>
using namespace std;

void insert_sort(int a[],int n){
    for(int i=1;i<n;++i){ // 排序的趟数, N-1
        int current = a[i];
        int index=i-1;
        while(index>-1&&current<a[index]){
            a[index+1]=a[index]; // 元素往后移
            --index; // 往前查找
        }
        // 找到插入的地方
        a[index+1] = current;
    }
}

int main(){
    int a[]={-1,2,45,3,9,8,-1,3,-2,10,7,-4,-3};
    int n = sizeof(a)/sizeof(int);
    int b[100];
    std::copy(a, a+n, b); // algorithm,iterator

    cout<<"before sort:"<<endl;
    std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    insert_sort(a,n);
    std::cout<<"\nafter sort:"<<endl;
    std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator


    cout<<"\nbefore sort:"<<endl;
    std::copy(b, b+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    sort(b,b+n,[=](int a,int b){ return a<=b; }); // 调用c++默认的排序函数进行排序
    std::cout<<"\nafter sort:"<<endl;
    std::copy(b, b+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
}
