// 选择排序：
#include<iostream>
#include<iterator>
#include<algorithm>
using namespace std;

void select_sort(int a[],int n){
    for(int i=0;i<n-1;++i){ // 排序的趟数, N-1
        int min_index = i;
        int min = a[i];
        for(int j=i+1;j<n;++j){ // 先排好前面的元素
            if(a[j]<min){ // 每次都将余下最小的元素排到最前面
                min_index = j;
                min=a[j];
            }
        }
        if(min_index!=i){
            std::swap(a[i],a[min_index]);
        }
    }
}

int main(){
    int a[]={-1, -1,3,-2,10,7,-4,-3};
    int n = sizeof(a)/sizeof(int);
    int b[100];
    std::copy(a, a+n, b); // algorithm,iterator

    cout<<"before sort:"<<endl;
    std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    select_sort(a,n);
    std::cout<<"\nafter sort:"<<endl;
    std::copy(a, a+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator


    cout<<"\nbefore sort:"<<endl;
    std::copy(b, b+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
    sort(b,b+n,[](int a,int b){ return a<=b; }); // 调用c++默认的排序函数进行排序
    std::cout<<"\nafter sort:"<<endl;
    std::copy(b, b+n, std::ostream_iterator<int>(std::cout, " ")); // algorithm,iterator
}
