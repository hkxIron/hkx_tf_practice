#include<iostream>
using namespace std;

// 不能如此写, print_arr(char**p,int m)
void print_arr(char (*p)[10], int m){
   for(int i =0;i<m;i++){
        cout<<p[i]<<"\n";
   }
}

int main(){
    char arr[][10]={"hello","i","love","cs"};
    print_arr(arr,4);
    return 1;
}