/*
g++ -std=c++11 -o permutation_combination permutation_combination.cc &&./permutation_combination
*/

#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <math.h>
using namespace std;

/*
求N个数中,r个数的组合
以n=5,r=3为例,组合结果如下:
加法:
1 2 3
1 2 4
1 2 5
1 3 4
1 3 5
1 4 5
2 3 4
2 3 5
2 4 5
3 4 5

减法:
5 4 3
5 4 2
5 4 1
5 3 2
5 3 1
5 2 1
4 3 2
4 3 1
4 2 1
3 2 1

*/
class Combination{
    public:
        int N;
        int R;
        int* a;
        int count;

        Combination(int n, int r):N(n), R(r), count(0) {
            a = new int[N];
            for(int i=0;i<N;i++) {
               a[i] = 0;
            }
            a[0] = 1; // 注意:此处将a[0]初始化为1
        }

        ~Combination(){
            if(a!=NULL){
                delete[] a;
            }
        }


        bool output(){
            for(int i=0;i<N;i++){
                if(a[i]!=0) printf("%d ", a[i]);
            }
            printf("\n");
        }

        void try_comb(){
            a[0]=0;
            int k=0;
            while(k>-1){ //
                a[k]=a[k]+1; // 在前一个元素上加1
                if(a[k]<=N){
                    if(k==R-1){ // 到达指定元素, 输出
                       count++;
                       output();
                    }else{
                       k=k+1; // 填下一个元素
                       a[k]=a[k-1]; // 将前一个元素的值复制过来
                    }
                }else{
                    k=k-1; // 回溯
                }
            }
        }
};

/**
输出N个元素的全排列 n!
*/
class Permutation{
  public:
        int N;
        int* a;
        int* used; // 元素是否使用
        int count;

        Permutation(int n):N(n), count(0) {
            a = new int[N+1];
            used = new int[N+1];
            for(int i=0;i<=N;i++) {
               a[i] = i; // 每个位置的值初始化为i,而不是0
            }
            for(int i=0;i<=N;i++) {
               used[i] = 0;
            }
        }

        ~Permutation(){
            if(a!=NULL){
                delete[] a;
            }
            if(used!=NULL){
                delete[] used;
            }
        }


        bool output(){
            // 第0个元素我们不输出,空着不用
            for(int i=1;i<=N;i++){
                printf("%d ", a[i]);
            }
            printf("\n");
        }


        /*
        回溯法
        */
        void try_permutation(int position){
            for(int num=1;num<=N;num++){ // 遍历所有的值num
                if(used[num]==0){ // 查看num是否被使用
                   a[position]=num; // 第position个位置填入元素num
                   used[num]=1;
                }else{
                    continue; // 找下一个未被使用的元素
                }
                if(position == N){ // 到达最后一个元素,输出
                    count++;
                    output();
                }else{
                    try_permutation(position+1);
                }
                // 回溯,设置position上的元素未被使用
                used[a[position]] = 0;
            }
        }

        /*
            全排列算法:
            用以上回溯法搜索算法完成的全排列问题的复杂度为 O(n**n), 不是一个好的算法,
            因此不可能用它的结果去搜索排列树,下面的全排列算法的复杂度为 O(n!),
            其结果可以为搜索排列树所用(注意:也是搜索排列树的算法框架).

            根据全排列的概念,定义数组初始值为(1,2,3,4,...,n),这是全排列中的一种结果,然后通过数据间的交换,则可产生所有的不同的排列

            算法说明:
            1.有读者会问 for是否应该改为for(j=position+1;j<=n;j++),答案是否定的,例如:当n=3时,
            算法的输出为:123,132,213,231,321,312.排列123的输出说明第一次到达叶结点是不经过数据交换的,
            排列132的排列也是不进行数据交换的结果.

            2.for循环体中的第二个swap()调用,是用来恢复原顺序的,这就是前面提到的回溯还原的操作,例如:
            132是由123进行2,3交换得到的,同样排列213是由123进行1,2交换得到的,所以在每次回溯时,
            都要恢复本次操作前的原始顺序.

            提示:对此算法不仅要理解,而且要记忆,因为它是解决所有与排列有关问题的算法框架.
        */
        void try_permutation2(int position){
            if(position == N){
                count++;
                output();
            }else{
                for(int j=position;j<=N;j++){ // 遍历所有的值num
                    std::swap(a[position], a[j]);  // 排列树算法交换,交换j与position位置上的元素
                    try_permutation2(position+1);
                    std::swap(a[position], a[j]);
                }
            }
        }
};


int main(int argc, char* argv[]){
   {
        printf("\n组合问题\n");
        Combination comb = Combination(5,3);
        comb.try_comb();
        printf("N:%d R:%d count:%d\n", comb.N, comb.R, comb.count);
   }
   {
        printf("\n排列问题\n");
        Permutation permutation = Permutation(4);
        permutation.try_permutation(1);
        printf("N:%d count:%d\n", permutation.N, permutation.count);
   }
   {
        printf("\n排列问题2\n");
        Permutation permutation = Permutation(3);
        permutation.try_permutation2(1);
        printf("N:%d count:%d\n", permutation.N, permutation.count);
   }
}

