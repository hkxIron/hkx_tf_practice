/*
g++ -std=c++11 -o string_permutaion StringPermutation.cc &&./string_permutaion
*/


#include <cstdio>
#include <stdlib.h>
//#include <iostream>
#include <cstring>
using namespace std;

void Permutation(char* pStr, char* pBegin);
// 输入一个字符串,打印出该字符串中字符的所有排列
void Permutation(char* pStr)
{
    if(pStr == nullptr)
        return;
    Permutation(pStr, pStr);
}

void swap(char* p, char* q) {
   char temp = *p;
   *p = *q;
   *q = temp;
}

/*
1. 将字符串分为两部分，第一部分为 字符串的第一个字符， 另一部分为第一个字符后的所有字符
2. 拿第一个字符与它后面的所有字符逐个交换

abc =>
abc  (a与a交换,就是保持不变) => b与b交换 abc
                                b与c交换 acb
bac (a与b交换)  => a与a交换 bac
                  a与c交换 bca
cba (a与c交换)  => b与b交换 cba
                  b与a交换 cab
*/
void Permutation(char* pStr, char* pBegin)
{
    if(*pBegin == '\0') // 遇到末尾
    {
        printf("%s\n", pStr);
    }
    else
    {
        // 2.拿第一个字符与它后面的所有字符逐个交换
        /* abcdef => bacdef  => ba cdef里将a与{cdef}里的各元素交换
                     cbadef
                     dbcaef
                     ebcdaf
                     fbcdea
        */
        for(char* pCh = pBegin; *pCh != '\0'; ++ pCh)
        {
            swap(pCh, pBegin);
            // 1. 将字符串分为两部分，第一部分为 字符串的第一个字符， 另一部分为第一个字符后的所有字符
            Permutation(pStr, pBegin + 1);
            swap(pCh, pBegin);
        }
    }
}

/*
   用一个array来记录每个字符是否被使用过

   因为在每种组合中字符串的每个字符只能出现一次，所以我们用一个boolean的数组来标识某个字符是否已经被添加到out过。
我们用一个StringBuilder的out来打印满足要求的的组合。

如果out长度和给定字符串的长度相等，说明我们已经得到一种排列组合，把它打印出来，然后return;

否则，
对字符串所有的字符进行循环，如果当前字符已经被添加过，
则跳过这个字符，否则，把字符添加到out里，
同时标记它已经被添加，接着通过递归得到以其作为开头的所有字符串的组合，
之后我们标记此字符已经处理结束（used[i] = false），
继续处理下一个位置的字符，直到将整个字符串都处理完毕。
*/
void permutaion2(char* pstr, char* gen_cur, char* gen_begin, bool* used, int len, int expect_len){
   if(len == expect_len) {
      *(++gen_cur) = '\0';
      printf("%s\n", gen_begin);
      return;
   }
    for(char*p=pstr;*p != '\0'; ++p){
        int index = p - pstr;
        if(used[index]) continue;
        char old = *gen_cur;
        // 先选中index
        used[index]=true;
        *gen_cur = *p;
        permutaion2(pstr, gen_cur+1, gen_begin, used, len+1, expect_len);
        // 再不选中index(恢复现场)
        *gen_cur = old;
        used[index] =false;
    }

}

// 拓展：求字符序列n的所有组合
// n:所有的数字， m:长度为m的组合
// 递归：如果组合里包含特殊元素a0, C(n-1, m-1),如果不包含a0, C(n-1, m)
// C(n,m) = C(n-1, m-1) + C(n-1, m)
// C(6,3) = C(5,2)+C(5,3) = 10 +10 =20

// pStr:原始串, pComb:生成的字符串
void Combination(char* pStr, char* pComb, char* begin, int& count)
{
    if(*pStr == '\0') // 遇到末尾
    {
        *pComb = '\0';
        count++;
        printf("%s\n", begin);
    }
    else {
        // 当前的字符有可能选中,也有可能不选中
        Combination(pStr+1, pComb, begin, count); //选不中
        *(pComb++) = *(pStr++);
        Combination(pStr, pComb, begin, count); //选中
    }
}

// ====================测试代码====================
void Test(char* pStr)
{
    if(pStr == nullptr)
        printf("Test for nullptr begins:\n");
    else
        printf("Test for %s begins:\n", pStr);

    Permutation(pStr);

    printf("\n");
}


/**

第21题：输入两个整数n和m，从数列 1,2,3,…,n中随意取几个数，使其和等于m。*要求将其中所有的可能组合列出来。
index: 0 1 2 3 4
value: 1 2 3 4 5

*/

void print_array(int length, int* used_value){
    for(int i=0;i<length;i++){
        if(used_value[i]>0){
            printf("%d ", used_value[i]);
        }
    }
    printf("\n");
}

void get_combination(int cur_num, int length, int remain_sum, int* used_value){
    if(cur_num>remain_sum||cur_num > length){
        return;
    }

    if(cur_num == remain_sum){
        used_value[cur_num-1] = cur_num; // 1,2
        print_array(length, used_value);
        return;
    }

    // cur_num被选中
    used_value[cur_num-1] = cur_num;
    get_combination(cur_num+1, length, remain_sum - cur_num, used_value);

    // cur_num未被选中
    used_value[cur_num-1] = -1;
    get_combination(cur_num+1, length, remain_sum , used_value);
}

void get_all_combinations(int n, int sum) {
    if(n<=0||sum<=0||(1+n)*n/2<sum){
        printf("输入参数非法");
        exit(1);
    }

    int* used_value = new int[n];
    //std::memset(used_value, 0, n);//以a为首地址，将sizeof(a)大小的连续内存空间全部填充0
    for(int i=0;i<n;i++){
        used_value[i]=-1;
    }
    get_combination(1, n, sum, used_value);
    delete[] used_value;
}


int main(int argc, char* argv[])
{
    Test(nullptr);

    char string1[] = "";
    Test(string1);

    char string2[] = "a";
    Test(string2);

    char string3[] = "ab";
    Test(string3);

    char string4[] = "abc";
    Test(string4);

    char string5[] = "abcd";
    Test(string5);

    {

        printf("第二种排列方法:\n");
        char str1[] = "abcd";
        char str2[10] = "";
        int len1= strlen(str1);
        bool* used = new bool[len1];
        for(int i=0;i<len1;i++) used[i] = false;
        permutaion2(str1, str2, str2, used, 0, len1);
    }

    char str_orgin[] = "abcd";
    char str_gen[10] = "";
    int count =0;
    printf("打印所有的组合:\n");
    Combination(str_orgin, str_gen, str_gen, count);
    printf("组合数:%d\n", count); // 2^4 = 16

    int sum = 8;
    printf("求指定和:%d\n", sum);
    get_all_combinations(7, sum); // 8= 1+2 +5 = 1+3 +4 = 1+7 = 2+6 =3+ 5
    return 0;
}