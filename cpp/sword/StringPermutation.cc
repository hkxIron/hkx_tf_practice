/*
g++ -std=c++11 -o string_permutaion StringPermutation.cc &&./string_permutaion
*/


#include <cstdio>

void Permutation(char* pStr, char* pBegin);

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

/*'
1. 将字符串分为两部分，第一部分为 字符串的第一个字符， 另一部分为第一个字符后的所有字符
2.拿第一个字符与它后面的所有字符逐个交换
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
        for(char* pCh = pBegin; *pCh != '\0'; ++ pCh)
        {
            swap(pCh, pBegin);
            // 1. 将字符串分为两部分，第一部分为 字符串的第一个字符， 另一部分为第一个字符后的所有字符
            Permutation(pStr, pBegin + 1);
            swap(pCh, pBegin);
        }
    }
}

// 拓展：求字符序列n的所有组合
// n:所有的数字， m:长度为m的组合
// 递归：如果组合里包含特殊元素a0, C(n-1, m-1),如果不包含a0, C(n-1, m)
// C(n,m) = C(n-1, m-1) + C(n-1, m)
// C(6,3) = C(5,2)+C(5,3) = 10 +10 =20
/*
void Combination(char* pStr, char* pBegin)
{
    if(*pBegin == '\0') // 遇到末尾
    {
        printf("%s\n", pStr);
    }
    else
    {
        // 2.拿第一个字符与它后面的所有字符逐个交换
        for(char* pCh = pBegin; *pCh != '\0'; ++ pCh)
        {

            Combination
        }
    }
}
*/

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
    return 0;
}