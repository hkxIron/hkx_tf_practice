#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

/**
    举个例子，如：有两条随机序列，如 1 3 4 5 7 5 ，and 2 4 5 3 5 7 6，则它们的最长公共子序列便是：4 5 5。
*/
// 最长公共子序列
int LCSLength(char* str1, char* str2, int **b)
{
    int i,j,length1,length2,len;
    length1 = strlen(str1);
    length2 = strlen(str2);

    // 双指针的方法申请动态二维数组
    // C矩阵大小：length1*length2, C矩阵用来存储子串长度
    int **c = new int*[length1+1]; //共有length1+1行
    for(i = 0; i < length1+1; i++)
        c[i] = new int[length2+1];//共有length2+1列

    for(i = 0; i < length1+1; i++)
        c[i][0]=0;        //第0列都初始化为0
    for(j = 0; j < length2+1; j++)
        c[0][j]=0;        //第0行都初始化为0

    for(i = 1; i < length1+1; i++) {
        for(j = 1; j < length2+1; j++) {
            if(str1[i-1]==str2[j-1]) { //由于c[][]的0行0列没有使用，c[][]的第i行元素对应str1的第i-1个元素
                c[i][j]=c[i-1][j-1]+1;
                b[i][j]='\';          // ↖" ，输出公共子串时的搜索方向, 对角线
            } else if (c[i-1][j]>c[i][j-1]) { // 上方的元素大于左方的元素
                c[i][j]=c[i-1][j]; // 选择上方的元素
                b[i][j]='|';  // "↑" ，上边的元素大
            } else { // 上方的元素小于左方的元素
                c[i][j]=c[i][j-1]; // 选择左方的元素
                b[i][j]= '-'; // "←"，左边的元素大
            }
        }
    }
    /*
    for(i= 0; i < length1+1; i++)
    {
    for(j = 0; j < length2+1; j++)
    printf("%d ",c[i][j]);
    printf("\n");
    }
    */
    len=c[length1][length2];
    for(i = 0; i < length1+1; i++)    //释放动态申请的二维数组
        delete[] c[i];
    delete[] c;
    return len;
}

void PrintLCS(int **b, char *str1, int i, int j)
{
    if(i==0 || j==0)
        return ;
    if(b[i][j]=='\') // 左上方
    {
        PrintLCS(b, str1, i-1, j-1);//从后面开始递归，所以要先递归到子串的前面，然后从前往后开始输出子串
        printf("%c",str1[i-1]);//c[][]的第i行元素对应str1的第i-1个元素
    }
    else if(b[i][j]=='|') // 上方
        PrintLCS(b, str1, i-1, j);
    else // '-'
        PrintLCS(b, str1, i, j-1); // 左边
}

// 最长公共子串(注意，不是子序列)
int max_sub_seq(char* str1, char* str2, int& last_index) {
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    int result = 0;     //记录最长公共子串长度
    const int N=20;
    int c[N][N]; // 类似于最长公共子序列中的子串长度矩阵
    for (int i = 0; i <= len1; i++) {
        for( int j = 0; j <= len2; j++) {
            if(i == 0 || j == 0) {
                c[i][j] = 0;
            } else if (str1[i-1] == str2[j-1]) {
                c[i][j] = c[i-1][j-1] + 1; //将左上角（对角线）的元素加1
                if(c[i][j]>result) last_index = i-1;
                result = std::max(c[i][j], result);
            } else { //如果一旦不相等，就置0，不像LCS中复制左边或者上方的长度
                c[i][j] = 0;
            }
        }
    }
    return result;
}

int main(void)
{
    //printf("请输入第一个字符串：");
    //gets(str1);
    //printf("请输入第二个字符串：");
    //gets(str2);
    char str1[100]="cnblogs";
    char str2[100]="belong";
    // 最长公共子序列：blog
    // 最长公共子串：lo
    int i,length1,length2,len;
    length1 = strlen(str1);
    length2 = strlen(str2);
    //双指针的方法申请动态二维数组
    int **b = new int*[length1+1];
    for(i= 0; i < length1+1; i++)
        b[i] = new int[length2+1];
    len=LCSLength(str1,str2,b);
    printf("str1:%s\n",str1);
    printf("str2:%s\n",str2);
    printf("最长公共子序列的长度为：%d\n",len);
    printf("最长公共子序列为：");
    PrintLCS(b,str1,length1,length2);

    int last_index;
    int max_sub_len = max_sub_seq(str1,str2,last_index);
    printf("\n最长公共子串长度：%d\n", max_sub_len);
    printf("最长公共子串：");
    for(int i = last_index-max_sub_len+1;i<=last_index;i++)
        cout<<str1[i];

    printf("\n");
    for(i = 0; i < length1+1; i++)//释放动态申请的二维数组
        delete[] b[i];
    delete[] b;
    //system("pause");
  }