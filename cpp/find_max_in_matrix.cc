/**
 * 二维数组找最大值
 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。 

答案的解法是从左下角开始
如果整数比这个元素大说明整数在这个元素的右边，
如果整数比这个元素小说明整数在这个元素的左边。
更新元素，重复上述步骤
如果这个元素已经到边界并且出界了还没找到，那么就是没有
 */
#include<stdio.h>

int search(int* arr,int row_len,int col_len,int num){
    // 把二维数组当做一维数组处理
    if(arr == NULL || row_len <= 0 || col_len<= 0) return -1;
    int row = row_len-1;
    int col = 0;
    while(row >= 0 && row < row_len && col >=0 && col < col_len){
        int cur_value = arr[row*col_len + col];
        if(cur_value > num) {
            row--;
        }
        else if(cur_value < num) {
            col++;
        }
        else{
            return 1;
        }
    }
    return -1;
}

int main() {
    int arr[5][5] = {1,55,79,82,88,
                     2,56,80,83,109,
                     3,57,81,84,110,
                     4,58,90,100,120,
                     5,59,99,111,122};

    int num=89;
    printf("find:%d contains: %d\n", num,search((int*)arr,5,5,num));
    num=110;
    printf("find:%d contains: %d\n", num,search((int*)arr,5,5,num));
    num=88;
    printf("find:%d contains: %d\n", num,search((int*)arr,5,5,num));
    return 0;
}
