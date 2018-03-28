#include <stdio.h>

#define BITSPERWORD 32
#define SHIFT 5
#define MASK 0x1F // 值为31= 32 -1
#define N 10000000

int a[1 + N/BITSPERWORD];//申请内存的大小

//set 设置所在的bit位为1
//clr 初始化所有的bit位为0
//test 测试所在的bit为是否为1

void set(int i) {
   a[i>>SHIFT] |=  (1<<(i & MASK));
}
void clear(int i) {
   a[i>>SHIFT] &= ~(1<<(i & MASK));
}

int contains(int i){
    return a[i>>SHIFT] &(1<<(i & MASK));
}

int main()
{   int i;
    for (i = 0; i < N; i++)
        clear(i);
    //while (scanf("%d", &i) != EOF) set(i);
    int b[]={10,20,21,34 };
    int len_b=sizeof(b)/sizeof(int);
    for(i=0;i<len_b;i++) set(b[i]);
    for (i = 0; i < N; i++)
        if (contains(i))
            printf("%d\n", i);
    return 0;
}