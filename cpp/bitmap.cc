#include <stdio.h>

#define BITSPERWORD 32
#define SHIFT 5　// 2^5 = 32
#define MASK 0x1F // 对32求余,其就就是和 31(=32-1)进行与操作
#define N 10000000

int a[1 + N/BITSPERWORD];//申请内存的大小

//set 设置所在的bit位为1
//clr 初始化所有的bit位为0
//test 测试所在的bit为是否为1

/**
i>>SHIFT: 指是在哪一个 BIT_WORD 中
i&MASK: 在 BIT_WORD中的偏移量
*/
void set(int i) {
   int index = i>>SHIFT;
   int offset = i&MASK;
   a[index] |=  (1<<offset); // 32位一起设置的
   // 为何不直接用 =，因为要保持以前原有的值
}
void clear(int i) {
   int index = i>>SHIFT;
   int offset = i&MASK;
   a[index] &= ~(1<<offset); //注意这里是取反后与
}

int contains(int i){
   int index = i>>SHIFT;
   int offset = i&MASK;
   return a[index] &(1<<(offset));
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