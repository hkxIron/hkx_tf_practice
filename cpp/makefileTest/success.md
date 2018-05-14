make -C f1
make[1]: 进入目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/f1”
gcc -c f1.c -o ../obj/f1.o
make[1]: 离开目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/f1”
make -C f2
make[1]: 进入目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/f2”
gcc -c f2.c -o ../obj/f2.o
make[1]: 离开目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/f2”
make -C main
make[1]: 进入目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/main”
gcc -c main.c -o ../obj/main.o
main.c: 在函数‘main’中:
main.c:5:5: 警告：隐式声明函数‘print1’ [-Wimplicit-function-declaration]
     print1();
     ^~~~~~
main.c:6:5: 警告：隐式声明函数‘print2’ [-Wimplicit-function-declaration]
     print2();
     ^~~~~~
make[1]: 离开目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/main”
make -C obj
make[1]: 进入目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/obj”
gcc -o ../bin/myapp f1.o f2.o main.o
make[1]: 离开目录“/cygdrive/d/public_code/hkx_tf_practice/cpp/makefileTest/obj”
