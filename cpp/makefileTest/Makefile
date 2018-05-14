CC=gcc
SUBDIRS=f1 f2 main obj #上面三个'\'后面不能有空格  
OBJS=f1.o f2.o main.o  
BIN=myapp
OBJS_DIR=obj
BIN_DIR=bin
export CC OBJS BIN OBJS_DIR BIN_DIR #导出环境变量，传递到下级目录  

all:CHECK_DIR $(SUBDIRS)  
CHECK_DIR:  
	[[ ! -d $(BIN_DIR) ]]&&mkdir -p $(BIN_DIR)  

# 进入到下级目录执行make,注意不一定是编译，有可能是链接
# 相当于 cd subdir && $(MAKE)
$(SUBDIRS):ECHO  
	make -C $@
  
ECHO:  
	@echo $(SUBDIRS)  
	@echo begin compile  
  
clean:  
	$(RM) $(OBJS_DIR)/*.o  
	rm -f $(BIN_DIR)/*
