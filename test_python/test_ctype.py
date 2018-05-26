import platform
from ctypes import *

print("system:",platform.system())
if platform.system() == 'Windows':
    #libc = cdll.LoadLibrary('C:\Windows\System32\msvcrt.dll') # Windows 系统下的 C 标准库动态链接文件为 msvcrt.dll
    libc = cdll.LoadLibrary('msvcrt.dll') # Windows 系统下的 C 标准库动态链接文件为 msvcrt.dll
elif platform.system() == 'Linux':
    libc = cdll.LoadLibrary('libc.so.6') #  /lib/x86_64-linux-gnu
print("libc:",libc)

libc.printf('%s\n', 'here!')        # here!
libc.printf('%S\n', u'there!')      # there!
libc.printf('%d\n', 42)             # 42
libc.printf('%ld\n', 60000000)      # 60000000

#libc.printf('%f\n', 3.14)          #>>> ctypes.ArgumentError
#libc.printf('%f\n', c_float(3.14)) #>>> dont know why 0.000000
libc.printf('%f\n', c_double(3.14)) # 3.140000
