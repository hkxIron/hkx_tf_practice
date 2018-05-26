#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: hkxIron
# 根据视频列表，下载视频文件

# python 2/3 compatibility imports
from __future__ import print_function
from __future__ import unicode_literals
import os,sys
"""
D:\lunaWorkspace\old_bak\youtube_downloadProj>D:\lunaWorkspace\old_bak\youtube_d
ownloadProj\youtube-dl.exe https://www.youtube.com/watch?v=NfnWJUyUJYU  --proxy "dev-proxy.oa.com:8080" 
  -o %(title).%(ext)  --write-auto-sub --verbose
"""
downloader=r"D:\public_code\hkx_tf_practice\test_python\youtube_download\youtube-dl.exe "
proxy = ' --proxy "dev-proxy.oa.com:8080" '
format = "_%(title)s_%(resolution)s.%(ext)s  --write-auto-sub --verbose "

def download(link_list, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("mkdir: ",output_dir)
    num = len(link_list)
    if len(link_list) == 0:
        print("no video need download!")
        return
    print("video num:%d \nlink_list:%s"%(num, "\n".join(link_list)))
    index = 1
    failed_list = []
    for link in link_list:
        prefix = str(index).zfill(2)  # 填充为两位
        print("begin to download: ", link)
        cmd = downloader + link +" -o "+ os.path.join(output_dir, prefix + format)
        if len(proxy) > 0:
            cmd += proxy
        print("cmd: ", cmd)
        success_flag = os.system(cmd)>>8 == 0
        if success_flag:
            print("download video success:", link)
        else:
            failed_list.append(link)
            print("download video failed:", link)
        index += 1
    print("total: %d success:%d failed:%d"%(num, num - len(failed_list), len(failed_list)))

def read_link_list(link_list_file):
    link_list = []
    with open(link_list_file,"r") as fr:
        for line in fr.readlines():
            link_list.append(line.rstrip("\n"))
    return link_list

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:%s link_list_file output_dir"%sys.argv[0])
        sys.exit(0)
    link_list_file = sys.argv[1]
    output_dir = sys.argv[2].strip()
    print("link list file: %s"%link_list_file)
    link_list = read_link_list(link_list_file)
    download(link_list, output_dir)
    print("job down.")

