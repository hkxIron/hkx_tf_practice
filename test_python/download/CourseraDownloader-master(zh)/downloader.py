#!/usr/bin/python
#coding:utf-8
"""
  royguo1988@gmail.com
"""
from auth import Course
import os
import re
import sys
import subprocess
import time

class Downloader(object):
  """下载器,登陆后使用"""
  def __init__(self, course, path):
    self.course = course
    self.path = path
    # {[id, name],[id, name]}
    self.links = []
    # 解析课程首页，获得链接信息
    self.parse_links()
    # 监控子进程的状态
    self.tasks = []

  def parse_links(self):
    html = 'lectures.html'
    # 下载课程首页，根据页面html抽取页面链接，应该有更好的方式实现...
    cmd = ['curl', 'https://class.coursera.org/' + self.course.class_name+'/lecture/index', 
           '-k', '-#','-L', '-o', html, 
           '--cookie', 'csrf_token=%s; session=%s' % (self.course.csrf_token, self.course.session)]
    subprocess.call(cmd)
    with open('lectures.html','r') as f:
      arr = re.findall(r'data-lecture-id="(\d+)"|class="lecture-link">\n(.*)</a>',f.read())
      i = 0
      while i < len(arr):
        lecture_id = arr[i][0]
        lecture_name = arr[i+1][1]
        self.links.append([lecture_id, lecture_name])
        i += 2
    print 'total lectures : ', len(self.links)
    os.remove(html)

  def download(self, url, target):
    if os.path.exists(target):
      print target,' already exist, continue...'
    else:
      print 'downloading : ', target
    # print 'url : ', url

    # -k : allow curl connect ssl websites without certifications.
    # -# : display progress bar.
    # -L : follow redirects.
    # -o : output file.
    # -s : slient mode, don't show anything
    # -C - : continue the downloading from last break point.
    # --cookie : String or file to read cookies from.
    cmd = ['curl', url, '-k','-s','-L','-C -', '-o', target, '--cookie',
           'csrf_token=%s; session=%s' % (self.course.csrf_token, self.course.session)]
    p = subprocess.Popen(cmd)
    self.tasks.append(p)

  def fetchAll(self):
    # count作为文件名的前缀遍历所有链接
    count = 1
    for link in self.links:
      # 下载字幕
      srt_url = "https://class.coursera.org/"+self.course.class_name+"/lecture/subtitles?q=%s_en&format=srt" %link[0]
      srt_name = self.path + str(count) + '.' +link[1]+'.srt'
      self.download(srt_url, srt_name)
      # 下载视频
      video_url = "https://class.coursera.org/"+self.course.class_name+"/lecture/download.mp4?lecture_id=%s" %link[0]
      video_name = self.path + str(count) + '.' +link[1]+'.mp4'
      self.download(video_url, video_name)

      count += 1

def main():
  if len(sys.argv) != 3:
    # class name example "neuralnets-2012-001"
    print 'usage : ./downloader.py download_dir class_name'
    return
  path = re.sub(r'/$','',sys.argv[1]) + "/"
  if not os.path.exists(path):
    os.makedirs(path)
  print 'download dir : ', path

  # 账号和课程信息
  from config import USER
  c = Course(USER['username'], USER['password'], sys.argv[2])
  d = Downloader(c, path)
  d.fetchAll()
  while True:
    print 'reminding tasks : ', len(d.tasks)
    for t in d.tasks:
      if t.poll() == 0:
        d.tasks.remove(t)
    time.sleep(1)

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt, e:
    print 'downloader has been killed !'
  
