#!/usr/bin/python
#coding:utf-8
"""
  royguo1988@gmail.com
"""
import cookielib
import os
import tempfile
import urllib
import urllib2

class Course(object):
  def __init__(self, username, password, class_name):
    self.username = username
    self.password = password
    self.class_name = class_name

    # 发送登陆POST数据的接口
    self.auth_url = "https://www.coursera.org/maestro/api/user/login"
    self.csrf_token = ""
    self.session = ""
    self.cookie_file = ""

    # 开启课程验证信息
    self.open()

  def __get_class_url(self):
      """获得当前课程的首页URL"""
      return 'https://class.coursera.org/%s/lecture/' % self.class_name

  def __set_csrf(self):
    """设置登陆页面的CSRF Token"""
    # 模拟cookie行为，主要用来获得cookie中的数据，并不真正存储在本地
    cookies = cookielib.LWPCookieJar()
    handlers = [
        urllib2.HTTPHandler(),
        urllib2.HTTPSHandler(),
        urllib2.HTTPCookieProcessor(cookies)
    ]
    opener = urllib2.build_opener(*handlers)
    req = urllib2.Request(self.__get_class_url())
    print 'Request for csrf token...'
    opener.open(req)
    for cookie in cookies:
        if cookie.name == 'csrf_token':
            self.csrf_token = cookie.value
            # break
    opener.close()

  def __auth(self):
    """登陆验证"""
    # 创建临时文件用来存储cookie数据
    hn, fn = tempfile.mkstemp()
    # 使用临时文件创建本地cookie
    cj = cookielib.MozillaCookieJar(fn)
    handlers = [
        urllib2.HTTPHandler(),
        urllib2.HTTPSHandler(),
        urllib2.HTTPCookieProcessor(cj)
    ]
    opener = urllib2.build_opener(*handlers)
    # 模拟浏览器，告诉登陆页面我们是从首页refer过来的
    std_headers = {
            'Cookie': ('csrftoken=%s' % self.csrf_token),
            'Referer': 'https://www.coursera.org',
            'X-CSRFToken': self.csrf_token,
            }
    auth_data = {
            'email_address': self.username,
            'password': self.password
            }
    # 把表单数据编码到form中, formatted_data中有数据会自动使用POST提交
    formatted_data = urllib.urlencode(auth_data)
    req = urllib2.Request(self.auth_url, formatted_data, std_headers)
    # 发送请求，请求结束后会自动把cookie数据存放到cookie jar中
    print 'Send login request...'
    opener.open(req)
    cj.save()
    opener.close()
    # 关闭临时文件
    os.close(hn)
    self.cookie_file = fn

  def __set_session(self):
    """获得课程session，下载的时候需要"""
    target = urllib.quote_plus(self.__get_class_url())
    auth_redirector_url = 'https://class.coursera.org/'+self.class_name+'/auth/auth_redirector?type=login&subtype=normal&email=&visiting='+target
    print 'redirect url : ',auth_redirector_url
    cj = cookielib.MozillaCookieJar()
    cj.load(self.cookie_file)

    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj),
                                  urllib2.HTTPHandler(),
                                  urllib2.HTTPSHandler())

    req = urllib2.Request(auth_redirector_url)
    print 'Request for session ... '
    opener.open(req)
    for cookie in cj:
        if cookie.name == 'session':
            self.session = cookie.value
            break
    opener.close()

  def open(self):
    """登陆某个课程首页，同时在本地存储cookie，需要预先enroll课程"""
    self.__set_csrf()
    print 'Get csrf token : ', self.csrf_token
    self.__auth()
    print 'Login in success, cookie file: ', self.cookie_file
    self.__set_session()
    print 'Get session : ', self.session
