#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python 2/3 compatibility imports
from __future__ import print_function
from __future__ import unicode_literals

try:
    from http.cookiejar import CookieJar
except ImportError:
    from cookielib import CookieJar

try:
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlencode

try:
    from urllib.request import urlopen
    from urllib.request import build_opener
    from urllib.request import install_opener
    from urllib.request import HTTPCookieProcessor
    from urllib.request import Request
    from urllib.request import URLError
except ImportError:
    from urllib2 import urlopen
    from urllib2 import build_opener
    from urllib2 import install_opener
    from urllib2 import HTTPCookieProcessor
    from urllib2 import Request
    from urllib2 import URLError

# we alias the raw_input function for python 3 compatibility
try:
    input = raw_input
except:
    pass
import argparse
import getpass
import json
import os
import os.path
import re
import sys
from subprocess import Popen, PIPE
from datetime import timedelta, datetime
from bs4 import BeautifulSoup


DEFAULT_USER_AGENTS = {"chrome": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.63 Safari/537.31",
                       "firefox": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:24.0) Gecko/20100101 Firefox/24.0",
                       "edx": 'edX-downloader/0.01'}
USER_AGENT = DEFAULT_USER_AGENTS["edx"]

resource_list_page="http://vision.stanford.edu/teaching/cs231n/syllabus.html"; #资源列表所在的页面
dest_video_path="E://TeachVideo/"

# To replace the print function, the following function must be placed before any other call for print
def print(*objects, **kwargs):
    """
    Overload the print function to adapt for the encoding bug in Windows Console.
    It will try to convert text to the console encoding before print to prevent crashes.
    """
    try:
        stream = kwargs.get('file', None)
        if stream is None:
            stream = sys.stdout
        enc = stream.encoding
        if enc is None:
            enc = sys.getdefaultencoding()
    except AttributeError:
        return __builtins__.print(*objects, **kwargs)
    texts = []
    for object in objects:
        try:
            original_text = str(object)
        except UnicodeEncodeError:
            original_text = unicode(object)
        texts.append(original_text.encode(enc, errors='replace').decode(enc))
    return __builtins__.print(*texts, **kwargs)

def get_page_contents(url, headers):
    """
    Get the contents of the page at the URL given by url. While making the
    request, we use the headers given in the dictionary in headers.
    """
    result = urlopen(Request(url, None, headers))
    try:
        charset = result.headers.get_content_charset(failobj="utf-8")  # for python3
    except:
        charset = result.info().getparam('charset') or 'utf-8'
    return result.read().decode(charset)


def directory_name(initial_name):#将初始的命名变为合法的命名
    import string
    allowed_chars = string.digits+string.ascii_letters+" _."
    result_name = ""
    for ch in initial_name:
        if allowed_chars.find(ch) != -1:
            result_name += ch
    return result_name if result_name != "" else "course_folder"


def parse_args():
    """
    Parse the arguments/options passed to the program on the command line.
    """
    parser = argparse.ArgumentParser(prog='edx-dl',
                                     description='Get videos from the OpenEdX platform',
                                     epilog='For further use information,'
                                     'see the file README.md',)
    # positional
    # optional
    parser.add_argument('-r',
                        '--res_list',
                        dest="resource_list",
                        default=resource_list_page,
                        action='store',
                        help='resource list page')


def main():
    #args = parse_args()

    # if no args means we are calling the interactive version
#     if len(sys.argv) == 1:
#         print("Argument not enough!")
    
    #if  args.resource_list:
    #    resource_list_page=args.resource_list
        #print("You must supply resource_list")
        #sys.exit(2)
    print("--------------")
    # Prepare Headers
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
        'Referer': "",
        'X-Requested-With': 'XMLHttpRequest',
        'X-CSRFToken': "",
    }
    # Get user info/courses
    video_list=[]
    dash = get_page_contents(resource_list_page, headers)
    soup = BeautifulSoup(dash)
    video_achor=soup.findAll('a', attrs={'href': re.compile('^https://youtu.be/')})  #'bs4.element.Tag'
    
    #os.path.join(os.getcwd(),dest_video_path)
    #pathName=os.path.dirname(fname);
    if not os.path.exists(dest_video_path):
        print(u'Create directory:'+dest_video_path)
        os.makedirs(dest_video_path)
        
    ind=0
    for achor in video_achor:
        video_url=achor.attrs["href"]
        ind+=1
        print("第%d个视频:%s"%(ind,video_url))
        video_list.append(video_url.replace("https://youtu.be/","")) 

    #video_urls = ['http://youtube.com/watch?v=' + v_id for v_id in video_list]#原来都是从youtube上下的,但它怎么知道youtube上一定有呢？
    #print("-----------------")
    #for vd in video_urls:
    #    print(vd)
        
        
#     for vd in video_url:
#         cmd = ["youtube-dl",#其实是使用youtube-dl来下载的
#                    "-o", os.path.join(target_dir, filename_prefix + "-%(title)s.%(ext)s")]
        
        
    #print(video_achor)
"""
E:\kp\lunaWorkspace\youtube_downloadProj>youtube-dl.exe  https://www.youtube.com
/watch?v=NfnWJUyUJYU   -o %(title).%(ext)  --write-auto-sub  --proxy https://127
.0.0.1:13658
"""
    #soup.find(attrs={'type': re.compile('^video/mp4')})['src']
    #soup.findAll(attrs={'class':re.compile('^course-item-list-header')}):
    #data = soup.find_all('td')[1]
    #data=soup.findAll(attrs={'class':re.compile('table')})
#     if len(data)==0:
#     print("no table has class of table")
#     sys.exit(-1)
    #查找table标签，且其class为table
    #video_achor=soup.findAll('table.table a',attrs={'href': re.compile('^https://youtu.be/')})
    #video_achor=soup.select('table.table tr td a[href^="https://youtu.be/]')
    #print(video_achor)
    #--------------------
    #<a href="https://youtu.be/NfnWJUyUJYU"><b>[video]</b></a>
    
"""    
    
    USERNAME='Unkown'
    if len(data.find_all('span'))>1:
        USERNAME = data.find_all('span')[1].string #stanford online才会出错

    COURSES = soup.find_all('article', 'course')
    courses = []
    for COURSE in COURSES:
        c_name = COURSE.h3.text.strip()
        c_link = BASE_URL + COURSE.a['href']
        if c_link.endswith('info') or c_link.endswith('info/'):
            state = 'Started'
        else:
            state = 'Not yet'
        courses.append((c_name, c_link, state))
    numOfCourses = len(courses) #得到所有选过的课程

    # Welcome and Choose Course

    print('Welcome %s' % USERNAME)
    print('You can access %d courses' % numOfCourses)

    c = 0
    for course in courses:
        c += 1
        print('%d - %s -> %s' % (c, course[0], course[2]))#前面是名称，后面是状态

    c_number = int(input('Enter Course Number: '))
    while c_number > numOfCourses or courses[c_number - 1][2] != 'Started':
        print('Enter a valid Number for a Started Course ! between 1 and ',
              numOfCourses)
        c_number = int(input('Enter Course Number: '))
    selected_course = courses[c_number - 1]
    COURSEWARE = selected_course[1].replace('info', 'courseware')

    ## Getting Available Weeks
    courseware = get_page_contents(COURSEWARE, headers)
    soup = BeautifulSoup(courseware)

    data = soup.find(*COURSEWARE_SEL)
    WEEKS = data.find_all('div')
    weeks = [(w.h3.a.string, [BASE_URL + a['href'] for a in
             w.ul.find_all('a')]) for w in WEEKS]
    numOfWeeks = len(weeks)

    # Choose Week or choose all
    print('%s has %d weeks so far' % (selected_course[0], numOfWeeks))
    w = 0
    for week in weeks:#weeks由元组组成的list
        w += 1
        print('%d - Download %s videos' % (w, week[0].strip()))#week[1]为每周的课程数目
    print('%d - Download them all' % (numOfWeeks + 1))

    w_number = int(input('Enter Your Choice: '))
    while w_number > numOfWeeks + 1:
        print('Enter a valid Number between 1 and %d' % (numOfWeeks + 1))
        w_number = int(input('Enter Your Choice: '))
    #end while
    weeksList=[]
    if w_number == numOfWeeks + 1:
        #links = [link for week in weeks for link in week[1]]
        weeksList=weeks
    else:
        #links = weeks[w_number - 1][1]
        weeksList=[weeks[w_number-1]]
    if is_interactive:
        args.subtitles = input('Download subtitles (y/n)? ').lower() == 'y'

    splitter = re.compile(r'data-streams=(?:&#34;|").*1.0[0]*:')
    re_subs = re.compile(r'data-transcript-translation-url=(?:&#34;|")([^"&]*)(?:&#34;|")')
    extra_youtube = re.compile(r'//w{0,3}\.youtube.com/embed/([^ \?&]*)[\?& ]')
    weektitle=''
    fileExt=['pdf','doc','docx','ppt','pptx','xls','xlsx','zip','rar','7z','mat']
    textFileExt=['m','r','py','jl','ipynb','java','txt','csv']
    fileExt+=textFileExt
    if args.docs and args.fileType!='*':
        fileExt=args.fileType.split(',') #list
    count = 0
    for wk in weeksList:#每周
        weektitle=re.sub(r'\W',' ',wk[0]).strip();#:,;,替换
        print('\n\n'+weektitle+u' , total of weeks: %d'%numOfWeeks);
        video_ids = []#每周都要清空视频及字幕列表以及文件列表
        sub_urls = []
        doc_urls=[]
        for link in wk[1]:#links为该周所有的课程列表
            print("Processing '%s'..." % link)
            page = get_page_contents(link, headers)#获取其中的子页面
            sections = splitter.split(page)[1:]
            for section in sections:
                video_id = section[:YOUTUBE_VIDEO_ID_LENGTH] #11
                sub_url =''
                if args.subtitles:
                    match_subs = re_subs.search(section)
                    if match_subs:
                        sub_url = BASE_URL + match_subs.group(1) + "en" + "?videoId=" + video_id
                if args.docs:
                    #先用a>切分，即 &#34;&gt切分
                    sectionList=section.split(r'&#34;&gt;')
                    for sect in sectionList:
                        res_doc = re.findall(r'href=&#34;.*[^(]\.(?:%s)\b'%'|'.join(fileExt),sect,re.IGNORECASE) #WHO.csv&#34;&gt;WHO.csv
                        for fe in res_doc:
                            tmp=fe[fe.rfind(' href=&#34;',0)+11:]
                            gtFind=tmp.find(r'&gt;')
                            if gtFind>=0:
                                tmp=tmp[0:gtFind] #去掉&gt;之后的，即>
                            fileType=re.compile(r'\.(?:%s)'%'|'.join(fileExt),re.IGNORECASE)
                            type=fileType.search(tmp).group(0)
                            tmp=tmp[:tmp.find(type)+len(type)]#截取.csv之前的字符串
                            print(tmp)
                            if tmp.startswith('https://') or tmp.startswith('http://'):
                                doc_urls+=[tmp]
                            else:
                                doc_urls+=[BASE_URL+tmp] #list,直接追加
                video_ids += [video_id]
                sub_urls += [sub_url]
                
            # Try to download some extra videos which is referred by iframe
            extra_ids = extra_youtube.findall(page)
            video_ids += [link[:YOUTUBE_VIDEO_ID_LENGTH] for link in
                         extra_ids]
            sub_urls += ['' for e_id in extra_ids]
    
        video_urls = ['http://youtube.com/watch?v=' + v_id
                      for v_id in video_ids]#原来都是从youtube上下的,但它怎么知道youtube上一定有呢？
    
        if len(video_urls) < 1:
            print('WARNING: No downloadable video found in '+weektitle)
            #sys.exit(0)
    
        if is_interactive:
            # Get Available Video formats
            os.system('youtube-dl -F %s' % video_urls[-1])
            print('Choose a valid format or a set of valid format codes e.g. 22/17/...')
            args.format = input('Choose Format code: ')
    
        print("[info] Output directory: " + args.output_dir)
        
        target_dir = os.path.join(args.output_dir,
                                      directory_name(selected_course[0]))
        #Download docs
        if args.docs and len(doc_urls)>0:#下载pdf,doc,docx,ppt,pptx,xls,xlsx,csv,r,py,ipynb,jl,m,mat
            print(u'Dowloading the documents...')
            for doc_url in doc_urls:
                fname=os.path.join(os.getcwd(),target_dir,doc_url.split('/')[-1])
                pathName=os.path.dirname(fname);
                if not os.path.exists(pathName):
                    print(u'Create directory:'+pathName)
                    os.makedirs(pathName)
#                     os.system(r'mkdir '+pathName);
                if not os.path.exists(r''+fname):
                    print(u"Downloading file:"+fname,'\nFrom:'+doc_url)
                    try: 
                        downloader(doc_url,fname)
                    except Exception as ex:
                        try:
                            os.system(u'curl -k -# -L -o "'+fname +'" '+doc_url)#curl -o page.html http://www.yahoo.com
                        except Exception as ex_curl:
                            print('Download Error,url:'+doc_url+'\n to:'+fname+'\n---------Ignored!-----------')  
                else:
                    print(fname+u' has existed!')
        else:
            print('No documents for:'+weektitle)
        # Download Videos
        if args.noVideoAndTitle:
            print(u"Don't download video and title\n")
            continue
        for v, s in zip(video_urls, sub_urls):
            count += 1 #每下载一个视频，都要计一次数
            filename_prefix = str(count).zfill(2) #填充为两位
            if args.rename:
               filename_prefix=filename_prefix+' '+weektitle 
            cmd = ["youtube-dl",#其实是使用youtube-dl来下载的
                   "-o", os.path.join(target_dir, filename_prefix + "-%(title)s.%(ext)s")]
            if args.format:
                cmd.append("-f")
                # defaults to mp4 in case the requested format isn't available
                cmd.append(args.format + '/mp4')
            if args.subtitles:
                cmd.append('--write-sub')
            cmd.append(str(v))
    
            popen_youtube = Popen(cmd, stdout=PIPE, stderr=PIPE)
            youtube_stdout = b''
            #youtube-dl自动跳过重复下载
            while True:  # Save output to youtube_stdout while this being echoed
                tmp = popen_youtube.stdout.read(1)
                youtube_stdout += tmp
                print(tmp, end="")
                sys.stdout.flush()
                # do it until the process finish and there isn't output
                if tmp == b"" and popen_youtube.poll() is not None:
                    break
                
            if args.subtitles:
                filename = get_filename(target_dir, filename_prefix)
                subs_filename = os.path.join(target_dir, filename + '.srt')
                if not os.path.exists(r''+subs_filename):
                    subs_string = edx_get_subtitle(s, headers)
                    if subs_string:
                        print('[info] Writing edX subtitles: %s' % subs_filename)
                        open(os.path.join(os.getcwd(), subs_filename),
                             'wb+').write(subs_string.encode('utf-8'))   
                else:
                    print(subs_filename+u' has existed!')
            
        
    print(u"-------Downloads have been done!-------\n")  
def downloader_simple(url,fname):
    import requests
    r = requests.get(url) 
    with open(fname, "wb") as fe:
         fe.write(r.content)
    fe.close()    
def downloader(url,fname): 
    import urllib2
    u = urllib2.urlopen(url)  
    f = open(fname, 'wb')  
    meta = u.info()  
    file_size = int(meta.getheaders("Content-Length")[0])  
      
    file_size_dl = 0  
    block_sz = 1024  
    while True:  
        buffer = u.read(block_sz)  
        if not buffer:  
            break  
      
        file_size_dl += len(buffer)  
        f.write(buffer)  
    f.close() 
                                
def get_filename(target_dir, filename_prefix):
    #returns the basename for the corresponding filename_prefix
    # this whole function is not the nicest thing, but isolating it makes
    # things clearer , a good refactoring would be to get
    # the info from the video_url or the current output, to avoid the
    # iteration from the current dir
    filenames = os.listdir(target_dir)
    subs_filename = filename_prefix
    for name in filenames:  # Find the filename of the downloaded video
        if name.startswith(filename_prefix):
            (basename, ext) = os.path.splitext(name)
            return basename

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCTRL-C detected, shutting down....")
        sys.exit(0)
"""