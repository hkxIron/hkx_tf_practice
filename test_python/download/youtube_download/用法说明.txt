1.直接下载
E:\kp\lunaWorkspace\youtube_downloadProj>youtube-dl.exe  https://www.youtube.com
/watch?v=NfnWJUyUJYU   -o %(title)s.%(ext)s  --write-auto-sub --verbose

youtube-dl.exe -v 可以看到版本等信息
youtube-dl.exe -U 可以更新


于是又开始google新的下载方法，结果发现了神器youtube-dl. 作者还很贴心给准备了可执行文件。youtube-dl提供了100多个选项，肯定能满足你的需求。但是由于youtube-dl给提供了100多个选项使用的时候肯定有很多疑惑。Linux可以用alias方便使用。windows需要写个bat包装下使用。像我用的downyt.bat,在命令行直接downyt.bat youtube-url 按自己的配置下载了。

downyt.bat d:\usrbin\youtube-dl.exe %* --proxy "https://127.0.0.1:8080" --write-sub --no-mtime --output %%(title)s_%%(resolution)s.%%(ext)s --no-part --all-subs --restrict-filenames

选项	含义
proxy	在天朝你没个代理你怎么混啊
write-sub	下载字幕，这里的字幕是用户上传的字幕，不是youtube自动生成的CC字幕
all-subs	如果有字幕的话，就下载所有字幕，收集癖:)
restrict-filenames	避免在下载的文件名中使用:等特殊字符
no-mtime	不修改文件的mtime，文件的更新时间为下载的时间，这样方便管理
output	下载文件命名模板，还是为了方便管理
 注意: youtube-dl默认是再下载清晰度最


2.从文件列表中下载
python downloader_from_list.py list_mlds17.txt data
