path="$1"
files="$(find $path -iname '*.webm' -or -name '*.mkv')"
OLD_IFS="$IFS"
#IFS=$'\n' # 或者如下格式均可
IFS="
"
for full_name in $files # 此处不要加""，否则无法分割
do
    file_name="${full_name%.*}" 
    ext_name="${full_name##*.}" 
    new_name="${file_name}.mp4"
    echo "process file:$full_name file_name:$file_name ext_name:$ext_name"
    if [ ! -f $new_name ];then
        set -x
        ffmpeg -i "$full_name" -strict -2 "$new_name"
        set +x
    else
        echo "$new_name exists, skip"
    fi
done
IFS="$OLD_IFS"


