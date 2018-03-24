OLD_IFS=$IFS
IFS=" "
for x in $(echo "one two three")
do
    echo "print: $x"
done
IFS="$OLD_IFS"

