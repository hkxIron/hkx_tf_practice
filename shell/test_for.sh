#!/usr/bin/env bash
OLD_IFS=$IFS
IFS=" "
for x in $(echo "one two three")
do
    echo "print: $x"
done
IFS="$OLD_IFS"
OLD_IFS=$IFS
IFS=" "
for x in $(echo "one two three")
do
    echo "print: $x"
done
IFS="$OLD_IFS"

echo "或者使用数组"
TABLES=("houyi_rules" "domain_dicts") # shell数组
for TABLE in "${TABLES[@]}"; do
    echo "$TABLE"
done
