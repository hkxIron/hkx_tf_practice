#coding:utf-8       #设置python文件的编码为utf-8，这样就可以写入中文注释
import json
objStr="""
[
    {
        "height": 1080,
        "path": "TB2jX7RpXXXXXb5XpXXXXXXXXXX_!!0-dgshop.jpg",
        "type": "pic",
        "width": 1080
    },
    {
        "desc": "衬衫",
        "flag": 1,
        "height": 800,
        "itemId": 531418059893,
        "path": "TB2qIyjoVXXXXalXpXXXXXXXXXX_!!14197825.jpg",
        "price": 11200,
        "title": "T5445-2016夏新款女装韩版圆领钩花镂空性感露肩衬衫上衣 0516",
        "type": "item",
        "width": 800,
        "xpos": 68,
        "ypos": 21
    }
]
"""
jsObj=json.loads(objStr,encoding="utf-8")
json.dumps()

print jsObj