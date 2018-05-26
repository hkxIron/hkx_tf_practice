import func as func
# python 反射
fn = getattr(func, "f1")
fn("hello")

fn = getattr(func, "f2")
fn("world")

"""
function f1: hello
function f2: world
"""
