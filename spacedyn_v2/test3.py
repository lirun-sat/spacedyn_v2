# -*- coding:utf-8 -*-
import test2 as gol

gol._init()  # 先必须在主模块初始化（只在Main模块需要一次即可）

# 定义跨模块全局变量
gol.set_value('CODE', 'UTF-8')
gol.set_value('PORT', 80)
gol.set_value('HOST', '127.0.0.1')

