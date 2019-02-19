#-*- encoding=utf8 -*-

import os,sys

if __name__=="__main__":

    print("__file__=%s" % __file__)

    print("os.path.realpath(__file__)=%s" % os.path.realpath(__file__))

    print("os.path.dirname(os.path.realpath(__file__))=%s" % os.path.dirname(os.path.realpath(__file__)))

    print("os.path.split(os.path.realpath(__file__))=%s" % os.path.split(os.path.realpath(__file__))[0])

    print("os.path.abspath(__file__)=%s" % os.path.abspath(__file__))

    print("os.getcwd()=%s" % os.getcwd())

    print("sys.path[0]=%s" % sys.path[0])

    print("sys.argv[0]=%s" % sys.argv[0])

    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("tensorflow-learn\\")+len("tensorflow-learn\\")]  # 获取myProject，也就是项目的根路径

    print("sys.argv[0]=%s" % rootPath)

    rootPath = os.path.split(os.path.realpath(__file__))[0]
    print("sys.argv[0]=%s" % rootPath)

