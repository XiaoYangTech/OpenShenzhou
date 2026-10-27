# -*- coding: utf-8 -*-
"""_paths.py —— 公共路径解析（仓库根、语音包根）

语音包（.dat/.bin/phones.txt 等微软模型文件）不入库，可用环境变量
OS_VOICE_ROOT 指定根目录；否则依次尝试 ../AllVoice、%TEMP%/AllVoice。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)


def voice_root():
    cand = os.environ.get("OS_VOICE_ROOT")
    if cand:
        return cand
    for c in (os.path.join(REPO_ROOT, "..", "AllVoice"),
              os.path.join(os.environ.get("TEMP", ""), "AllVoice")):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(os.environ.get("TEMP", ""), "AllVoice")


def voice_pkg(name):
    return os.path.join(voice_root(), name)


def sdk_root():
    """SDK 代码根目录：优先 OS_SDK_ROOT，其次仓库根（main 分支不含 SDK 时需指向 python 分支）"""
    cand = os.environ.get("OS_SDK_ROOT")
    if cand and os.path.isdir(os.path.join(cand, "openshenzhou")):
        return cand
    if os.path.isdir(os.path.join(REPO_ROOT, "openshenzhou")):
        return REPO_ROOT
    return cand or REPO_ROOT
