#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口：python -m openshenzhou --voice <语音包> --text "..." --out out.wav"""
import sys

from . import _main

if __name__ == "__main__":
    sys.exit(_main())
