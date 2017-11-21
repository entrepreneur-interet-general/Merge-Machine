#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:46:03 2017

@author: m75380
"""
from sys import version_info

class NotSupportedException(BaseException): pass

if version_info.major < 3:
    raise NotSupportedException("Only Python 3.x Supported")  