#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
#File    : replace.py
#Author  : ganguohua
#Time    : 2021/10/19 8:33 下午
"""
import re
string1=''' {
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceRoot}",
                "D:/mingw-w64/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++",
                "D:/mingw-w64/mingw64/lib\gcc_64-w64-mingw32\8.1.0\includWe\c++_64-w64-mingw32",
                "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/backward",
                "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include",
                "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/tr1",
                "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/x86_64-w64-mingw32/include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "__GNUC__=6",
                "__cdecl=__attribute__((__cdecl__))"
            ],
            "intelliSenseMode": "msvc-x64",
            "browse": {
                "path": [
                    "${workspaceRoot}",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/x86_64-w64-mingw32",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/backward",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/lib/gcc/x86_64-w64-mingw32/8.1.0/include/c++/tr1",
                    "D:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/x86_64-w64-mingw32/include"
                ]
            },
            "limitSymbolsToIncludedHeaders": true,
            "databaseFilename": "",
            "compilerPath": "D:\mingw-w64\mingw64\bin\gcc.exe",
            "cStandard": "gnu17",
            "cppStandard": "gnu++14"
        }
    ],
    "version": 4
}  '''
ans=string1.replace('/','\\')
ans=ans.replace('')
print(ans)
