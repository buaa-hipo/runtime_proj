#!/usr/bin/python3

import os
import operator
import sys

files = {}
for x in os.listdir(sys.argv[1]):
    if x.endswith(".doxy"):
        with open(sys.argv[1]+x, "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                if "\page" in line:
                    line = line.replace("/*! \page ", "").strip()
                    files[x] = line[0:line.index(" ")]+".html"

sfiles= dict(sorted(files.items(), key=operator.itemgetter(0)))
htmlfiles = ["index.html"]
htmlfiles.extend(sfiles.values())

htmldir=sys.argv[2]+"/"

chapter=0
for x in htmlfiles:
    chapter+=1
    section=0
    with open(htmldir+x, "r", encoding="utf-8") as fin:
        with open(htmldir+x+".count.html", "w", encoding="utf-8") as fout:
            for line in fin.readlines():
                if "<div class=\"title\">" in line:
                    line = line.replace("<div class=\"title\">", "<div class=\"title\">"+str(chapter)+". ")
                if "<h1>" in line:
                    section += 1
                    line = line.replace("<h1>", "<h1>" + str(chapter) + "." + str(section))
                    subsection = 0
                if "<h2>" in line:
                    subsection += 1
                    line = line.replace("<h2>", "<h2>" + str(chapter) + "." + str(section) + "." + str(subsection))
                    subsubsection = 0
                if "<h3>" in line:
                    subsubsection += 1
                    line = line.replace("<h3>", "<h3>" + str(chapter) + "." + str(section) + "." + str(subsection) + "." + str(subsubsection))
                fout.write(line)
    os.rename(htmldir+x+".count.html", htmldir+x)
