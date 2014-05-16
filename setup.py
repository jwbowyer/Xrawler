#!/usr/bin/env python

#docs.python.org/2.7/distutils/setupscript.html

import os
import sys

from distutils.core import setup

def XrawlerVersion():

   ffile="README"

   if os.path.isfile(ffile)!=1:
    print "No README file!"
    sys.exit(1)

   try:
     with open(ffile) as f:
	for line in f:
	 Version=line
	 return Version

   except:
     print "Cannot get version number"
     sys.exit(1)

   return Version

#docs.python.org/2/distutils/apiref.html
setup(name="Xrawler",description='Filter your arXiv RSS feeds',author='J. Bowyer',  author_email='j.bowyer07@imperial.ac.uk',version=XrawlerVersion(),py_modules=["Xrawler"],url="http://github.com/jwbowyer/Xrawler")





