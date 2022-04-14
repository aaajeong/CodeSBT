import sys
import re
import os

dataset = 'emse'
model = 'hybrid'
log_file = os.path.abspath(os.path.join(os.curdir, os.pardir))+'/'+dataset+'/model/'+model+'/log.txt'
print(log_file)