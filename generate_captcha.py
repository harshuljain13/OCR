import subprocess
import os
import sys

try:
    os.makedirs('../data/captcha/test')
except Exception:
    print Exception

try:
    os.makedirs('../data/captcha/train')
except Exception:
    print Exception

for i in range(100):
    #print('\r%d' % i, flush=True )
    subprocess.call('./captcha -f Ubuntu-Mono ../data/captcha/test/Image%s.png' % i, shell=True)
subprocess.call('mv captcha_bash_annotations.csv ../data/captcha/captcha_bash_test_annotations.csv', shell=True)

for i in range(100):
    subprocess.call('./captcha -f Ubuntu-Mono ../data/captcha/train/Image%s.png' % i, shell=True)
subprocess.call('mv captcha_bash_annotations.csv ../data/captcha/captcha_bash_train_annotations.csv', shell=True)
