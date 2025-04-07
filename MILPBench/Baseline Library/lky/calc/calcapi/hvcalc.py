import subprocess
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description="receive input to be calculated from higher level")
parser.add_argument("--input_path", required=True, type=str, help="the input path")
args = parser.parse_args()

input_path = args.input_path

if os.path.exists('calc/hbda/build/nonincremental/tmp.txt'):
    os.remove('calc/hbda/build/nonincremental/tmp.txt')
    
shutil.copy(input_path, 'calc/hbda/build/nonincremental/tmp.txt')

subprocess.run(['./calc/hbda/build/nonincremental/nonincremental', '-O', 'calc/hbda/build/nonincremental/tmp.txt'])