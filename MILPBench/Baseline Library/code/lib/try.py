import re
AA = "IS_easy_instance_01.lp"
BB = re.match(r"(.*)_[0-9]+", AA)
if BB == None:
    print("!@!@")
else :
    print(BB.group(1))