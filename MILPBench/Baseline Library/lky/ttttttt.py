import re
a = "IS_easy_instance_0.lp"
w = re.match(r"S_easy_instance_[0-9]+", a)
print(w)