import os
import re
#folder = 'Dataset/IS_test_instance/IS_test_instance/LP' 
folder = 'logs/work/IS'
dirs = os.listdir(folder)
for dir in dirs:
    if not os.path.isdir(os.path.join(folder, dir)):
        continue
    files = os.listdir(os.path.join(folder, dir))
    for file in files:
        if os.path.isdir(os.path.join(folder, dir, file)):
            continue
        tmp = re.match(r"(.*)_[0-9].*", file)
        tmp = tmp.group(1)
        if not os.path.exists(os.path.join(folder, dir, tmp)):
            os.makedirs(os.path.join(folder, dir, tmp))
        os.rename(os.path.join(folder, dir, file), os.path.join(folder, dir, tmp, file))
    print(dir)
# for filename in files:
#     old_path = os.path.join(folder, filename)
# #    new_filename = "IS_fakemedium_" + filename
#     new_filename = "IS_" + filename
#     new_path = os.path.join(folder, new_filename)
#     print(old_path, new_path)
#     os.rename(old_path, new_path)
    
    
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_10.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_11.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_12.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_13.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_14.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_15.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_16.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_17.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_18.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_19.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_20.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_21.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_22.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_23.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_24.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_25.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_26.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_27.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_28.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/
#python main.py --device cuda:2 --taskname IS --instance_path ./Dataset/IS_test_instance/IS_test_instance/LP/IS_test_instance_29.lp --graphencode default --predict gurobi --predict_time_limit 30 --modify default --search gurobi --search_time_limit 30 --benchmark_path ./Dataset/IS_test_instance/IS_test_instance/Pickle/