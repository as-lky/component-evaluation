import subprocess


# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "MIKS", "--instance_path", f"./Dataset/MIKS_fakemedium_instance/MIKS_fakemedium_instance/LP/MIKS_fakemedium_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "4000", "--benchmark_path", "./Dataset/MIKS_fakemedium_instance/MIKS_fakemedium_instance/Pickle/"] for i in range(9)
# ]

# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "SC", "--instance_path", f"./Dataset/SC_fakemedium_instance/SC_fakemedium_instance/LP/SC_fakemedium_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "4000", "--benchmark_path", "./Dataset/SC_fakemedium_instance/SC_fakemedium_instance/Pickle/"] for i in range(3, 10)
# ]

# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "IS", "--instance_path", f"./Dataset/IS_fakehard_instance/IS_fakehard_instance/LP/IS_fakehard_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "18000", "--benchmark_path", "./Dataset/IS_fakehard_instance/IS_fakehard_instance/Pickle/"] for i in range(2)
# ]

# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "MIKSC", "--instance_path", f"./Dataset/MIKSC_easy_instance/MIKSC_easy_instance/LP/MIKSC_easy_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "4000", "--benchmark_path", "./Dataset/MIKSC_easy_instance/MIKSC_easy_instance/Pickle/"] for i in range(3, 30)
# ]

# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "MT", "--instance_path", f"./Dataset/MT_easy_instance/MT_easy_instance/LP/MT_easy_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "10000", "--benchmark_path", "./Dataset/MT_easy_instance/MT_easy_instance/Pickle/"] for i in range(0, 25)
# ]

# commands = [
#     ["python", "main.py", "--device", "cuda", "--taskname", "MT", "--instance_path", f"./Dataset/MT_medium_instance/MT_medium_instance/LP/MT_medium_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "10000", "--benchmark_path", "./Dataset/MT_medium_instance/MT_medium_instance/Pickle/"] for i in range(0, 25)
# ]

commands = [
    ["python", "main.py", "--device", "cuda", "--taskname", "MT", "--instance_path", f"./Dataset/MT_2000_instance/MT_2000_instance/LP/MT_2000_instance_{i}.lp", "--graphencode", "default", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "10000", "--benchmark_path", "./Dataset/MT_2000_instance/MT_2000_instance/Pickle/"] for i in range(0, 25)
]


processes = [subprocess.Popen(cmd) for cmd in commands]

for p in processes:
    p.wait()
    
    