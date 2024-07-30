import os

components = ["motor","gearbox","leftaxlebox","rightaxlebox"]
for component in components[:]:
    os.system(f"python m0_1d_cnn.py {component}")

for n_fault_type in range(1,17):
    fault_type = f"TYPE{n_fault_type}"
    os.system(f"python m2_1d_cnn.py {fault_type}")

