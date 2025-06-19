import os

#components = ["motor","gearbox","leftaxlebox","rightaxlebox"]
#for component in components:
#    os.system(f"python m0_1d_cnn_ae.py {component}")

for n_fault_type in range(1,18):
    fault_type = f"TYPE{n_fault_type}"
    os.system(f"python3 infer_fft-convAE.py {fault_type}")

