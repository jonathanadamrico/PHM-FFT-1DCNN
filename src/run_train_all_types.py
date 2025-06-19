import os

#components = ["motor","gearbox","leftaxlebox","rightaxlebox"]
#for component in components:
#    os.system(f"python m0_1d_cnn_ae.py {component}")

for n_fault_type in range(1,18):
#for n_fault_type in [x for x in range(1,18) if x not in (1,10)]:
    fault_type = f"TYPE{n_fault_type}"
    #os.system(f"python3 m2_1d_cnn_ae.py {fault_type}")
    os.system(f"python3 train_fft-unsupervisedAE.py {fault_type} > train_logs/train_logs_{fault_type}.txt")
    #os.system(f"python3 train_fft-convAE.py {fault_type}")
