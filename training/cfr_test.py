from cfr import CFR
import sys
import os
sys.path.append(os.getcwd())

if __name__ == "__main__":


    cfr_iters = 1
    mcc_iters = 10
    round_iters = 10
    res_size = 4

    test_cfr = CFR(cfr_iters, mcc_iters, round_iters, res_size)
    test_cfr.CFR_training()
