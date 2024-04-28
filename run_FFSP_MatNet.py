from solutions.FFSP_MatNet import train
from solutions.FFSP_MatNet import test

def main(execute_mode):
    if execute_mode == 'train':
        train.main()
    if execute_mode == 'test':
        test.main()

if __name__ == "__main__":
    main('test')
    # 'train': train the model using FFSPEnv
    # 'test': test using trained network with FFSPEnv_test (JobShopModule)