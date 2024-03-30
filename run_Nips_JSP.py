from solutions.JSP_Nips import training
from solutions.JSP_Nips import test

def main(execute_mode):
    if execute_mode == 'train':
        training.main()
    if execute_mode == 'test':
        test.main(test_mode='benchmark', instance_class='taillard')
        # optional test_mode and instance_class:
        # 1 generatedData generatedData
        # 2 benchmark dmu
        # 3 benchmark tai
        # 4 benchmark adams
        # 5 benchmark taillard
        # please specify the test file in configs

if __name__ == "__main__":
    main('train')
    # 'train' for training with JSSP_Env
    # 'test' for test using trained network with test_env (JobShopModule)