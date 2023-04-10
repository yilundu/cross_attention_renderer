import os

mode = 'satori'

if mode == 'satori':
    logging_root = '/nobackup/users/yilundu/my_repos/local_lightfield_networks/logs'
    results_root = '/nobackup/users/yilundu/my_repos/local_lightfield_networks/logs'
    os.environ["TORCH_HOME"] = '/nobackup/users/yilundu'
elif mode == 'openmind':
    logging_root = '/om2/user/yilundu/light_fields_equivariant/logs/light_fields'
    results_root = '/om2/user/yilundu/light_fields_equivariant/results/light_fields'
    figures_root = '/om2/user/yilundu/light_fields_equivariant/results/light_fields/figures'
    data_root = '/om2/user/yilundu/'
    os.environ["TORCH_HOME"] = '/om2/user/yilundu/'
elif mode == 'local':
    logging_root = '/home/sitzmann/test'
    results_root = '/home/sitzmann/test'
    figures_root = '/home/sitzmann/test'
    data_root = '/home/sitzmann/test'
    os.environ["TORCH_HOME"] = '/home/sitzmann/test'
