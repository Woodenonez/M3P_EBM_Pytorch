import os
import yaml

from pathlib import Path

'''
This is used to load and dump parameters in the form of YAML
'''

file_name = 'sidv2e_1t10_train.yml'
sl_path = os.path.join(Path(__file__).resolve().parents[2], 'config/', file_name)

general_param = {'pred_len'   : 10, 
                 'obsv_len'   : 5, 
                 'device' : 'multi',
                 }
general_param['input_channel'] = (general_param['obsv_len']) + 1

training_param = {'epoch'                : 20, 
                  'batch_size'           : 20, 
                  'early_stopping'       : 0,
                  'learning_rate'        : 1e-3,
                  'weight_regularization': 1e-4,
                  'checkpoint_dir'       : 'Model/SID2e'
                  }

path_param = {'model_path': 'Model/sidv2e_1t10',
              'data_name':  'SID_1t10_train/5',
              }
path_param['data_path']  = os.path.join('data/', path_param['data_name'])
path_param['label_path'] = os.path.join('data/', path_param['data_name'], 'all_data.csv')

def to_yaml(data, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def to_yaml_all(data_list, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump_all(data_list, f, explicit_start=True, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def from_yaml(load_path, vb=True):
    with open(load_path, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            if vb:
                print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

def from_yaml_all(load_path, vb=True):
    with open(load_path, 'r') as stream:
        parsed_yaml_list = []
        try:
            for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                parsed_yaml_list.append(data)
            if vb:
                print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml_list

if __name__ == '__main__':

    # param = {**general_param, **training_param, **converting_param, **path_param}
    # to_yaml(param, sl_path, style=None)
    # test_dict = from_yaml(sl_path) # ensure the yaml file is saved

    param_list = [general_param, training_param, path_param]
    to_yaml_all(param_list, sl_path, style=None)
    test_dict_list = from_yaml_all(sl_path) # ensure the yaml file is saved