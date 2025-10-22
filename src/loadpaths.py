'''
Use by:

import loadpaths
user_paths_dict = loadpath.loadpaths()
'''


import os
import json
import sys
import getpass
from pathlib import Path

def find_vcs_root(test=os.getcwd(), dirs=(".git",), default=None):
    '''From https://stackoverflow.com/a/43786287/21221244'''
    prev, test = None, os.path.abspath(test)
    while prev != test:
        if any(os.path.isdir(os.path.join(test, d)) for d in dirs):
            return test
        prev, test = test, os.path.abspath(os.path.join(test, os.pardir))
    return default

def loadpaths(username=None):
    '''Function that loads data paths from json file based on computer username'''

    ## Get json:
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent, 'content/data_paths.json')
    json_path = str(json_path)

    if username is None:
        username = getpass.getuser()  # get username of PC account
        if username == 'runner':
            if os.uname().sysname == 'Darwin':
                username = 'runner_Darwin'
            else:
                username = 'runner_Linux'

    ## Load paths corresponding to username:
    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        if username not in config_info.keys():
            print(f'WARNING: Using default paths and example data. To fix, please add your username <{username}> and data paths to content/data_paths.json')
            username = 'default'
        user_paths_dict = config_info[username]['paths']  # extract paths from current user
    
    if username == 'default':
        user_paths_dict['repo'] = find_vcs_root(__file__)
        user_paths_dict['home'] = os.path.expanduser('~')
        # user_paths_dict['s2bms_images'] = os.path.join(user_paths_dict['repo'], 'tests/data_test/images_tests')
        # user_paths_dict['s2bms_presence'] = os.path.join(user_paths_dict['repo'], 'tests/data_test/presence_tests/ukbms_presence_test16.csv')
    
    # Expand tildes in the json paths
    user_paths_dict = {k: str(v) for k, v in user_paths_dict.items()}
    return {k: os.path.expanduser(v) for k, v in user_paths_dict.items()}

if __name__ == '__main__':
    # print(find_vcs_root(__file__))
    print(loadpaths('test'))
