import os
import tarfile
import requests
import shutil
import time


def run_data_retriever(url, target):
    # download dataset from url provided
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        print('Retrieving dataset from website')
        with open(target, 'wb') as f:
            f.write(response.content)
        print('Done!\nFile saved as: {}'.format(target))
    print('Proceeding to unzip contents...')
    time.sleep(5)

    # extract zipped folder contents
    opener, mode = tarfile.open, 'r:gz'
    with opener(target, mode) as o:
        print('Unzipping contents')
        o.extractall()
        print('Done!')
    print('Moving electronics folder contents to dataset folder...')
    time.sleep(3)

    # remove all but the electronics folder from dataset
    path = os.path.join('sorted_data_acl', 'electronics/')
    new_path = os.path.join('dataset/')

    for f in os.listdir(path):
        print('Moving files')
        shutil.copy(path+f, new_path+f)
    print('Done!\nCleaning up...')
    time.sleep(5)

    shutil.rmtree('sorted_data_acl')
    print('Extracted folder removed!')
    os.remove(target)
    print('Downloaded zipped folder removed!')
    print('Done!')
