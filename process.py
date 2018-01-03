from os import listdir
from os.path import isfile, join, basename, splitext
from os import system, remove
import argparse
import pickle

def conv_dirs(FLAGS, directory, files):
    data = []
    for obj in files:
        file_dir = join(directory, obj)
        if '.DS_Store' in file_dir:
            continue
        file_dir_base = splitext(file_dir)[0]
        if (not isfile('{}.obj'.format(file_dir_base))) or FLAGS.ow:
            with open(file_dir, 'r') as f:
                lines = f.readlines()
            if lines[0].strip() != 'OFF':
                lines[0] = lines[0].replace('OFF', '')
                new_lines = ['OFF\n']
                new_lines += lines
                with open(file_dir, 'w') as f:
                    f.writelines(new_lines)
            system('meshconv -c obj {} > /dev/null'.format(file_dir))
        if not isfile('{}.pcd'.format(file_dir_base)) or FLAGS.ow:
            system('./pcl_mesh_sampling {}.obj {}.pcd -n_samples {} -no_vis_result > /dev/null'.format(file_dir_base, file_dir_base, FLAGS.n_samples))

        if FLAGS.remove:
            remove('{}.obj'.format(file_dir_base))
            remove('{}.off'.format(file_dir_base))

        pcd_file = '{}.pcd'.format(file_dir_base)
        if not isfile(pcd_file): # THis file is not converted :/
            continue
        with open(pcd_file, 'r') as f:
            points = f.readlines()
        points = points[11:FLAGS.n_samples+11]
        points = [list(map(float, p.split(' '))) for p in points]
        data.append(points)
    return data

folders = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand',
           'sofa', 'table', 'toilet']

parser = argparse.ArgumentParser()
parser.add_argument('--remove', dest='remove', action='store_true', default=False, help='Remove .off and .obj files after conversion')
parser.add_argument('--ow', dest='ow', action='store_true', default=False, help='Overwrite .obj files if they exist')
parser.add_argument('--n_samples', dest='n_samples', default=2048, help='Number of samples on edges')
FLAGS = parser.parse_args()

train = {}
test = {}

for folder in folders:
    train_dir = join(folder, 'train')
    test_dir = join(folder, 'test')
    train_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    test_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

    print("Converting {} train data".format(folder))
    train_data = conv_dirs(FLAGS, train_dir, train_files)
    print("Converting {} test data".format(folder))
    test_data = conv_dirs(FLAGS, test_dir, test_files)

    train[folder] = train_data
    test[folder] = test_data

pickle.dump(train, open('model40_train.pk', 'wb'))
pickle.dump(test, open('model40_test.pk', 'wb'))
