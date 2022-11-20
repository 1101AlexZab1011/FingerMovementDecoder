import sys
import os
import inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

if __name__ == '__main__':
    for subject in os.listdir('./Source/Subjects'):
        nn_path = os.path.join('./Source/Subjects', subject, 'LFCNN')
        subject_params = os.path.join(nn_path, 'Parameters')
        if os.path.exists(subject_params):
            shutil.rmtree(subject_params)
        subject_preds = subject_params = os.path.join(nn_path, 'Predictions')
        if os.path.exists(subject_preds):
            shutil.rmtree(subject_preds)
