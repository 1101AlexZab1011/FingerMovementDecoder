import mne
from mne.datasets import multimodal
import os
import tensorflow as tf
import mneflow


if __name__ == '__main__':

    mne.set_log_level(verbose='CRITICAL')
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.autograph.set_verbosity(0)

    fname_raw = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')
    raw = mne.io.read_raw_fif(fname_raw)
    cond = raw.acqparser.get_condition(raw, None)
    # get the list of condition names
    condition_names = [k for c in cond for k, v in c['event_id'].items()]
    epochs_list = [mne.Epochs(raw, **c) for c in cond]
    epochs = mne.concatenate_epochs(epochs_list)
    epochs = epochs.pick_types(meg='grad')
    # print(epochs.info)
    print(tf.__version__)
    import sys
    current_dir = os.path.dirname(os.path.abspath('./'))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # Specify import options
    import_opt = dict(
        savepath='../tfr/',  # path where TFR files will be saved
        out_name='mne_sample_epochs',  # name of TFRecords files
        fs=600,
        input_type='trials',
        target_type='int',
        picks={'meg': 'grad'},
        scale=True,  # apply baseline_scaling
        crop_baseline=True,  # remove baseline interval after scaling
        decimate=None,
        scale_interval=(0, 60),  # indices in time axis corresponding to baseline interval
        n_folds=5,  # validation set size set to 20% of all data
        overwrite=True,
        segment=False,
        test_set='holdout'
    )
    # write TFRecord files and metadata file to disk
    meta = mneflow.produce_tfrecords([epochs], **import_opt)

    dataset = mneflow.Dataset(meta, train_batch=100)
    # specify model parameters
    lf_params = dict(
        n_latent=32,  # number of latent factors
        filter_length=17,  # convolutional filter length in time samples
        nonlin=tf.nn.relu,
        padding='SAME',
        pooling=5,  # pooling factor
        stride=5,  # stride parameter for pooling layer
        pool_type='max',
        model_path=import_opt['savepath'],
        dropout=.5,
        l1_scope=["weights"],
        l1=3e-3
    )

    model = mneflow.models.LFCNN(dataset, lf_params)
    model.build()

    import sklearn
    print(sklearn.__version__)

    from importlib.metadata import version
    print(f'mneflow: {version("mneflow")}')
    print(f'matplotlib: {version("matplotlib")}')
    print(f'mne: {version("mne")}')
    print(f'numpy: {version("numpy")}')
    print(f'scipy: {version("scipy")}')
    print(f'sklearn: {sklearn.__version__}')
    print(f'tensorflow: {version("tensorflow")}')

    model.train(n_epochs=25, eval_step=100, early_stopping=5)
    model.plot_hist()
