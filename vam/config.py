import ml_collections
import numpy as np


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_default_config():
    """Get the hyperparameters for the model"""
    config = ml_collections.ConfigDict()
    config.expt_name = None
    config.project = None
    config.tags = None
    config.notes = None

    config.model = d(
        conv_n_features=[64, 64, 128, 128, 128, 256],  # original
        fc_n_units=[1024],
        w_scale=0.1,
        n_acc=4,
        a_init=-0.35,
        c_init=-0.35,
        t0_init=-0.9,
        s=1.0,
        n_mc=10,
        dropout_rate=0.5,
        param_dtype=np.float32,
        elbo_type="local",
        test_only=False,
        model_type="vam",
    )

    config.training = d(
        seed=0,
        train_frac=0.65,
        val_frac=0.15,
        batch_size=256,
        n_epochs=225,
        metrics_every=10,
        pretrained_model=None,
        n_pretrained_layers=2,  # layers from VGG to transfer
        n_batches_plot=0,
    )

    config.optimizer = d(
        clip_val=5.0,  # gradient clipping value
        # CNN optimizer
        cnn_opt="adam",
        cnn_lr=1e-3,
        cnn_momentum=0.9,  # if using e.g. SGD
        # Variational inference optimizer
        vi_opt="adam",
        vi_lr=1e-3,
    )

    config.data = d(
        data_prep_type="slow",
        n_prep=None,
        img_h=128,
        img_w=128,
        do_data_aug=True,
        translate_xmax=2,
        translate_ymax=1,
        p_warp=0.75,
        warp_str=3,
        warp_coarseness=32,
        splits=["train", "val"],
    )

    return config


def get_task_opt_config():
    config = get_default_config()
    config.model.model_type = "task_opt"
    config.training.n_epochs = 30
    config.training.metrics_every = 1
    return config


def get_binned_rt_config(n_rt_bins, rt_bin):
    config = get_default_config()
    config.model.model_type = "binned_rt"
    config.training.n_epochs = 600
    config.training.metrics_every = 10
    config.data.data_prep_type = "binned_rt"
    config.data.n_rt_bins = n_rt_bins
    config.data.rt_bin = rt_bin
    return config


def get_config_from_cli(args):
    """Update the config from command line arguments."""
    if args.model_type == "vam":
        config = get_default_config()
    elif args.model_type == "task_opt":
        config = get_task_opt_config()
    elif args.model_type == "binned_rt":
        config = get_binned_rt_config(args.n_rt_bins, args.rt_bin)
    config.project = args.project
    config.notes = args.notes
    config.expt_name = args.expt_name
    return config


def get_test_config(model_type, user_id, n_rt_bins=None, rt_bin=None):
    config = get_default_config()
    config.model.test_only = True
    config.data.splits = ["test"]

    if model_type == "vam":
        config.expt_name = f"user{user_id}"
    elif model_type == "task_opt":
        config.expt_name = f"user{user_id}"
        config.model.model_type = "task_opt"
    elif model_type == "binned_rt":
        config.expt_name = f"user{user_id}_rt_bin{rt_bin}"
        config.data.data_prep_type = "binned_rt"
        config.data.n_rt_bins = n_rt_bins
        config.data.rt_bin = rt_bin

    return config
