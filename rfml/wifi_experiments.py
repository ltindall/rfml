from pathlib import Path

from rfml.experiment import *
from rfml.train_iq import *
from rfml.train_spec import *


# Ensure that data directories have sigmf-meta files with annotations
# Annotations can be generated using scripts in label_scripts directory or notebooks/Label_WiFi.ipynb and notebooks/Label_DJI.ipynb

experiments = {
    "experiment_test": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-sd-gr-ieee-wifi/test_offline"],
        "iq_epochs": 10,
        "spec_epochs": 0,
        "notes": "TESTING",
    },
    "experiment_nz_wifi_ettus": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, Ettus B200Mini, anarkiwi collect",
    },
    "experiment_nz_wifi_ettus_blade": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect",
    },
    "experiment_nz_wifi_ettus_ap": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["data/gamutrf/gamutrf-nz-wifi"],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on Ettus B200Mini RX/TX, validate on real Wi-Fi AP TX & Ettus B200Mini RX, anarkiwi collect",
    },
    "experiment_emair": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "data/gamutrf/wifi-data-03082024/20msps/normal/train",
            "data/gamutrf/wifi-data-03082024/20msps/normal/test",
            "data/gamutrf/wifi-data-03082024/20msps/normal/inference",
            "data/gamutrf/wifi-data-03082024/20msps/mod/train",
            "data/gamutrf/wifi-data-03082024/20msps/mod/test",
            "data/gamutrf/wifi-data-03082024/20msps/mod/inference",
        ],
        "val_dir": [
            "data/gamutrf/wifi-data-03082024/20msps/normal/validate",
            "data/gamutrf/wifi-data-03082024/20msps/mod/validate",
        ],
        "iq_num_samples": 16 * 25,
        "iq_epochs": 10,
        "iq_batch_size": 16,
        "spec_batch_size": 32,
        "spec_epochs": 40,
        "spec_n_fft": 16,
        "spec_time_dim": 25,
        "notes": "Ettus B200Mini RX, emair collect",
    },
    "experiment_nz_wifi_blade": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, BladeRF, anarkiwi collect",
    },
    "experiment_nz_wifi_blade_ettus": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["data/gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": [
            "data/gamutrf/gamutrf-nz-anon-wifi",
            "data/gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect",
    },

    "experiment_train_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on BladeRF TX & Ettus B200Mini RX, validate on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_train_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_train_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, train on BladeRF TX & Ettus B200Mini RX, validate on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_train_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 40,
        "spec_epochs": 0,
        "spec_batch_size": -1,
        "spec_skip_export": True,  # USE WITH CAUTION (but speeds up large directories significantly): skip after first run if using separate train/val directories
        "notes": "Wi-Fi vs anomalous Wi-Fi, validate on BladeRF TX & Ettus B200Mini RX, train on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on Ettus B200Mini RX/TX, anarkiwi collect 1",
    },
    "experiment_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect 1",
    },
    "experiment_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on BladeRF TX & Ettus B200Mini RX, anarkiwi collect 2",
    },
    "experiment_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 200,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train and validate on Ettus B200Mini RX/TX, anarkiwi collect 2",
    },
    "experiment_ettus_1_to_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_2_to_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 2, validate Ettus 1",
    },
    "experiment_blade_1_to_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Blade 1, validate Blade 2",
    },
    "experiment_blade_2_to_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Blade 2, validate Blade 1",
    },
    "experiment_ettus_1_blade_1_to_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_blade_2_to_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": ["/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf"
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_to_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
        ],
        "iq_epochs": 150,
        "iq_learning_rate": 0.0000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_blade_1_to_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 150,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_2_blade_1_blade_2_to_ettus_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.0000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_blade_1_blade_2_to_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.0000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_ettus_2_blade_1_to_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_ettus_2_blade_2_to_blade_1": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-wifi-and-anom-bladerf",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.0000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_1_to_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-nz-anon-wifi",
            "/data/s3_gamutrf/gamutrf-nz-nonanon-wifi",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 100,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_ettus_2_to_blade_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "iq_epochs": 150,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    "experiment_blade_2_to_ettus_2": {
        "class_list": ["wifi", "anom_wifi"],
        "train_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/blade/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/blade/",
        ],
        "val_dir": [
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx/ettus/",
            "/data/s3_gamutrf/gamutrf-anom-wifi2/collect/wifi_tx_mod/ettus/",
        ],
        "iq_epochs": 150,
        "iq_learning_rate": 0.000001,
        "spec_epochs": 0,
        "notes": "Wi-Fi vs anomalous Wi-Fi, train Ettus 1, validate Ettus 2",
    },
    # ettus1, blade1, blade2

}


if __name__ == "__main__":

    experiments_to_run = [
        # "experiment_test",
        # "experiment_nz_wifi_ettus",
        # "experiment_nz_wifi_ettus_blade",
        # "experiment_nz_wifi_ettus_ap",
        # "experiment_emair",
        # "experiment_nz_wifi_blade",
        # "experiment_nz_wifi_blade_ettus",
        # "experiment_train_blade_1",
        # "experiment_train_ettus_1",
        # "experiment_train_blade_2",
        # "experiment_train_ettus_2",
        # "experiment_ettus_1",
        # "experiment_blade_1",
        # "experiment_ettus_2",
        # "experiment_blade_2",
        # "experiment_ettus_1_to_2",
        # "experiment_ettus_2_to_1",
        # "experiment_blade_1_to_2",
        # "experiment_blade_2_to_1",
        # "experiment_ettus_1_blade_1_to_blade_2",
        # "experiment_ettus_1_blade_2_to_blade_1",
        # "experiment_ettus_1_blade_1",
        # "experiment_ettus_1_blade_2",
        # "experiment_ettus_2_blade_1_blade_2_to_ettus_1",
        # "experiment_ettus_1_blade_1_blade_2_to_ettus_2",
        "experiment_ettus_1_ettus_2_blade_1_to_blade_2",
        # "experiment_ettus_1_ettus_2_blade_2_to_blade_1",
        # "experiment_ettus_1_to_blade_2",
        "experiment_blade_2_to_ettus_2",
        "experiment_ettus_2_to_blade_2",
        "experiment_ettus_1_to_blade_1",
        "experiment_blade_1_to_ettus_1",

    ]

    for experiment_name in experiments_to_run:
        print(f"Running {experiment_name}")
        try:
            exp = Experiment(
                experiment_name=experiment_name, **experiments[experiment_name]
            )

            logs_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

            if exp.iq_epochs > 0:
                train_iq(
                    train_dataset_path=exp.train_dir,
                    val_dataset_path=exp.val_dir,
                    num_iq_samples=exp.iq_num_samples,
                    only_use_start_of_burst=exp.iq_only_start_of_burst,
                    epochs=exp.iq_epochs,
                    batch_size=exp.iq_batch_size,
                    class_list=exp.class_list,
                    output_dir=Path("experiment_logs", exp.experiment_name),
                    logs_dir=Path("iq_logs", logs_timestamp),
                    learning_rate=exp.iq_learning_rate,
                    experiment_name=exp.experiment_name,
                )
            else:
                print("Skipping IQ training")

            if exp.spec_epochs > 0:
                train_spec(
                    train_dataset_path=exp.train_dir,
                    val_dataset_path=exp.val_dir,
                    n_fft=exp.spec_n_fft,
                    time_dim=exp.spec_time_dim,
                    epochs=exp.spec_epochs,
                    batch_size=exp.spec_batch_size,
                    class_list=exp.class_list,
                    yolo_augment=exp.spec_yolo_augment,
                    skip_export=exp.spec_skip_export,
                    force_yolo_label_larger=exp.spec_force_yolo_label_larger,
                    output_dir=Path("experiment_logs", exp.experiment_name),
                    logs_dir=Path("spec_logs", logs_timestamp),
                )
            else:
                print("Skipping spectrogram training")

        except Exception as error:
            print(f"Error: {error}")
