import glob

from pathlib import Path
from tqdm import tqdm

import rfml.annotation_utils as annotation_utils
import rfml.data as data_class

import time


# samples_dir = Path("/home/ltindall/iqt/gamutrf-deploy/docker_rundir/samples/")
samples_dir = Path("/data/s3_gamutrf/postprocess/")
samples_glob = f"{str(samples_dir)}/**/*.sigmf-meta"
files_to_process = []
processed_files = []
while True: 
    files_to_process.extend(glob.glob(samples_glob, recursive=True))
    files_to_process = list(set(files_to_process)-set(processed_files))
    print(f"{files_to_process=}")
    if len(files_to_process) > 0: 
        
        # for _ in range(len(files_to_process)):
        for f in tqdm(files_to_process):
            # data_obj = data_class.Data(files_to_process.pop(0))
            data_obj = data_class.Data(f)


            annotation_utils.annotate(
                data_obj,
                avg_window_len=256,
                debug_duration=0.25,
                # debug=True,
                dry_run=False,
                bandwidth_estimation=0.99,
                force_threshold_db=None,
                overwrite=True,
                max_annotations=None,
                dc_block=None,
                # verbose=True,
                # time_start_stop=(0,0.05),
                power_estimate_duration=0.5,  # only process n seconds of I/Q samples at a time
                # labels={
                #     "f":{},
                # },
                labels={
                    "?mini2_video": {
                        # "bandwidth_limits": (16e6, None),
                        "annotation_length": (2048, None),
                        # "annotation_seconds": (0.001, None),
                        # "set_bandwidth": (-8.5e6, 9.5e6)
                    },
                    # "mini2_telem": {
                    #     "bandwidth_limits": (None, 16e6),
                    #     "annotation_length": (10000, None),
                    #     "annotation_seconds": (None, 0.001),
                    # }
                    "?environment": {
                        "annotation_length": (2048, None),
                    },
                },
                
                n_components=None,
                n_init=1,
                fft_len=256,
                # model_file="/home/ltindall/iqt/rfml/weights/apartment_experiment_torchscript.pt",
                model_file="/home/ltindall/iqt/rfml/weights/apartment_experiment_torchserve.pt",
                index_to_name_file="/home/ltindall/iqt/rfml/experiment_logs/apartment_experiment/iq_logs/09_19_2024_02_38_55/index_to_name.json"
            )
            processed_files.append(f)
        files_to_process = []
                
    else:
        print("sleep")
        time.sleep(1)


data_globs = ["/data/s3_gamutrf/gamutrf-lucas-collect/train/mini2/*.zst"]

for file_glob in data_globs:
    for f in tqdm(glob.glob(str(Path(file_glob)))):
        data_obj = data_class.Data(f)
        annotation_utils.reset_annotations(data_obj)
        annotation_utils.annotate(
            data_obj,
            avg_window_len=256,
            debug=False,
            bandwidth_estimation=0.99,  # True,
            overwrite=False,
            # power_estimate_duration = 0.1,
            # n_components=3,
            # n_init=2,
            # dry_run=True,
            # time_start_stop=(1,None),
            labels={
                "?mini2_video": {
                    # "bandwidth_limits": (16e6, None),
                    "annotation_length": (10000, None),
                    "annotation_seconds": (0.001, None),
                    # "set_bandwidth": (-8.5e6, 9.5e6)
                },
                # "mini2_telem": {
                #     "bandwidth_limits": (None, 16e6),
                #     "annotation_length": (10000, None),
                #     "annotation_seconds": (None, 0.001),
                # }
                "?environment": {
                    "annotation_length": (2048, None),
                },
            },
        )

        # option 1) load one file as dataset and run inference on file 

# option 2) load all files as dataset and run inference


# data_globs = ["/data/s3_gamutrf/gamutrf-lucas-collect/train/environment/*.zst"]


# for file_glob in data_globs:
#     for f in tqdm(glob.glob(str(Path(file_glob)))):
#         data_obj = data_class.Data(f)
#         annotation_utils.reset_annotations(data_obj)
#         annotation_utils.annotate(
#             data_obj,
#             avg_window_len=1024,
#             debug=False,
#             bandwidth_estimation=0.99,  # True,
#             overwrite=False,
#             # power_estimate_duration = 0.1,
#             # n_components=3,
#             # n_init=2,
#             # dry_run=True,
#             labels={
#                 "environment": {
#                     "annotation_length": (2048, None),
#                 },
#             },
#         )
