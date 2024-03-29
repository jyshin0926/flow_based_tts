import argparse
import glob
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/neutral_aihub_genesis_base_detdp.json",
        # default="./configs/neutral_aihub_base_detdp.json",
        # default="/home/jaeyoung/workspace/genesis-speech/tests/test_lib/vits/configs/genesis_base.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/data/jaeyoung/speech_synthesis/VITS/model/230915_aihub_genesis_hier_vits_detdp_rmsnorm_neutral",
        # default="/data/jaeyoung/speech_synthesis/VITS/model/230902_aihub_hier_vits_sdp",
        # default="/data/jaeyoung/speech_synthesis/VITS/model/230915_aihub_hier_vits_detdp_rmsnorm_neutral",
        help="Model name",
    )
    # parser.add_argument('-m', '--model',type=str, default="/data/jaeyoung/speech_synthesis/VITS/model/1108_1938_finetune",
    #                     help='Model name')
    # parser.add_argument('-m', '--model',type=str, default="/data/jaeyoung/speech_synthesis/VITS/model/finetune_1103_1645",
    #                     help='Model name')
    # parser.add_argument('-m', '--model', type=str, default="/data/jaeyoung/speech_synthesis/VITS/model/1021_1136",
    #                     help='Model name')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

def load_checkpoint(checkpoint_path, model, optimizer=None):
    hps = get_hparams()

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    saved_state_dict = checkpoint_dict["model"]
    ignore_layers_warmstart = []
    if len(ignore_layers_warmstart):
        pretrained_dict = {
            k: v
            for k, v in saved_state_dict.items()
            if all(l not in k for l in ignore_layers_warmstart)
        }
    else:
        pretrained_dict = saved_state_dict

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # new_state_dict[k] = saved_state_dict[k]
            new_state_dict[k] = pretrained_dict[k]

        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = sf.read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# # TODO:: 현재 training_files 만 받는데 이걸 dictionary 형태로 받기
def load_filepaths_and_text(datasets: HParams):
    dset = []
    # for dset_name, dset_dict in datasets.items():
    data_paths = datasets["datadir"]
    speaker_name = datasets["speaker"]
    speaker_id = create_speaker_lookup_table(speaker_name)

    for i, path in enumerate(data_paths):
        folder_path = path
        audiodir = datasets["audiodir"][i]
        filename = datasets["filelist"][i]

        wav_folder_prefix = os.path.join(folder_path, audiodir)
        metadata = pd.read_csv(filename)
        metadata = metadata[metadata["speaker_name"].isin(list(speaker_id.keys()))]
        metadata["path"] = wav_folder_prefix + "/" + metadata["speaker_name"] + "/" + metadata["basename"] + ".flac"
        metadata["text"] = metadata["text"].apply(lambda text: text.strip("{}"))
        metadata["speaker_id"] = metadata["speaker_name"].apply(lambda name: speaker_id[name])
        dset.extend(metadata[["path", "speaker_id", "text"]].values.tolist())

    return dset


def create_speaker_lookup_table(data):
    static_speaker_ids = ["정윤성","임지윤","이승은","이영복"]
    other_speaker_ids = list(np.sort(np.unique([x for x in data if x not in static_speaker_ids]))[::-1])
    speaker_ids = static_speaker_ids + other_speaker_ids
    # speaker_ids = other_speaker_ids

    d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
    print("Number of speakers:", len(d))
    print("Speaker IDS", d)
    return d
