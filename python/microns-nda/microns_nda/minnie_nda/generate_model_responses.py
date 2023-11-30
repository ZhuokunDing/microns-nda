# run this script in dynamic vision environment: at-docker.ad.bcm.edu:5000/ewang/dv:cuda-11.3
import datajoint as dj
from scans import scan, oracle, scan_dataset
from nns import scan as nns_scan, model, architecture, transform, requests, train
from stimuli import stimulus, trial
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

def generate_responses():
    netflix = dj.create_virtual_module("netflix", "pipeline_netflix")
    minnie_function = dj.create_virtual_module(
        "microns_minnie_function", "microns_minnie_function"
    )

    scan_keys = (
        (
            minnie_function.ScanSet().Member()
            & (minnie_function.ScanSet() & 'name="HighQuality"')
        )
        .proj(session="scan_session")
        .fetch("KEY")
    )
    response_stimulus_rel = (
        scan.Response.Trial * trial.TrialStimulus & stimulus.StimulusConfig.Clip & scan_keys
    )

    stimulus_keys = (
        ((stimulus.StimulusConfig().Clip() & "cut_after=10") & netflix.UniqueSet)
        - response_stimulus_rel
    ).fetch("KEY", order_by="clip_number", limit=250)

    nn_rel = (
        nns_scan.NnConfig.TransferTrain()
        * architecture.TransferConfig().Scan3GroupCore()
        * architecture.ArchitectureConfig().Scan()
        * transform.TransformConfig().PlaneGrid()
        * (train.TrainConfig().Scan() * train.TrainerConfig.Cosine() & "max_epochs=100")
        * (scan_dataset.DatasetGroup() & "n_members=6").proj(
            core_dataset_group_hash="dataset_group_hash"
        )
    )  # 'nn_hash = "979ca9b3282f36acfb7f79892b2417ea"'

    model_keys = (
        nns_scan.ScanModel
        & nn_rel
        & scan_keys
        & (nns_scan.ScanConfig.Scan3 & scan_dataset.UnitConfig.All)
    ).fetch("KEY", order_by="animal_id, session, scan_idx")
    assert len(model_keys) == 13

    # generate responses
    all_response = []
    all_unit_df = []
    for model_key in tqdm(model_keys):
        scan_unit_df = pd.DataFrame(
            (nns_scan.Scan.Unit() & scan_keys & model_key)
            .proj(..., scan_session="session")
            .fetch(
                "animal_id",
                "scan_session",
                "scan_idx",
                "unit_id",
                "nn_response_index",
                order_by="unit_id",
                as_dict=True,
            )
        )
        # preprocessing info
        info = model.ModelInfo * nns_scan.ScanInfo & (
            nns_scan.ScanModelInstance & model_key
        )
        mid, fs, height, width, ft = info.fetch1(
            "monitor_id",
            "frame_step",
            "stimulus_height",
            "stimulus_width",
            "stimulus_filter_type",
        )

        # load network
        net = nns_scan.ScanModelInstance().load(
            device="cuda", training=False, key=model_key
        )
        scan_response = []
        for clip_key in tqdm(stimulus_keys):
            # load movie
            mov = stimulus.StimulusConfig().load(
                monitor_id=mid,
                frame_step=fs,
                height=height,
                width=width,
                filter_type=ft,
                key=clip_key,
            )
            # generate response (frames x neurons)
            with torch.no_grad():
                x = torch.tensor(mov[None, None], dtype=torch.float, device="cuda")
                scan_response.append(
                    net(x, log=False, standardized=False, centered=False)[0].cpu().numpy()
                )
        scan_response = np.concatenate(scan_response)
        scan_response = scan_response[:, scan_unit_df["nn_response_index"]]
        all_response.append(scan_response)
        del scan_unit_df["nn_response_index"]
        all_unit_df.append(scan_unit_df)
    all_response = np.concatenate(all_response, axis=1)
    all_unit_df = pd.concat(all_unit_df).reset_index(drop=True)

    all_response = all_response.T  # neuron x frames


    # compile all conditions
    all_condition_df = [
        (stimulus.StimulusCondition() & key).fetch1() for key in stimulus_keys
    ]

    all_condition_df = pd.DataFrame(all_condition_df)

    all_condition_df["col_idx_start"] = np.arange(0, 5000, 20)
    all_condition_df["col_idx_end"] = all_condition_df["col_idx_start"] + 20
    del all_condition_df["stimulus_hash"]

    # export stimuli
    movs = []
    info = model.ModelInfo * nns_scan.ScanInfo & (nns_scan.ScanModelInstance & model_keys)
    info_table = dj.U(
        "monitor_id",
        "frame_step",
        "stimulus_height",
        "stimulus_width",
        "stimulus_filter_type",
    )
    mid, fs, height, width, ft = (info_table & info).fetch1(
        "monitor_id",
        "frame_step",
        "stimulus_height",
        "stimulus_width",
        "stimulus_filter_type",
    )

    for clip_key in tqdm(stimulus_keys):
        movs.append(
            stimulus.StimulusConfig().load(
                monitor_id=mid,
                frame_step=fs,
                height=height,
                width=width,
                filter_type=ft,
                key=clip_key,
            )
        )
    movs = np.array(movs)
    return all_response, all_unit_df, all_condition_df, movs


def save_response_to_file_table(
    response: np.ndarray,
    stimuili: np.ndarray,
    unit_df,
    condition_df,
    dynamic_model_scan_set_hash,
    note,
):
    # insert into minnie_function.RespArrNnsV10
    minnie_function = dj.create_virtual_module(
        "minnie_function", "microns_minnie_function"
    )
    resp_array_store = {
        "resp_array_file": {
            "location": "/mnt/dj-stor01/microns/minnie65/nda/function/resp_array_file",
            "protocol": "file",
            "stage": "/mnt/dj-stor01/microns/minnie65/nda/function/resp_array_file",
            "subfolding": (2, 2),
        }
    }
    dj.config["stores"].update(**resp_array_store)
    dj.config["enable_python_native_blobs"] = True
    if "row_idx" not in unit_df:
        unit_df["row_idx"] = unit_df.index
    if len(minnie_function.RespArrNnsV10File()) > 0:
        resp_array_idx = (
            minnie_function.RespArrNnsV10File().fetch(
                "resp_array_idx", order_by="resp_array_idx DESC"
            )[0]
            + 1
        )
    else:
        resp_array_idx = 0
    assert response.shape[0] == len(unit_df) == unit_df.row_idx.max() + 1
    assert unit_df.row_idx.is_unique and (unit_df.row_idx >= 0).all()
    fp = (
        "/mnt/dj-stor01/microns/minnie65/nda/function/resp_array_file/"
        + str(resp_array_idx)
        + ".npy"
    )
    stim_fp = (
        "/mnt/dj-stor01/microns/minnie65/nda/function/resp_array_file/"
        + str(resp_array_idx)
        + "_stim.npy"
    )
    with open(fp, "wb+"):
        np.save(fp, response)
    with open(stim_fp, "wb+"):
        np.save(stim_fp, stimuili)
    master_key = dict(
        resp_array_idx=resp_array_idx,
        dynamic_model_scan_set_hash=dynamic_model_scan_set_hash,
        resp_array=fp,  # neurons x bins
        description=note,
    )
    minnie_function.RespArrNnsV10File().insert1(master_key)
    unit_df["resp_array_idx"] = resp_array_idx
    minnie_function.RespArrNnsV10File.Unit.insert(unit_df, ignore_extra_fields=True)
    condition_df["resp_array_idx"] = resp_array_idx
    minnie_function.RespArrNnsV10File.Condition.insert(
        condition_df, ignore_extra_fields=True
    )
    minnie_function.RespArrNnsV10File.Stimuli.insert(
        {**master_key, "stimulus_array": stim_fp}, ignore_extra_fields=True
    )

def save_response_to_file_path(
    file_path: Path,
    response: np.ndarray,
    stimuili: np.ndarray,
    unit_df,
    condition_df,
):
    file_path = Path(file_path)
    assert file_path.is_dir()
    if "row_idx" not in unit_df:
        unit_df["row_idx"] = unit_df.index
    assert response.shape[0] == len(unit_df) == unit_df.row_idx.max() + 1
    assert unit_df.row_idx.is_unique and (unit_df.row_idx >= 0).all()
    resp_fp = file_path / "resp_array.npy"
    stim_fp = file_path / "stimuli.npy"
    unit_df_fp = file_path / "unit_df.csv"
    condition_df_fp = file_path / "condition_df.csv"
    with open(resp_fp, "wb+"):
        np.save(resp_fp, response)
    with open(stim_fp, "wb+"):
        np.save(stim_fp, stimuili)
    unit_df.to_csv(unit_df_fp)
    condition_df.to_csv(condition_df_fp)
        

if __name__ == "__main__":

    all_response, all_unit_df, all_condition_df, movs = generate_responses()

    save_response_to_file_table(
        response=all_response,
        unit_df=all_unit_df,
        condition_df=all_condition_df,
        stimuili=movs,
        dynamic_model_scan_set_hash="cf02215b75981bcf95e8339c97f9d2f4",
        note=(
            "Model responses to 250 10 sec clips without behavior modulation from the unique set in netflix schema. "
            "Model responses are generated with (log=False, standardized=False, centered=False). "
            "Burnin period is 10 frames per clip."
        ),
    )

    save_response_to_file_path(
        file_path='/mnt/at-export01/Minnie_Export_APL/function/',
        response=all_response,
        unit_df=all_unit_df,
        condition_df=all_condition_df,
        stimuili=movs,
    )
