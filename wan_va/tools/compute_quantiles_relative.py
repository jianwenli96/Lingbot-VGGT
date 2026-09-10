# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
import sys
import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

from pathlib import Path
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from functools import partial
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory, followlinks=True):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config,
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    with ThreadPool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)

    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()

    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

def get_relative_pose_6d(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()

    rot = R.from_euler("xyz", pose[:, 3:6])
    first_rot = R.from_euler(
        "xyz",
        np.tile(pose[:1, 3:6], (pose.shape[0], 1)),
    )

    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_euler = relative_rot.as_euler("xyz")

    relative_pose = np.concatenate(
        [relative_trans, relative_euler],
        axis=1,
    )

    return torch.from_numpy(relative_pose)

def get_left_gripper_data(grapper_effort):
    temp = grapper_effort.clone()
    positive_mask = grapper_effort > 0
    temp[positive_mask] = temp[positive_mask] / 7.0
    data = 0.5 - temp
    return data

def get_right_gripper_data(grapper_effort):
    temp = grapper_effort.clone()
    negative_mask = grapper_effort < 0
    temp[negative_mask] = temp[negative_mask] / 7.0
    data = temp + 0.5
    return data


class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config,
                                                           num_init_worker,
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]

        # Argument
        self.random_frame_cut = config.random_frame_cut
        self.min_frames = config.min_frames
        self.max_frames = config.max_frames

        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data

        return self._flatten_latent_dict(out)

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        elif self.config.env_type == 'tennis_tshape':
            action = get_relative_pose_6d(action[:, -6:]).numpy()
        else:
            action = action
        return torch.from_numpy(action).float()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)

        out_dict = dict()
        out_dict['actions'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        return out_dict

    def random_crop_n_dim(self, out_dict):
        _, n_total, *_ = out_dict["latents"].shape
        seq_length = torch.randint(self.min_frames, self.max_frames, (1,)).item()
        max_start = n_total - seq_length
        start = torch.randint(0, max_start + 1, (1,)).item()
        for key in ["latents", "actions", "actions_mask"]:
            # 3. 执行切片
            out_dict[key] = out_dict[key][:, start:start + seq_length, ...]
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['tennis_train'],
        num_init_worker=128
    )
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=False,
            num_workers=16,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        action_flat = rearrange(data['actions'], "b n c -> (b n) c")
        action_list.append(action_flat)
    action_all = torch.cat(action_list, dim=0)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    a_min = action_all.min(dim=0)[0]
    a_max = action_all.max(dim=0)[0]
    q01 = torch.quantile(action_all, 0.01, dim=0)
    q99 = torch.quantile(action_all, 0.99, dim=0)
    q00 = torch.min(action_all, dim=0)
    q100 = torch.max(action_all, dim=0)

    print("min", a_min - q01 / (q99-q01))
    print("max", a_max - q01 / (q99-q01))
    print("zero min", 0 * a_min - q01 / (q99-q01))
    print("one max", (1 + 0 * a_max) - q01 / (q99-q01))
    print("q01:", q01)
    print("q99:", q99)
    print("q00:", q00)
    print("q100:", q100)
