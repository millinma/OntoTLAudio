from functools import cached_property
from itertools import chain
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import librosa
from omegaconf import DictConfig
import pandas as pd

from autrainer.transforms import SmartCompose

from autrainer.datasets.abstract_dataset import BaseMLClassificationDataset
from autrainer.datasets.utils import ZipDownloadManager


balanced_train_file = "balanced_train_segments.csv"
unbalanced_train_file = "unbalanced_train_segments.csv"
eval_file = "eval_segments.csv"

ontology_file = "ontology.json"

# META_FILES = [balanced_train_file, eval_file]
META_FILES = [balanced_train_file, eval_file, unbalanced_train_file]

# only relevant for downloading files from AudioSet Website and YT
DOWNLOAD_FILES = {
    "unbalanced_train_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv",
    "balanced_train_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "eval_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
}

LABEL_FILE = "label_list.txt"


class AudioSet(BaseMLClassificationDataset):
    def __init__(
        self,
        path: str,
        features_subdir: str,
        seed: int,
        metrics: List[Union[str, DictConfig, Dict]],
        tracking_metric: Union[str, DictConfig, Dict],
        index_column: str,
        file_type: str,
        file_handler: Union[str, DictConfig, Dict],
        batch_size: int,
        target_column: List[str] = [],
        inference_batch_size: Optional[int] = None,
        features_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        dev_split: float = 0.0,  # proportion of train
        balanced_version=True,
        label_categories: Optional[
            List[str]
        ] = None,  # considered labels from audioset ontology
        train_set_size: float = 1.0,  # proportion of total train data
        test_set_size: float = 1.0,  # proportion of total test data
        dev_split_seed: Optional[int] = None,
        stratify: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        """AudioSet dataset.

        Args:
            path: Root path to the dataset.
            features_subdir: Subdirectory containing the features.
            seed: Seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track.
            index_column: Index column of the dataframe.
            target_column: Target column of the dataframe.
            file_type: File type of the features.
            file_handler: File handler to load the data.
            batch_size: Batch size.
            inference_batch_size: Inference batch size. If None, defaults to
                batch_size. Defaults to None.
            features_path: Root path to features. Useful
                when features need to be extracted and stored
                in a different folder than the root of the dataset.
                If `None`, will be set to `path`. Defaults to `None`.
            csv_path: Root path to csv files for train, validation and test.
                Useful when those files could not be created  in
                the root of the dataset.
                If `None`, will be set to `path`. Defaults to `None`.
            train_transform: Transform to apply to the training set.
                Defaults to None.
            dev_transform: Transform to apply to the development set.
                Defaults to None.
            test_transform: Transform to apply to the test set.
                Defaults to None.
            stratify: Columns to stratify the dataset on. Defaults to None.
            dev_split: Fraction of the training set to use as the development
                set. Defaults to 0.0.
            dev_split_seed: Seed for the development split. If None, seed is
                used. Defaults to None.
        """
        # self._assert_dev_split(dev_split)
        self.dev_split = dev_split
        self.dev_split_seed = dev_split_seed or seed
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.balanced_version = balanced_version
        self.label_categories = label_categories
        self.index_column = index_column
        if csv_path is None:
            csv_path = path
        self.csv_path = csv_path
        print(self.csv_path)
        if target_column == []:
            with open(os.path.join(self.csv_path, LABEL_FILE), "r") as file:
                target_column = [line.strip() for line in file]
        if self.label_categories is not None:
            label_list = []
            with open(os.path.join(path, ontology_file), "r") as file:
                ontology_data = json.load(file)
            name_to_id = {
                entry["name"]: entry["id"] for entry in ontology_data
            }
            id_to_child_nodes = {
                entry["id"]: entry["child_ids"] for entry in ontology_data
            }

            def get_child_nodes(node_id):
                node_list = [node_id]
                for child_id in id_to_child_nodes[node_id]:
                    node_list += get_child_nodes(child_id)
                return node_list

            for label_category in self.label_categories:
                label_list += get_child_nodes(name_to_id[label_category])
            # remove all entries that do not appear in the data
            label_list = [
                label for label in label_list if label in target_column
            ]
            label_list = list(set(label_list))
            label_list.sort()
            target_column = label_list
            self.target_column = target_column

        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            features_path=features_path,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
            threshold=threshold,
        )

    @property
    def audio_subdir(self) -> str:
        """Subfolder containing audio data.
        Defaults to `default` for our standard format.
        Should be overridden for datasets
        that do not conform to it.
        """
        return ""

    @cached_property
    def df_train(self) -> pd.DataFrame:
        return self._load_df[0]

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        return self._load_df[1]

    @cached_property
    def df_test(self) -> pd.DataFrame:
        return self._load_df[2]

    @cached_property
    def _load_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.balanced_version:
            train_df = pd.read_csv(
                os.path.join(self.csv_path, "balanced_train_autrainer.csv")
            )
        else:
            train_df = pd.read_csv(
                os.path.join(self.csv_path, "unbalanced_train_autrainer.csv")
            )
        test_df = pd.read_csv(
            os.path.join(self.csv_path, "eval_autrainer.csv")
        )
        # reduced the dfs to only include specified label columns and lines with non-zero entries therein
        if self.label_categories is not None:
            train_df = train_df[[self.index_column] + self.target_column]
            test_df = test_df[[self.index_column] + self.target_column]
            train_df = train_df.loc[
                (train_df[self.target_column] != 0).any(axis=1)
            ]
            test_df = test_df.loc[
                (test_df[self.target_column] != 0).any(axis=1)
            ]

        if self.train_set_size < 1.0:
            train_df = train_df.sample(
                frac=self.train_set_size, random_state=self.dev_split_seed
            )

        if self.test_set_size < 1.0:
            test_df = test_df.sample(
                frac=self.test_set_size, random_state=self.dev_split_seed
            )

        if self.dev_split > 0.0:
            dev_df = train_df.sample(
                frac=self.dev_split, random_state=self.dev_split_seed
            )
            train_df = train_df.drop(dev_df.index)
        else:
            dev_df = test_df.copy()
        print(train_df.head(3))
        # self.train_df = train_df
        # self.dev_df = dev_df
        # self.test_df = test_df

        return train_df, dev_df, test_df

    @staticmethod
    def download(path: str) -> None:  # pragma: no cover
        """Download the AudioSet dataset.

        For more information on the dataset, see:
        https://research.google.com/audioset/download.html

        Args:
            path: Path to the directory to download the dataset to.
        """
        # )
        # download method is very limited as of now, as it is very
        # slow to download the data and extract the audios.
        # It would probably need some parallelization option.
        if os.path.isdir(path):
            return
        os.makedirs(path, exist_ok=True)

        # # download and extract files
        dl_manager = ZipDownloadManager(DOWNLOAD_FILES, path)
        dl_manager.download(check_exist=["eval_segments.csv"])

        import ffmpeg
        import yt_dlp

        failed_downloads = 0
        success_downloads = 0

        for metadata_file in DOWNLOAD_FILES.keys():
            metadata = pd.read_csv(
                os.path.join(path, "unbalanced_train_segments.csv"),
                skiprows=2,
                delimiter=", ",
            )
            print(metadata.shape)

            temp_path = os.path.join(path, "temp")
            os.makedirs(temp_path, exist_ok=True)

            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": os.path.join(temp_path, "%(id)s.%(ext)s"),
            }

            for index, row in metadata.iterrows():
                ytid = row["# YTID"]
                start_time = row["start_seconds"]
                end_time = row["end_seconds"]
                labels = row["positive_labels"]

                input_file = os.path.join(temp_path, f"{ytid}.mp3")
                output_file = os.path.join(
                    path, f"{ytid}_{start_time}_{end_time}.wav"
                )

                if os.path.exists(output_file):
                    print("Skipping", output_file)
                    continue
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        ydl.download(
                            [f"https://www.youtube.com/watch?v={ytid}"]
                        )
                    except Exception:
                        # print("Failed to download video", ytid)
                        failed_downloads += 1

                if os.path.exists(input_file):
                    try:
                        ffmpeg.input(
                            input_file, ss=start_time, to=end_time
                        ).output(output_file).run(overwrite_output=True)
                        os.remove(input_file)
                        success_downloads += 1
                    except Exception:
                        failed_downloads += 1
                else:
                    failed_downloads += 1
        print("Successful downloads", success_downloads)
        print("Failed downloads", failed_downloads)
        for meta_file in META_FILES:
                out_file_path = os.path.join(
                    path, meta_file.replace("segments", "autrainer")
                )
                split_dir = meta_file.replace(".csv", "")
                meta_df = pd.read_csv(
                    os.path.join(path, meta_file),
                    skiprows=2,
                    delimiter=", ",
                )

                print("Creating", meta_file.replace("segments", "autrainer"))
                if os.path.exists(out_file_path):
                    continue
                if len(unique_labels) == 0:
                    all_labels = []
                    for index, row in meta_df.iterrows():
                        labels = row["positive_labels"]
                        labels = labels.replace('"', "").split(",")
                        all_labels.append(labels)
                    unique_labels = list(set(chain.from_iterable(all_labels)))
                    unique_labels.sort()
                    with open(os.path.join(path, LABEL_FILE), "w") as f:
                        for label in unique_labels:
                            f.write(label + "\n")
                # remove the "positive_labels" column
                meta_df = meta_df[meta_df.columns[:-1]]
                meta_df["filename"] = (
                    split_dir + "/" + meta_df["# YTID"] + ".wav"
                )
                new_columns_df = pd.DataFrame(
                    0.0, index=meta_df.index, columns=unique_labels
                )

                # Overwrite labels
                for idx, labels in tqdm(enumerate(all_labels)):
                    new_columns_df.loc[idx, labels] = 1.0
                meta_df = pd.concat([meta_df, new_columns_df], axis=1)

                def file_exists(filename):
                    # file_path = os.path.join(path, out_path, filename)
                    file_path = os.path.join(path, filename)
                    file_exist = os.path.isfile(file_path)
                    if file_exist:
                        # Checks if files are loadable and of more than 0 length (which was the case in 2 eval files)
                        try:
                            sfile = librosa.load(file_path)
                            if len(sfile[0]) == 0:
                                print("File", filename, "is broken")
                                return False
                        except Exception:
                            print("File", filename, "is broken")
                            return False
                    return file_exist

                # removes all non-existing and broken files from the meta_file
                meta_df = meta_df[
                    meta_df["filename"].progress_apply(file_exists)
                ]
                # meta_df = meta_df[meta_df["filename"].parallel_apply(file_exists)]

                meta_df.to_csv(out_file_path, index=False)