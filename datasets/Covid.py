import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class Covid(DatasetBase):

    dataset_dir = "Covid"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(
            self.dataset_dir, "split_fewshot")

       
        train, val, test = read_split(self.dataset_dir)
       

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(
                    f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(
                    train, num_shots=num_shots,split="train")
                val = self.generate_fewshot_dataset(
                    val, num_shots=min(num_shots, 4),split="val")
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)


def read_split(dir):
        def _convert(split):
            if split == "train":
                file_path=os.path.join(dir,"train")
            elif split == "val":
                file_path=os.path.join(dir,"val")
            elif split == "test":
                file_path = os.path.join(dir, "test")
            
           
            df_list = []
            for name in ["Covid.CSV","Normal.CSV"]:
                filename=os.path.join(file_path,name)
                print('load data from', filename)
                df = pd.read_csv(filename, index_col=0)
                df_list.append(df)
            df_list = pd.concat(df_list, axis=0).reset_index(drop=True)

            out = []
            for index in range(df_list.shape[0]) :
                row = df_list.iloc[index]
                impath=row.imgpath
                label=row.label
                classname=row.classname
                setpath = os.path.join(file_path, classname)
                impath = os.path.join(setpath,impath)
                item = Datum(impath=impath, label=int(
                    label), classname=classname)
                out.append(item)
            return out

        
        train = _convert("train")
        val = _convert("val")
        test = _convert("test")

        return train, val, test

