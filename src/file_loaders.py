import numpy as np
import pandas as pd
from pathlib import Path
import re
import h5py
import os
from tinydb import Query

# from tqdm import tqdm TODO : delete line
from src.utils import utils
from numba.typed import List, Dict  #TODO : delete Dict from line


def pandas_load_file(filename) :
    # df_raw = pd.read_csv(file)  # .convert_dtypes()

    with h5py.File(filename, "r") as f :
        df_raw = pd.DataFrame(f["df"][()])

    df = df_raw.copy()

    for state in ["E", "I"] :
        cols = [col for col in df_raw.columns if state in col and len(col) == 2]
        df[state] = sum((df_raw[col] for col in cols))
        df = df.drop(columns=cols)

    # only keep relevant columns
    df.rename(columns={"Time" : "time"}, inplace=True)

    # remove duplicate timings
    df = df.loc[df["time"].drop_duplicates().index]

    return df


def path(file) :
    if isinstance(file, str) :
        file = Path(file)
    return file


def file_is_empty(file) :
    return path(file).stat().st_size == 0


def get_all_ABM_filenames(base_dir="Output/ABM", filetype="hdf5") :
    "get all ABM result files with filetype {filetype}"
    files = path(base_dir).rglob(f"*.{filetype}")
    # files = sorted(files, )
    return sorted(
        [str(file) for file in files if not file_is_empty(file)],
        key=os.path.getmtime,
    )

def load_Network_file( filename) :
    with h5py.File(filename, "r") as f :
        print(list(f.keys()))
        print(filename, f)
        day_found_infected = pd.DataFrame(f["day_found_infected"][()])
        R_true = pd.DataFrame(f["R_true"][()])
        R_true_brit = pd.DataFrame(f["R_true_brit"][()])
        freedom_impact = pd.DataFrame(f["freedom_impact"][()])
        my_state = pd.DataFrame(f["my_state"][()])

    return day_found_infected, R_true, freedom_impact, R_true_brit, my_state

def get_all_ABM_folders(filenames) :
    folders = list()
    for filename in filenames :
        folder = str(path(filename).parent)
        if not folder in folders :
            folders.append(folder)
    return folders


def filename_to_hash(filename) :
    filename = str(filename)
    # split at "_" and "."
    return re.split("_|\.", filename)[-5]


def filename_to_cfg(filename) :
    return hash_to_cfg(filename_to_hash(filename))


def folder_to_hash(folder) :
    folder = str(folder)
    # split at "_" and "."
    return folder.split(os.sep)[-1]


def folder_to_cfg(folder) :
    hash_ = folder_to_hash(folder)
    cfgs_dir = os.path.join(*folder.split(os.sep)[:-2])
    return hash_to_cfg(hash_, cfgs_dir=cfgs_dir)


def hash_to_cfg(hash_, cfgs_dir="./Output/cfgs") :
    db_cfg = utils.get_db_cfg()
    q = Query()
    q_result = db_cfg.search(q.hash == hash_)
    if len(q_result) == 0 :
        cfgs = [str(file) for file in Path(cfgs_dir).rglob(f"*{hash_}.yaml")]
        if len(cfgs) == 1 :
            cfg = utils.load_yaml(cfgs[0])
            cfg.hash = utils.cfg_to_hash(cfg)
            return cfg
        else :
            return None
    assert len(q_result) == 1
    cfg = utils.DotDict(q_result[0])
    return cfg

def query_to_hashes(subset=None, base_dir="Output") :
    db_cfg = utils.get_db_cfg(path=os.path.join(base_dir, "db.json"))
    q = Query()

    if subset is None :
        q_result = db_cfg.all()
    else :
        q_result = db_cfg.search(utils.dict_to_query(subset))

    return q_result
    # if len(q_result) == 0 :
    #     cfgs = [str(file) for file in Path(cfgs_dir).rglob(f"*{hash_}.yaml")]
    #     if len(cfgs) == 1 :
    #         cfg = utils.load_yaml(cfgs[0])
    #         cfg.hash = utils.cfg_to_hash(cfg)
    #         return cfg
    #     else :
    #         return None
    # assert len(q_result) == 1
    # cfg = utils.DotDict(q_result[0])
    # return cfg


def get_cfgs(all_folders, subset=None) :

    hashes = set()
    cfgs = []
    for folder in all_folders :
        cfg = folder_to_cfg(folder)

        # cfg must be loaded
        if cfg is not None :

            cfg_flat = utils.flatten_cfg(cfg.deepcopy())

            # cfg must match the subset
            if subset is not None and not subset.items() <= cfg_flat.items() :
                continue

            # cfg hash should not allredy be loaded
            if not cfg.hash in hashes :
                cfgs.append(cfg)
            hashes.add(cfg.hash)

    return cfgs


class ABM_simulations :
    def __init__(self, base_dir="Output/ABM", filetype="hdf5", subset=None, verbose=False) :
        self.base_dir = utils.path(base_dir)
        self.filetype = filetype
        self.subset = subset
        self.verbose = verbose
        if verbose :
            print("Loading ABM_simulations \n", flush=True)
        self.all_filenames = get_all_ABM_filenames(base_dir, filetype)
        self.all_folders = get_all_ABM_folders(self.all_filenames)

        # Steps:
        # Connect to data base
        # Get the hashes for the relevent subset
        # Only load these

        self.cfgs = get_cfgs(self.all_folders, subset=self.subset)  # TODO: This function is slow
        self.d = self._convert_all_files_to_dict(filetype)

    def _convert_all_files_to_dict(self, filetype) :
        """
        Dictionary containing all files related to a given hash :
        d[hash] = list of filenames
        """
        d = {}
        for cfg in self.cfgs :
            d[cfg.hash] = utils.hash_to_filenames(cfg.hash, self.base_dir, self.filetype)
        return d

    def iter_all_files(self) :
        for filename in self.all_filenames :
            yield filename

    def iter_folders(self) :
        for cfg in self.cfgs :
            filenames = self.d[cfg.hash]
            yield cfg, filenames

    def iter_cfgs(self) :
        for cfg in self.cfgs :
            yield cfg

    def cfg_to_filenames(self, cfg) :

        cfg = utils.DotDict(cfg)
        cfg_list = utils.query_cfg(cfg)

        if not len(cfg_list) == 1 :
            raise AssertionError(
                f"cfg did not give unique results in the database",
                "cfg :",
                cfg,
                "cfg_list :",
                cfg_list,
            )
        try :
            return self.d[cfg_list[0].hash]
        except KeyError :
            return None

    # def __getitem__(self, key) :
    #     if isinstance(key, int) :
    #         return self.all_files[key]
    #     # elif isinstance(key, int) :
    #     #     return self.all_files[key]

    def __len__(self) :
        return len(self.all_filenames)

    def __repr__(self) :
        return (
            f"ABM_simulations(base_dir='{self.base_dir}', filetype='{self.filetype}').\n"
            + f"Contains {len(self)} files with "
            + f"{len(self.cfgs)} different simulation parameters."
        )


#%%


from io import BytesIO
from zipfile import ZipFile
import urllib.request
from urllib.error import HTTPError
import datetime


def load_SSI_url(SSI_data_url) :
    with ZipFile(BytesIO(urllib.request.urlopen(SSI_data_url).read())) as zfile :
        df = pd.read_csv(zfile.open("Municipality_cases_time_series.csv"), sep=";")
    return df


def load_newest_SSI_data(max_days_back=30) :
    # SSI_data_url = "https ://files.ssi.dk/Data-Epidemiologiske-Rapport-22102020-20mg"
    today = datetime.date.today()
    for i in range(max_days_back) :
        day = today - datetime.timedelta(days=i)
        s_day = day.strftime("%d%m%Y")
        SSI_data_url = f"https ://files.ssi.dk/Data-Epidemiologiske-Rapport-{s_day}-20mg"
        try :
            df = load_SSI_url(SSI_data_url)
            return df
        except HTTPError :
            continue
    raise AssertionError("Could not find any data from SSI")


def load_kommune_data(df_coordinates) :
    my_kommune = List(df_coordinates["kommune"].tolist())
    df = load_newest_SSI_data().set_index("date_sample")
    dates = df.index[-8 :]
    kommune_names = List(set(my_kommune))
    infected_per_kommune_ints = np.zeros(len(kommune_names))
    for date in dates :
        infected_per_kommune_series = df.loc[date]

        for ith_kommune, kommune in enumerate(kommune_names) :
            if kommune == "Samsø" :
                infected_per_kommune_ints[ith_kommune] += 1
            elif kommune == "København" :
                infected_per_kommune_ints[ith_kommune] += infected_per_kommune_series["Copenhagen"]
            else :
                infected_per_kommune_ints[ith_kommune] += infected_per_kommune_series[kommune]
    return infected_per_kommune_ints, kommune_names, my_kommune