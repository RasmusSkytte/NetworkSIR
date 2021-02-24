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

import urllib.request

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
    assert len(q_result) > 0
    cfg = [utils.DotDict(cfg) for cfg in q_result if cfg["network"]["ID"] == 0][0]
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


def get_cfgs(all_folders) :

    hashes = set()
    cfgs = []
    for folder in all_folders :
        cfg = folder_to_cfg(folder)

        # cfg must be loaded
        if cfg is not None :

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

        if self.subset is None:
            self.all_filenames = get_all_ABM_filenames(base_dir, filetype)
            self.all_folders   = get_all_ABM_folders(self.all_filenames)
            self.cfgs          = get_cfgs(self.all_folders)

        else :
            # Steps:
            # Connect to data base
            # Get the hashes for the relevent subset
            # Only load these

            db = utils.get_db_cfg()
            q = Query()
            query = q.version == 2.1
            for key, val in subset.items() :
                query = query & (q[key] == val)

            cfgs = db.search(query)

            self.all_filenames = []

            for hash_ in [cfg["hash"] for cfg in cfgs] :
                self.all_filenames.extend(utils.hash_to_filenames(hash_))

            self.all_folders   = get_all_ABM_folders(self.all_filenames)
            self.cfgs          = get_cfgs(self.all_folders)


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
import datetime


def load_SSI_url(SSI_data_url) :
    with ZipFile(BytesIO(urllib.request.urlopen(SSI_data_url).read())) as zfile :
        df = pd.read_csv(zfile.open("Municipality_cases_time_series.csv"), sep=";", index_col=0)
    return df


def download_newest_SSI_data(return_data=False, return_name=False) :

    url = "https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata"

    with urllib.request.urlopen(url) as response :
        html = str(response.read())

    s = re.search("Data-Epidemiologisk", html, re.IGNORECASE)
    data_url = html[s.start()-46:s.end()+23] + ".zip"

    df = load_SSI_url(data_url)

    name = datetime.datetime.strptime(html[s.end()+10:s.end()+18],'%d%m%Y').strftime('%Y_%m_%d')

    filename = 'Data/municipality_cases/' + name + '.csv'

    if not (os.path.dirname(filename) == '') and not os.path.exists(os.path.dirname(filename)) :
        os.makedirs(os.path.dirname(filename))

    df.to_csv(filename)

    if return_data :
        return df

    if return_name :
        return name


def load_kommune_data(df_coordinates, initial_distribution_file) :

    my_kommune = List(df_coordinates["kommune"].tolist())
    kommune_names = List(set(my_kommune))
    infected_per_kommune = np.zeros(len(kommune_names))
    immunized_per_kommune = np.zeros(len(kommune_names))


    if initial_distribution_file.lower() == "random" :
        infected_per_kommune  = np.ones(np.shape(infected_per_kommune))
        immunized_per_kommune = np.ones(np.shape(immunized_per_kommune))

    else :

        if initial_distribution_file.lower() == "newest" :
            df = download_newest_SSI_data(return_data=True)
        else :
            df = pd.read_csv('Data/municipality_cases/' + initial_distribution_file + ".csv")
        dates = df.index

        # First fill the immunized per kommune array
        arr = immunized_per_kommune

        for i, date in enumerate(dates) :
            infected_per_kommune_series = df.loc[date]

            # Last 7 days counts the currently infected
            if i == len(dates) - 7 :
                arr = infected_per_kommune

            for ith_kommune, kommune in enumerate(kommune_names) :
                if kommune == "Samsø" :
                    arr[ith_kommune] += 1
                elif kommune == "København" :
                    arr[ith_kommune] += infected_per_kommune_series["Copenhagen"]
                else :
                    arr[ith_kommune] += infected_per_kommune_series[kommune]

    return infected_per_kommune, immunized_per_kommune, kommune_names, my_kommune