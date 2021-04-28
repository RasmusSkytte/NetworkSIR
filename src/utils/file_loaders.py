import numpy as np
import numba as nb
import pandas as pd
from pathlib import Path
import re
import h5py
import os
import yaml
from tinydb import Query

from src.utils import utils
from src.simulation import nb_jitclass

from numba.typed import List


from io import BytesIO
from zipfile import ZipFile
import urllib.request
import datetime

import geopandas as gpd



 ######  #### ##     ## ##     ## ##          ###    ######## ####  #######  ##    ##  ######
##    ##  ##  ###   ### ##     ## ##         ## ##      ##     ##  ##     ## ###   ## ##    ##
##        ##  #### #### ##     ## ##        ##   ##     ##     ##  ##     ## ####  ## ##
 ######   ##  ## ### ## ##     ## ##       ##     ##    ##     ##  ##     ## ## ## ##  ######
      ##  ##  ##     ## ##     ## ##       #########    ##     ##  ##     ## ##  ####       ##
##    ##  ##  ##     ## ##     ## ##       ##     ##    ##     ##  ##     ## ##   ### ##    ##
 ######  #### ##     ##  #######  ######## ##     ##    ##    ####  #######  ##    ##  ######

def get_all_filenames(base_dir="Output/ABM", filetype="hdf5") :
    "get all result files with filetype {filetype}"
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
            cfg = load_yaml(cfgs[0])
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
    def __init__(self, base_dir='Output', filetype='hdf5', subset=None, verbose=False) :
        self.base_dir = utils.path(base_dir)
        self.filetype = filetype
        self.subset = subset
        self.verbose = verbose

        if verbose :
            print('Loading ABM_simulations \n', flush=True)

        if self.subset is None:
            self.filenames   = get_all_filenames(os.path.join(base_dir, 'ABM'), filetype)
            self.networks    = get_all_filenames(os.path.join(base_dir, 'network'), filetype)
            self.all_folders = get_all_ABM_folders(self.all_filenames)
            self.cfgs        = get_cfgs(self.all_folders)

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

            self.filenames = []
            self.networks = []

            for hash_ in [cfg["hash"] for cfg in cfgs] :
                self.filenames.extend(utils.hash_to_filenames(hash_, base_dir=os.path.join(base_dir, 'ABM')))
                self.networks.extend( utils.hash_to_filenames(hash_, base_dir=os.path.join(base_dir, 'network')))

            self.all_folders   = get_all_ABM_folders(self.filenames)
            self.cfgs          = get_cfgs(self.all_folders)


        self.d = self._convert_all_files_to_dict(filetype)

    def _convert_all_files_to_dict(self, filetype) :
        """
        Dictionary containing all files related to a given hash :
        d[hash] = list of filenames
        """
        d = {}
        for cfg in self.cfgs :
            d[cfg.hash] = utils.hash_to_filenames(cfg.hash, os.path.join(self.base_dir, 'ABM'), self.filetype)
        return d

    def iter_files(self) :
        for filename in self.filenames :
            yield filename

    def iter_network_files(self) :
        for filename in self.networks :
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
        return len(self.filenames)

    def __repr__(self) :
        return (
            f"ABM_simulations(base_dir='{self.base_dir}', filetype='{self.filetype}').\n"
            + f"Contains {len(self)} files with "
            + f"{len(self.cfgs)} different simulation parameters."
        )








########   #######  ########  ##     ## ##          ###    ######## ####  #######  ##    ##
##     ## ##     ## ##     ## ##     ## ##         ## ##      ##     ##  ##     ## ###   ##
##     ## ##     ## ##     ## ##     ## ##        ##   ##     ##     ##  ##     ## ####  ##
########  ##     ## ########  ##     ## ##       ##     ##    ##     ##  ##     ## ## ## ##
##        ##     ## ##        ##     ## ##       #########    ##     ##  ##     ## ##  ####
##        ##     ## ##        ##     ## ##       ##     ##    ##     ##  ##     ## ##   ###
##         #######  ##         #######  ######## ##     ##    ##    ####  #######  ##    ##


def parse_age_distribution_data(filename, label_map=None) :

    age_dist_raw      = pd.read_csv(filename, index_col=0)
    kommune_names_raw = age_dist_raw.index
    age_dist_raw      = age_dist_raw.to_numpy()

    age_dist = np.zeros( (np.max(label_map['kommune_to_kommune_idx']) + 1, age_dist_raw.shape[1], len(eval(age_dist_raw[0, 0]))), dtype=float)

    for i in range(age_dist_raw.shape[0]) :

        i_out = label_map['kommune_to_kommune_idx'][kommune_names_raw[i]]

        for j in range(age_dist_raw.shape[1]) :
            age_dist[i_out, j, :] = eval(age_dist_raw[i, j])


    return age_dist


def parse_age_distribution_data_sogn(filename, label_map=None) :

    age_dist_raw      = pd.read_csv(filename, index_col=0)
    kommune_names_raw = age_dist_raw.index
    age_dist_raw      = age_dist_raw.to_numpy()

    age_dist = np.zeros( (len(label_map), age_dist_raw.shape[1], len(eval(age_dist_raw[0, 0]))), dtype=float)

    for i in range(age_dist_raw.shape[0]) :

        i_out = label_map[kommune_names_raw[i]]

        for j in range(age_dist_raw.shape[1]) :
            age_dist[i_out, j, :] = eval(age_dist_raw[i, j])


    return age_dist


def parse_household_data(filename, label_map=None) :

    household_dist_raw = pd.read_csv(filename, index_col=0)
    kommune_names_raw  = household_dist_raw.index
    household_dist_raw = household_dist_raw.to_numpy()

    household_dist = np.zeros((np.max(label_map['kommune_to_kommune_idx']) + 1, household_dist_raw.shape[1]))

    for i in range(household_dist_raw.shape[0]) :

        i_out = label_map['kommune_to_kommune_idx'][kommune_names_raw[i]]

        for j in range(household_dist_raw.shape[1]) :
            household_dist[i_out, j] = household_dist_raw[i, j] / (j + 1)

    return household_dist


def load_sogn_to_kommune_idx(filename="Data/population_information/sogn_to_kommune.csv"):
    # list of the n-th sogn in sogne_household_file to kommune idx
    return np.loadtxt(filename).astype(int)


def parse_household_data_sogn(filename) :

    df = pd.read_excel(filename, skiprows=3, engine='openpyxl')
    df = df.set_index(df.columns[0])
    df = df.fillna(0)
    df = df.rename({"6+":6},axis='columns')
    return df


def load_kommune_shapefiles(verbose=False) :
    shapefile_size = "large"
    shp_file = {}
    shp_file["large"]  = "Data/population_information/KOMMUNE.shp"

    if verbose :
        print(f"Loading {shapefile_size} kommune shape files")
    kommuner = gpd.read_file(shp_file[shapefile_size]).to_crs({"proj" : "latlong"})  # convert to lat lon, compared to UTM32_EUREF89
    #print(kommuner)

    kommune_navn, kommune_idx = np.unique(kommuner["KOMNAVN"], return_inverse=True)
    name_to_idx = dict(zip(kommune_navn, range(len(kommune_navn))))
    idx_to_name = {v : k for k, v in name_to_idx.items()}

    kommuner["idx"] = kommune_idx
    kommuner = kommuner.set_index(kommuner.columns[12])
    return kommuner, name_to_idx, idx_to_name

def load_sogne_shapefiles(verbose=False) :
    shapefile_size = "large"
    shp_file = {}
    shp_file["large"]  = "Data/population_information/SOGN.shp"

    if verbose :
        print(f"Loading {shapefile_size} sogne shape files")
    sogne = gpd.read_file(shp_file[shapefile_size]).to_crs({"proj" : "latlong"})  # convert to lat lon, compared to UTM32_EUREF89

    sogn_code = sogne['SOGNEKODE'].drop_duplicates().values.astype(int)
    sogn_idx  = np.arange(len(sogn_code))

    sogne = sogne.set_index(sogne.columns[9])
    return sogne, sogn_code, sogn_idx

def load_household_data(label_map) :

    household_dist = parse_household_data(load_yaml('cfg/files.yaml')['PeopleInHousehold'], label_map=label_map)
    age_dist = parse_age_distribution_data(load_yaml('cfg/files.yaml')['AgeDistribution'],  label_map=label_map)

    return household_dist, age_dist


def load_household_data_sogn(label_map) :

    household_dist_sogn = parse_household_data_sogn(load_yaml('cfg/files.yaml')['PeopleInHouseholdSogn'])
    age_dist = parse_age_distribution_data_sogn(load_yaml('cfg/files.yaml')['AgeDistribution'],  label_map=label_map)

    return household_dist_sogn, age_dist


def load_age_stratified_file(file) :
    """ Loads and parses the contact matrix from the .csv file specifed
        Parameters :
            file (string) : path the the .csv file
    """

    # Load using pandas
    data = pd.read_csv(file, index_col=0)

    # Get the age groups from the dataframe
    age_groups = list(data)
    age_groups = [age_group.replace('+', '-') for age_group in age_groups]

    # Extract the lowest age from the age group intervals
    lower_breaks = [int(age_group.split('-')[0]) for age_group in age_groups]

    # Get the row_names
    row_names = data.index.values

    return data.to_numpy(), row_names, lower_breaks

def load_contact_matrices(scenario = 'reference', N_labels = 1) :
    """ Loads and parses the contact matrices corresponding to the chosen scenario.
        The function first determines what the relationship between work activites and other activites are
        After the work_other_ratio has been calculated, the function returns the normalized contact matrices
        Parameters :
            scenario (string) : Name for the scenario to load
            N_labels (int) : The number of labels in the simulation. I.e the number of contact matrices to output
    """


    matrix_work  = []
    matrix_other = []
    work_other_ratio = []
    age_groups_work = []

    base_path = load_yaml('cfg/files.yaml')['contactMatrixFolder']

    filenames = []
    for label in range(N_labels) :
        # Does a specific contact matrix exist?
        if file_exists(os.path.join(base_path, scenario + '_label_' + str(label) + '_work.csv')) :
            filenames.append(os.path.join(base_path, scenario + '_label_' + str(label)))
        else :
            filenames.append(os.path.join(base_path, scenario))

    for filename_set in filenames:
        tmp_matrix_work, tmp_matrix_other, tmp_work_other_ratio, tmp_age_groups_work = load_contact_matrix_set(filename_set)

        matrix_work.append(tmp_matrix_work)
        matrix_other.append(tmp_matrix_other)
        work_other_ratio.append(tmp_work_other_ratio)
        age_groups_work.append(tmp_age_groups_work)

    return matrix_work, matrix_other, work_other_ratio, age_groups_work

def load_seasonal_model(scenario=None, offset = 0) :

    if scenario.lower() == 'none' :
        return np.ones(365)

    # Load season data 2020-12-28 - 2021-12-15
    base_path = load_yaml('cfg/files.yaml')['seasonalFolder']

    # Load data from offset and forward
    model = np.squeeze(pd.read_csv(os.path.join(base_path, scenario + '.csv'), index_col = 0)[offset:].to_numpy())

    # Scale to starting value
    return model / model[0]


def load_contact_matrix_set(matrix_path) :

    # Load the contact matrices
    matrix_work,   _, age_groups_work   = load_age_stratified_file(matrix_path + '_work.csv')
    matrix_school, _, age_groups_school = load_age_stratified_file(matrix_path + '_school.csv')
    matrix_other,  _, age_groups_other  = load_age_stratified_file(matrix_path + '_other.csv')

    # Assert the age_groups are the same
    if not age_groups_work == age_groups_other :
        raise ValueError('Age groups for work contact matrix and other contact matrix not equal')
    matrix_work = matrix_work + matrix_school

    # Determine the work-to-other ratio
    work_other_ratio = matrix_work.sum() / (matrix_other.sum() + matrix_work.sum())

    # Normalize the contact matrices after this ratio has been determined
    # TODO : Find out if lists or numpy arrays are better --- I am leaning towards using only numpy arrays
    return matrix_work.tolist(), matrix_other.tolist(), work_other_ratio, age_groups_work



def load_vaccination_schedule(cfg) :
    """ Loads and parses the vaccination schedule corresponding to the chosen scenario.
        This includes scaling the number of infections and adjusting the effective start dates
        Parameters :
            cfg (dict) : the configuration file
    """

    if cfg.Intervention_vaccination_schedule_name == 'None' :
        return np.zeros( (1, 1, len(cfg.network.work_matrix)), dtype=np.int64), np.zeros( (1, 2), dtype=np.int32)


    vaccinations_per_age_group, vaccination_schedule = load_vaccination_schedule_file(scenario = cfg.Intervention_vaccination_schedule_name)

    # Check that lengths match
    utils.test_length(vaccinations_per_age_group, cfg.Intervention_vaccination_effect_delays, "Loaded vaccination schedules does not match with the length of vaccination_effect_delays")
    utils.test_length(vaccinations_per_age_group[0][0], cfg.network.work_matrix, "Number of age groups in vaccination schedule does not match the number of age groups in contact matrices")

    # Scale and adjust the vaccination schedules
    for i in range(len(vaccinations_per_age_group)) :

        # Scale the number of vaccines
        np.multiply(vaccinations_per_age_group[i], cfg.network.N_tot / 5_800_000, out=vaccinations_per_age_group[i], casting='unsafe')

        # Determine the timing of effective vaccines
        vaccination_schedule[i] = np.array([0, (vaccination_schedule[i][-1] - vaccination_schedule[i][0]).days]) + cfg.Intervention_vaccination_effect_delays[i] - cfg.start_date_offset

    return np.array(vaccinations_per_age_group), np.array(vaccination_schedule, dtype=np.int32)


def load_vaccination_schedule_file(scenario = "reference") :
    """ Loads and parses the vaccination schedule file corresponding to the chosen scenario.
        Parameters :
            scenario (string) : Name for the scenario to load
    """
    # Prepare output files
    vaccine_counts  = []
    schedule        = []

    # Determine the number of files that matches the requested scenario
    i = 0
    filename = lambda i : f'Data/vaccination_schedule/{scenario}_{i}.csv'

    while file_exists(filename(i)) :

        # Load the contact matrices
        tmp_vaccine_counts, tmp_schedule, _ = load_age_stratified_file(filename(i))

        # Convert schedule to datetimes
        tmp_schedule = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in tmp_schedule]

        # Store the loaded schedule
        vaccine_counts.append(tmp_vaccine_counts)
        schedule.append(tmp_schedule)

        # Increment
        i += 1

    # Check if any schedule was loaded
    if len(vaccine_counts) == 0 :
        raise ValueError(f'No vaccine schedule found that matches: {scenario}')

    # Normalize the contact matrices after this ratio has been determined
    return vaccine_counts, schedule














 ######   ######  ####    ########     ###    ########    ###
##    ## ##    ##  ##     ##     ##   ## ##      ##      ## ##
##       ##        ##     ##     ##  ##   ##     ##     ##   ##
 ######   ######   ##     ##     ## ##     ##    ##    ##     ##
      ##       ##  ##     ##     ## #########    ##    #########
##    ## ##    ##  ##     ##     ## ##     ##    ##    ##     ##
 ######   ######  ####    ########  ##     ##    ##    ##     ##


def load_municipality_test_data(zfile) :
    return pd.read_csv(zfile.open('Municipality_tested_persons_time_series.csv'), sep=';', index_col=0)


def load_municipality_case_data(zfile) :
    return pd.read_csv(zfile.open('Municipality_cases_time_series.csv'), sep=';', index_col=0)


def load_age_data(zfile) :
    df = pd.read_csv(zfile.open('Cases_by_age.csv'), usecols=['Aldersgruppe', 'Antal_bekræftede_COVID-19'], dtype={'Aldersgruppe' : str, 'Antal_bekræftede_COVID-19' : str}, sep=';', index_col=0)

    # Remove thousands seperator
    col = 'Antal_bekræftede_COVID-19'
    df[col] = df[col].str.replace('.', '', regex=False)
    df[col] = df[col].astype(float)

    return df[:-1]


def newest_SSI_filename() :

    date = datetime.datetime.now()
    if date.hour < 14 :
        date -= datetime.timedelta(days=1)

    while date.isoweekday() > 5 :
        date -= datetime.timedelta(days=1)

    return date.strftime('%Y_%m_%d')

def SSI_data_missing(filename) :
    # Check if it is already downloaded
    # If it does not exist return flag
    if not os.path.exists(filename) :
        return True
    else :
        return False


def download_SSI_data(date=None,
                      download_municipality_tests=True,
                      path_municipality_tests=None,
                      download_municipality_cases=True,
                      path_municipality_cases=None,
                      download_age=True,
                      path_age=None) :
    # Parse date
    date = datetime.datetime.strptime(date, '%Y_%m_%d')


    url = 'https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata'

    with urllib.request.urlopen(url) as response :
        html = str(response.read())

    date_SSI = date.strftime('%d%m%Y')

    # Format changed on April 22nd 2021
    if date < datetime.datetime(2021, 4, 22) :
        s = re.search(f'rapport-{date_SSI}', html, re.IGNORECASE)
        if s is None :
            raise ValueError(f'No data found for date: {date}')

        data_url = html[s.start()-80:s.end()+5]
        data_url = data_url.split('="')[1] + ".zip"

    else :
        s = re.search(f'overvaagningsdata-covid19-{date_SSI}', html, re.IGNORECASE)
        if s is None :
            raise ValueError(f'No data found for date: {date}')

        data_url = html[s.start()-90:s.end()+5]
        data_url = data_url.split('="')[1] + ".zip"

    filename = date.strftime('%Y_%m_%d') + '.csv'

    with ZipFile(BytesIO(urllib.request.urlopen(data_url).read())) as zfile :

        if download_municipality_tests :
            df = load_municipality_test_data(zfile)
            save_dataframe(df, path_municipality_tests, filename)

        if download_municipality_cases :
            df = load_municipality_case_data(zfile)
            save_dataframe(df, path_municipality_cases, filename)

        if download_age :
            df = load_age_data(zfile)
            save_dataframe(df, path_age, filename)


def get_SSI_data(date=None, return_data=False, return_name=False, verbose=False) :

    if date.lower() == 'newest' :
        date = newest_SSI_filename()

    filename = date + '.csv'

    f_municipality_tests = os.path.join(load_yaml('cfg/files.yaml')['municipalityTestsFolder'], filename)
    f_municipality_cases = os.path.join(load_yaml('cfg/files.yaml')['municipalityCasesFolder'], filename)
    f_age                = os.path.join(load_yaml('cfg/files.yaml')['ageCasesFolder'], filename)

    download_municipality_tests = SSI_data_missing(f_municipality_tests)
    download_municipality_cases = SSI_data_missing(f_municipality_cases)
    download_age                = SSI_data_missing(f_age)


    if download_municipality_tests or download_municipality_cases or download_age:

        if verbose:
            print("Downloading new data")

        download_SSI_data(date=date,
                          download_municipality_tests=download_municipality_tests,
                          path_municipality_tests=os.path.dirname(f_municipality_tests),
                          download_municipality_cases=download_municipality_cases,
                          path_municipality_cases=os.path.dirname(f_municipality_cases),
                          download_age=download_age,
                          path_age=os.path.dirname(f_age))

    if return_data :
        # Load the dataframes
        df_municipality_tests = pd.read_csv(f_municipality_tests, index_col=0)
        df_municipality_cases = pd.read_csv(f_municipality_cases, index_col=0)
        df_age                = pd.read_csv(f_age,                index_col=0)

        return df_municipality_cases, df_municipality_tests, df_age

    if return_name :
        return date



def load_infection_age_distributions(initial_distribution_file, N_ages) :

    if initial_distribution_file.lower() == 'random' :
        age_distribution_infected  = np.ones(N_ages)
        age_distribution_immunized = np.ones(N_ages)

    else :

        if initial_distribution_file.lower() == 'newest' :
            date_current = newest_SSI_filename()
        else :
            date_current = initial_distribution_file

        date_delayed = datetime.datetime.strptime(date_current, '%Y_%m_%d') - datetime.timedelta(days=7)
        date_delayed = date_delayed.strftime('%Y_%m_%d')

        _, _, df_current = get_SSI_data(date=date_current, return_data=True)
        _, _, df_delayed = get_SSI_data(date=date_delayed, return_data=True)

        age_distribution_current_raw = df_current.to_numpy().flatten()
        age_distribution_delayed_raw = df_delayed.to_numpy().flatten()

        # Filter out ages of 70 and below
        age_distribution_current = age_distribution_current_raw[:N_ages]
        age_distribution_delayed = age_distribution_delayed_raw[:N_ages]

        # Add the data for the ages above 70
        age_distribution_current[-1] += np.sum(age_distribution_current[N_ages:])
        age_distribution_delayed[-1] += np.sum(age_distribution_delayed[N_ages:])

        age_distribution_infected  = age_distribution_current - age_distribution_delayed
        age_distribution_immunized = age_distribution_delayed

        if np.any(age_distribution_infected < 0) :
            raise ValueError(f'Age distribution is corrupted for {date_current}')

    return age_distribution_infected, age_distribution_immunized


def load_kommune_infection_distribution(initial_distribution_file, label_map, beta = 0.55) :

    N_kommuner = len(label_map)
    infected_per_kommune  = np.zeros(N_kommuner)
    immunized_per_kommune = np.zeros(N_kommuner)

    if initial_distribution_file.lower() == 'random' :
        infected_per_kommune  = np.ones(np.shape(infected_per_kommune))
        immunized_per_kommune = np.ones(np.shape(immunized_per_kommune))

    else :

        if initial_distribution_file.lower() == 'newest' :
            df_cases, df_tests, _ = get_SSI_data(date='newest', return_data=True)
        else :
            df_cases = pd.read_csv(os.path.join(load_yaml('cfg/files.yaml')['municipalityCasesFolder'], initial_distribution_file + '.csv'), index_col=0)
            df_tests = pd.read_csv(os.path.join(load_yaml('cfg/files.yaml')['municipalityTestsFolder'], initial_distribution_file + '.csv'), index_col=0)

        df_cases = df_cases.rename(columns={'Copenhagen' : 'København'}).drop(columns=['NA'])
        df_tests = df_tests.rename(columns={'Copenhagen' : 'København'}).drop(columns=['NA', 'Christiansø'])

        names_c  = df_cases.columns
        values_c = df_cases.to_numpy()

        names_t  = df_tests.columns
        values_t = df_tests.to_numpy()

        # Must have same dates in both datasets
        intersection = df_cases.index.intersection(df_tests.index)
        idx_c = np.isin(df_cases.index, intersection)
        idx_t = np.isin(df_tests.index, intersection)

        values_c = values_c[idx_c, :]
        values_t = values_t[idx_t, :]

        test_per_kommune   = np.zeros((len(values_t), N_kommuner))
        cases_per_kommune  = np.zeros((len(values_c), N_kommuner))


        # Match the columns indicies
        i_out_c = label_map[names_c]
        i_out_t = label_map[names_t]

        cases_per_kommune[:, i_out_c] = values_c
        test_per_kommune[:,  i_out_t] = values_t

        # Divide and correct for nans
        with np.errstate(divide="ignore", invalid="ignore") :
            incidence_per_kommune = cases_per_kommune / test_per_kommune ** beta
            incidence_per_kommune[test_per_kommune == 0] = 0

        # Swap arrays for the last 7 days
        I = len(df_cases.index) - 7

        # Fill the arrays
        immunized_per_kommune += np.sum(incidence_per_kommune[:I, :], axis=0)
        infected_per_kommune  += np.sum(incidence_per_kommune[I:, :], axis=0)

    return infected_per_kommune, immunized_per_kommune


def load_UK_fraction(start_date) :

    raw_data = pd.read_csv(load_yaml("cfg/files.yaml")["wgsDistribution"], sep=";")

    raw_data['percent'] = raw_data['yes'] / raw_data['total']

    data = pd.pivot_table(raw_data.drop(columns=['yes', 'total']), index='Week', columns='Region', values='percent')

    data.index = data.index.str.replace('[0-9]{4}-W', '', regex=True).astype(int)

    week = start_date.isocalendar().week
    if start_date.isoweekday() <= 3 :
        week -= 1

    return data.loc[week]




 ######   ######## ##    ## ######## ########  ####  ######
##    ##  ##       ###   ## ##       ##     ##  ##  ##    ##
##        ##       ####  ## ##       ##     ##  ##  ##
##   #### ######   ## ## ## ######   ########   ##  ##
##    ##  ##       ##  #### ##       ##   ##    ##  ##
##    ##  ##       ##   ### ##       ##    ##   ##  ##    ##
 ######   ######## ##    ## ######## ##     ## ####  ######


def jitclass_to_hdf5_ready_dict(jitclass, skip=["cfg", "cfg_network"]):

    if isinstance(skip, str):
        skip = [skip]

    typ = jitclass._numba_type_
    fields = typ.struct

    ints_floats_bool_set = (nb.types.Integer, nb.types.Float, nb.types.Boolean, nb.types.Set)

    d_out = {}
    for key, dtype in fields.items():
        val = getattr(jitclass, key)

        if key.lower() in skip:
            continue

        if isinstance(dtype, nb.types.ListType):
            if utils.is_nested_numba_list(val):
                d_out[key] = utils.NestedArray(val).to_dict()
            else:
                d_out[key] = list(val)
        elif isinstance(dtype, nb.types.Array):
            d_out[key] = np.array(val, dtype=dtype.dtype.name)
        elif isinstance(dtype, ints_floats_bool_set):
            d_out[key] = val
        else:
            print(f"Just ignoring {key} for now.")

    return d_out


def save_jitclass_hdf5ready(f, jitclass_hdf5ready):

    # with h5py.File(filename, "w", **hdf5_kwargs) as f:
    for key, val in jitclass_hdf5ready.items():
        if isinstance(val, dict):
            group = f.create_group(key)
            for k, v in val.items():
                group.create_dataset(k, data=v)
        else:
            f.create_dataset(key, data=val)


def load_jitclass_to_dict(f):
    d_in = {}
    # with h5py.File(filename, "r") as f:
    for key, val in f.items():
        if isinstance(val, h5py.Dataset):
            d_in[key] = val[()]
        else:
            d_tmp = {}
            for k, v in val.items():
                d_tmp[k] = v[()]
            d_in[key] = d_tmp
    return d_in


def load_My_from_dict(d_in, cfg):
    spec_my = nb_jitclass.spec_my
    my = nb_jitclass.initialize_My(cfg)
    for key, val in d_in.items():
        if isinstance(val, dict) and "content" in val and "offsets" in val:
            val = utils.NestedArray.from_dict(val).to_nested_numba_lists()

        # if read as numpy array from hdf5 but should be list, convert
        if isinstance(val, np.ndarray) and isinstance(spec_my[key], nb.types.ListType):
            val = List(val.tolist())
        setattr(my, key, val)
    return my





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





def load_yaml(filename) :
    with open(filename, encoding='utf8') as file :
        tmp = yaml.safe_load(file)

        for key, val in tmp.items() :

            # Check if val is a vaild path
            if isinstance(val, str) :
                try :
                    p = Path(val)
                    if p.exists() :
                        tmp[key] = p.__str__()
                except :
                    pass

            if isinstance(val, dict) :

                tmp[key] = utils.DotDict(val)

        return utils.DotDict(tmp)


def save_dataframe(df, path, filename) :
    if not os.path.exists(path) :
        os.makedirs(path)

    df.to_csv(os.path.join(path, filename))


def delete_file(filename) :
    try :
        Path(filename).unlink()
    except FileNotFoundError :
        pass

def file_exists(filename) :
    if isinstance(filename, str) :
        filename = Path(filename)
    return filename.exists()


def make_sure_folder_exist(filename, delete_file_if_exists=False) :
    if isinstance(filename, str) :
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if delete_file_if_exists and filename.exists() :
        filename.unlink()