import matplotlib.pyplot as plt

from src.utils      import file_loaders
from src.plot       import plot_single_number_of_contacts
from src.simulation import simulation

from tqdm import tqdm


# This script generates a network based on the defualt network parameters and those specified in this file
# After network generation, the script computes the contact distribution for each label
# Then, matrix restrictions are applied to each label and the contact distributions are recomputed



# 1) Define network to generate
f = 0.01
verbose = True


# 2) Generate network
simulation.run_simulations({'N_tot' : int(5_800_000 * f), 'day_max' : 0, 'initial_infection_distribution' : 'random'}, num_cores_max=1, verbose=verbose)

# 3) Get the contact distribution
network_files = file_loaders.ABM_simulations(base_dir="Output/network", filetype="hdf5")


print(len(network_files))
x = x

#pdf_name = f"Figures/Number_of_contacts.pdf"
#Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

#if Path(pdf_name).exists() and not force_rerun:
#    print(f"{pdf_name} already exists")
#    return None

#with PdfPages(pdf_name) as pdf:
for network_filename in tqdm(
        network_files.iter_all_files(),
        desc="Number of contacts",
        total=len(network_files),
    ):
        # cfg = utils.string_to_dict(str(network_filename))
        if "_ID__0" in network_filename:
            fig_ax = plot_single_number_of_contacts(network_filename)
            if fig_ax is not None:
                fig, ax = fig_ax
            plt.savefig('test.png', dpi=100)
            plt.close()




# 4) Apply restrictions
# 5) get new contact distiutions