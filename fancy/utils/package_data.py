from pathlib import Path
from pkg_resources import resource_filename


def get_path_to_energy_loss_tables(file_name: str) -> Path:

    file_path = resource_filename(
        "fancy", "physics/energy_loss/tables/%s" % file_name
    )

    return Path(file_path)

def get_path_to_exposure_tables(file_name: str) -> Path:

    file_path = resource_filename(
        "fancy", "physics/effective_exposure/tables/%s" % file_name
    )

    return Path(file_path)


def get_path_to_stan_includes(model_type: str) -> Path:

    include_path = resource_filename("fancy", "interfaces/stan/%s" % model_type)

    return Path(include_path)


def get_path_to_stan_file(model_type: str, file_name : str) -> Path:

    file_path = resource_filename("fancy", "interfaces/stan/%s/%s" % (model_type, file_name))

    return Path(file_path)


def get_path_to_lens(lens_name: str) -> Path:

    lens_path = resource_filename(
        "fancy", "physics/gmf/gmf_lens/%s/lens.cfg" % lens_name
    )

    return Path(lens_path)

def get_path_to_kappa_theta(file_name : str = "kappa_theta_map.pkl") -> Path:

    kappa_theta_path = resource_filename(
        "fancy", "utils/resources/{0:s}".format(file_name)
    )
    return Path(kappa_theta_path)

def get_path_to_meanlnA(file_name : str = "meanlnA_logE_fit") -> Path:

    meanlnA_path = resource_filename(
        "fancy", "utils/resources/{0:s}".format(file_name)
    )
    return Path(meanlnA_path)

def get_path_to_prince_config(file_name : str) -> Path:
    prince_config_path = resource_filename(
        "fancy", "physics/energy_loss/prince_config/{0:s}".format(file_name)
    )
    return Path(prince_config_path)

def get_path_to_injection_solvers(file_name : str = "injection_solvers.pkl") -> Path:
    injection_solvers_path = resource_filename(
        "fancy", "physics/energy_loss/injection_solvers/{0:s}".format(file_name)
    )
    return Path(injection_solvers_path)

def get_path_to_loss_length_tables(file_name : str = "loss_length_tables.pkl") -> Path:
    loss_length_tables_path = resource_filename(
        "fancy", "physics/energy_loss/loss_length_tables/{0:s}".format(file_name)
    )
    return Path(loss_length_tables_path)

def get_path_to_datafiles(file_name : str = "sourcedata.h5") -> Path:

    data_path = resource_filename(
        "fancy", "utils/resources/{0:s}".format(file_name)
    )
    return Path(data_path)
