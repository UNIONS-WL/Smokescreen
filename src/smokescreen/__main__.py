import os
import getpass
from typing import Union, Dict, Tuple
from jsonargparse import CLI
from jsonargparse.typing import Path_drw, Path_fr
# warnings related to sacc files
import warnings
from smokescreen import ConcealDataVector
from smokescreen.encryption import encrypt_file, decrypt_file
from smokescreen.utils import load_sacc_file
from . import __version__
warnings.filterwarnings("ignore")

# banner to be printed in the terminal
banner = rf"""

 (
 )\ )                 )
(()/(    )         ( /(    (          (      (    (
 /(_))  (      (   )\())  ))\ (    (  )(    ))\  ))\  (
(_))    )\  '  )\ ((_)\  /((_))\   )\(()\  /((_)/((_) )\ )
/ __| _((_))  ((_)| |(_)(_)) ((_) ((_)((_)(_)) (_))  _(_/(
\__ \| '  \()/ _ \| / / / -_)(_-</ _|| '_|/ -_)/ -_)| ' \))
|___/|_|_|_| \___/|_\_\ \___|/__/\__||_|  \___|\___||_||_|

 - DESC Pipeline for Concealing your Cosmology Results -
                 Version {__version__}
"""


def datavector_main(path_to_sacc: Path_fr,
                    fiducial_params: Dict[str, float],
                    shifts_dict: Dict[str, Union[float, Tuple[float, float]]],
                    seed: Union[int, str],
                    shift_type: str = 'add',
                    shift_distribution: str = 'flat',
                    path_to_output: Path_drw = None,
                    keep_original_sacc: bool = False,
                    output_suffix: str = None,
                    ) -> None:
    r"""Conceal a cosmic-shear SACC file with the default CCL theory backend.

    Args:
        path_to_sacc (str): Path to the SACC file to blind. It must contain
            exactly the cosmic-shear rows the default CCL backend models
            (galaxy_shear_cl_ee and/or galaxy_shear_xi_plus/minus), with
            weak-lensing tracers carrying n(z).
        fiducial_params (dict): Fiducial cosmological parameters (CCL-native
            names, e.g. {"sigma8": 0.8, "Omega_c": 0.25, "Omega_b": 0.05,
            "h": 0.67, "n_s": 0.96}).
        shifts_dict (dict): Shift envelopes, interpreted as deltas about zero.
            Example: {"Omega_c": (-0.05, 0.05), "sigma8": 0.05}
        seed (int, str): Seed for the blinding process (no default; must be a
            deliberate, secret choice).
        shift_type (str): Concealing factor type, 'add' or 'mult'. Default 'add'.
        shift_distribution (str): 'flat' or 'gaussian'. Default 'flat'.
        path_to_output (str): Directory to save the blinded SACC. Default None
            (uses the input file's directory).
        keep_original_sacc (bool): If True, keeps the original SACC file.
            Default False (keeps only the encrypted file).
        output_suffix (str): Custom suffix for the output file name.
    """
    print(banner)
    assert os.path.exists(path_to_sacc), f"File {path_to_sacc} does not exist."
    # reads the sacc file (returns sacc object and detected format)
    sacc_data, input_format = load_sacc_file(path_to_sacc)
    # creates the smokescreen object with the default CCL backend
    smoke = ConcealDataVector(fiducial_params, shifts_dict, sacc_data,
                              seed=seed, shift_distr=shift_distribution,
                              input_format=input_format)
    # blinds the sacc file
    smoke.calculate_concealing_factor(factor_type=shift_type)
    smoke.apply_concealing_to_likelihood_datavec()
    print(f">> User {getpass.getuser()}",
          f"used Smokescreen on {path_to_sacc} ... it is super effective!")
    # get root name of the input file
    root_name = os.path.splitext(os.path.basename(path_to_sacc))[0]
    # saves the blinded sacc file
    if path_to_output is None:
        path_to_output = os.path.dirname(path_to_sacc)
    smoke.save_concealed_datavector(path_to_output, root_name,
                                    output_format=input_format,
                                    suffix=output_suffix)
    ext = '.hdf5' if input_format == 'hdf5' else '.fits'
    _suffix = output_suffix if output_suffix is not None else "concealed_data_vector"
    outprintfile = f"{path_to_output}/{root_name}_{_suffix}{ext}"
    print(f"\nConcealed sacc file saved as:\n\t{outprintfile}")

    print(f"\nEncrypting the original sacc file {path_to_sacc} ...", end="")
    # encrypt the file
    encrypted_sacc, key = encrypt_file(path_to_sacc, path_to_output, save_file=True,
                                       keep_original=keep_original_sacc)
    print("Done!")
    print(f"Key saved as {path_to_output}/{root_name}.key")
    if keep_original_sacc is False:
        print(f"\nOriginal file {path_to_sacc} removed.")


def encrypt_main(path_to_sacc: Path_fr,
                 path_to_save: Path_drw = None,
                 keep_original: bool = False) -> None:
    """
    [!] WARNING: BY DEFAULT, IT DELETES THE ORIGINAL SACC FILE. [!]
    use the flag --keep_original true to keep the original file.

    Main function to encrypt a SACC file from the command line.

    Parameters
    ----------
    path_to_sacc : str
        Path to the SACC file to be encrypted.
    path_to_save : str, optional
        Directory to save the key used to encrypt the SACC file, and the
        encrypted file. It must exist and be writable.
        By default None [saves in the same directory as the encrypted file].
    keep_original : bool, optional
        If True, keeps the original file, by default False.
    """
    print(banner)
    # check if the file exists
    assert os.path.exists(path_to_sacc), f"File {path_to_sacc} does not exist."

    # encrypt the file
    encrypted_sacc, key = encrypt_file(path_to_sacc, path_to_save, save_file=True,
                                       keep_original=keep_original)
    print(f"\nSACC file {path_to_sacc} encrypted successfully.")
    if path_to_save is None:
        path_to_save = os.path.dirname(path_to_sacc)
    print(f"\nKey saved as {path_to_save}/{os.path.basename(path_to_sacc).split('.')[0]}.key")
    file_name = os.path.basename(path_to_sacc).split('.')[0]
    print(f"\nEncrypted file saved as {path_to_save}/{file_name}.encrpt")

    if keep_original is False:
        print(f"\nOriginal file {path_to_sacc} removed.")


def decrypt_main(path_to_sacc: Path_fr, path_to_key: Path_fr) -> None:
    """
    This function decrypts a SACC file using a key previously generated by Smokescreen.
    """
    print(banner)
    # check if the file exists
    assert os.path.exists(path_to_sacc), f"File {path_to_sacc} does not exist."
    # checks if the key exists
    assert os.path.exists(path_to_key), f"Key {path_to_key} does not exist."

    # gets the path from the encrypted file to save the decrypted file
    path = os.path.dirname(path_to_sacc)

    # decrypt the file
    _ = decrypt_file(path_to_sacc, path_to_key, save_file=True)
    print(f"\nSACC file {path_to_sacc} decrypted successfully.")
    # Extract original filename from .encrpt pattern
    basename = os.path.basename(path_to_sacc)
    if basename.endswith(".encrpt"):
        original_name = basename[:-7]  # Remove ".encrpt"
    else:
        original_name = basename.split('.')[0]
    print(f"\nDecrypted file saved as {path}/{original_name}")


def main():  # pragma: no cover
    CLI({"datavector": datavector_main,
         "encrypt": encrypt_main,
         "decrypt": decrypt_main,
         },
        as_positional=False,
        description="Smokescreen CLI Tool")


if __name__ == "__main__":  # pragma: no cover
    CLI({"datavector": datavector_main,
         "encrypt": encrypt_main,
         "decrypt": decrypt_main,
         },
        as_positional=False,
        description="Smokescreen CLI Tool")
