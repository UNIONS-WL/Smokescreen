import pytest  # noqa: F401
import os
from unittest.mock import patch, MagicMock
from cryptography.fernet import Fernet
from smokescreen import __main__
from smokescreen.__main__ import encrypt_main, decrypt_main


FIDUCIAL = {"sigma8": 0.8, "Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96}
SHIFTS = {"Omega_c": (-0.1, 0.2), "sigma8": (-0.1, 0.1)}


@patch('smokescreen.__main__.encrypt_file')
@patch('smokescreen.__main__.ConcealDataVector')
@patch('smokescreen.__main__.load_sacc_file')
def test_datavector_main(mock_load_sacc, mock_smokescreen, mock_encrypt, mock_print=None):
    path_to_sacc = "./examples/cosmic_shear/cosmicshear_sacc.fits"
    seed = 2112
    shift_type = 'add'
    shift_distribution = 'flat'
    path_to_output = "./tests/test_data/"

    mock_smoke_inst = MagicMock()
    mock_smokescreen.return_value = mock_smoke_inst
    sacc_file = MagicMock()
    mock_load_sacc.return_value = (sacc_file, 'fits')
    mock_encrypt.return_value = (b'enc', b'key')

    with patch('os.path.exists', return_value=True), patch('builtins.print'):
        __main__.datavector_main(path_to_sacc, FIDUCIAL, SHIFTS, seed,
                                 shift_type, shift_distribution,
                                 path_to_output, keep_original_sacc=True)

    mock_load_sacc.assert_called_once_with(path_to_sacc)
    mock_smokescreen.assert_called_once_with(
        FIDUCIAL, SHIFTS, sacc_file, seed=seed,
        shift_distr=shift_distribution, input_format='fits')
    mock_smoke_inst.calculate_concealing_factor.assert_called_once_with(
        factor_type='add')
    mock_smoke_inst.apply_concealing_to_likelihood_datavec.assert_called_once()
    mock_smoke_inst.save_concealed_datavector.assert_called_once_with(
        path_to_output, 'cosmicshear_sacc', output_format='fits', suffix=None
    )


@patch('smokescreen.__main__.encrypt_file')
@patch('smokescreen.__main__.ConcealDataVector')
@patch('smokescreen.__main__.load_sacc_file')
def test_datavector_main_gaussian_shift(mock_load_sacc, mock_smokescreen, mock_encrypt):
    path_to_sacc = "./examples/cosmic_shear/cosmicshear_sacc.fits"
    seed = 2112
    path_to_output = "./tests/test_data/"

    mock_smoke_inst = MagicMock()
    mock_smokescreen.return_value = mock_smoke_inst
    sacc_file = MagicMock()
    mock_load_sacc.return_value = (sacc_file, 'fits')
    mock_encrypt.return_value = (b'enc', b'key')

    with patch('os.path.exists', return_value=True), patch('builtins.print'):
        __main__.datavector_main(path_to_sacc, FIDUCIAL, SHIFTS, seed,
                                 shift_type='add', shift_distribution='gaussian',
                                 path_to_output=path_to_output, keep_original_sacc=True)

    mock_smokescreen.assert_called_once_with(
        FIDUCIAL, SHIFTS, sacc_file, seed=seed,
        shift_distr='gaussian', input_format='fits')


@patch('smokescreen.__main__.encrypt_file')
@patch('smokescreen.__main__.ConcealDataVector')
@patch('smokescreen.__main__.load_sacc_file')
def test_datavector_main_custom_suffix(mock_load_sacc, mock_smokescreen, mock_encrypt):
    path_to_sacc = "./examples/cosmic_shear/cosmicshear_sacc.fits"
    seed = 2112
    path_to_output = "./tests/test_data/"

    mock_smoke_inst = MagicMock()
    mock_smokescreen.return_value = mock_smoke_inst
    sacc_file = MagicMock()
    mock_load_sacc.return_value = (sacc_file, 'fits')
    mock_encrypt.return_value = (b'enc', b'key')

    with patch('os.path.exists', return_value=True), patch('builtins.print'):
        __main__.datavector_main(path_to_sacc, FIDUCIAL, SHIFTS, seed,
                                 shift_type='add', shift_distribution='flat',
                                 path_to_output=path_to_output,
                                 keep_original_sacc=True,
                                 output_suffix="my_suffix")

    mock_smoke_inst.save_concealed_datavector.assert_called_once_with(
        path_to_output, 'cosmicshear_sacc', output_format='fits', suffix="my_suffix"
    )


@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test_file.sacc"
    with open(file_path, "w") as f:
        f.write("This is a test SACC file.")
    return file_path


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path / "output"


@patch('smokescreen.__main__.encrypt_file')
@patch('smokescreen.__main__.getpass.getuser', return_value='test_user')
def test_encrypt_main(mock_getuser, mock_encrypt_file, temp_file, temp_dir, capsys):
    # Mock the encrypt_file function
    mock_encrypt_file.return_value = (b'encrypted_sacc', b'key')

    # Call the encrypt_main function
    encrypt_main(path_to_sacc=str(temp_file), path_to_save=str(temp_dir), keep_original=True)

    # Check if the encrypt_file function was called with the correct parameters
    mock_encrypt_file.assert_called_once_with(str(temp_file),
                                              str(temp_dir), save_file=True,
                                              keep_original=True)

    # Check if the output messages are correct
    captured = capsys.readouterr()
    assert "SACC file" in captured.out
    assert "encrypted successfully" in captured.out
    assert "Key saved as" in captured.out
    assert "Encrypted file saved as" in captured.out
    assert "Original file" not in captured.out


@patch('smokescreen.__main__.encrypt_file')
@patch('smokescreen.__main__.getpass.getuser', return_value='test_user')
def test_encrypt_main_remove_original(mock_getuser, mock_encrypt_file, temp_file, temp_dir, capsys):
    # Mock the encrypt_file function
    mock_encrypt_file.return_value = (b'encrypted_sacc', b'key')

    # Call the encrypt_main function
    encrypt_main(path_to_sacc=str(temp_file), path_to_save=str(temp_dir), keep_original=False)

    # Check if the encrypt_file function was called with the correct parameters
    mock_encrypt_file.assert_called_once_with(str(temp_file),
                                              str(temp_dir),
                                              save_file=True,
                                              keep_original=False)

    # Check if the output messages are correct
    captured = capsys.readouterr()
    assert "SACC file" in captured.out
    assert "encrypted successfully" in captured.out
    assert "Key saved as" in captured.out
    assert "Encrypted file saved as" in captured.out
    assert "Original file" in captured.out


def test_encrypt_main_file_not_found():
    # Test encrypting a nonexistent file
    with pytest.raises(AssertionError):
        encrypt_main(path_to_sacc="nonexistent_file.sacc")


def test_encrypt_main_no_save_path(temp_file, capsys):
    # Call the encrypt_main function without specifying a save path
    encrypt_main(path_to_sacc=str(temp_file), keep_original=True)

    # Check if the encrypted file and key file are saved in the same directory as the original file
    encrypted_file_path = temp_file.with_suffix('.encrpt')
    key_file_path = temp_file.with_suffix('.key')
    assert os.path.exists(encrypted_file_path)
    assert os.path.exists(key_file_path)

    # Check if the output messages are correct
    captured = capsys.readouterr()
    assert "SACC file" in captured.out
    assert "encrypted successfully" in captured.out
    assert "Key saved as" in captured.out
    assert "Encrypted file saved as" in captured.out
    assert "Original file" not in captured.out


@pytest.fixture
def temp_encrypted_file_and_key(tmp_path):
    # Create a temporary encrypted file and key
    encrypted_file_path = tmp_path / "test_file.encrpt"
    key_file_path = tmp_path / "test_file.key"
    with open(encrypted_file_path, "wb") as f:
        f.write(b"encrypted_sacc")
    with open(key_file_path, "wb") as f:
        f.write(Fernet.generate_key())
    return encrypted_file_path, key_file_path


@patch('smokescreen.__main__.decrypt_file')
@patch('smokescreen.__main__.getpass.getuser', return_value='test_user')
def test_decrypt_main(mock_getuser, mock_decrypt_file, temp_encrypted_file_and_key, capsys):
    encrypted_file_path, key_file_path = temp_encrypted_file_and_key

    # Mock the decrypt_file function
    mock_decrypt_file.return_value = b'decrypted_sacc'

    # Call the decrypt_main function
    decrypt_main(path_to_sacc=str(encrypted_file_path), path_to_key=str(key_file_path))

    # Check if the decrypt_file function was called with the correct parameters
    mock_decrypt_file.assert_called_once_with(str(encrypted_file_path),
                                              str(key_file_path),
                                              save_file=True)

    # Check if the output messages are correct
    captured = capsys.readouterr()
    assert "SACC file" in captured.out
    assert "decrypted successfully" in captured.out
    assert "Decrypted file saved as" in captured.out


def test_decrypt_main_file_not_found():
    # Test decrypting a nonexistent file
    with pytest.raises(AssertionError):
        decrypt_main(path_to_sacc="nonexistent_file.encrpt", path_to_key="nonexistent_key.key")


def test_decrypt_main_key_not_found(temp_encrypted_file_and_key):
    encrypted_file_path, _ = temp_encrypted_file_and_key

    # Test decrypting with a nonexistent key
    with pytest.raises(AssertionError):
        decrypt_main(path_to_sacc=str(encrypted_file_path), path_to_key="nonexistent_key.key")

# def test_decrypt_main_no_save_path(temp_encrypted_file_and_key, capsys):
#     encrypted_file_path, key_file_path = temp_encrypted_file_and_key

#     # Call the decrypt_main function without specifying a save path
#     decrypt_main(path_to_sacc=str(encrypted_file_path), path_to_key=str(key_file_path))

#     # Check if the decrypted file is saved in the same directory as the original file
#     decrypted_file_path = encrypted_file_path.with_suffix('.fits')
#     assert os.path.exists(decrypted_file_path)

#     # Check if the output messages are correct
#     captured = capsys.readouterr()
#     assert "SACC file" in captured.out
#     assert "decrypted successfully" in captured.out
#     assert "Decrypted file saved as" in captured.out
