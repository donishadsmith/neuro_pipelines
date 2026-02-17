from pathlib import Path
import subprocess

import numpy as np

from nifti2bids.logging import setup_logger

LGR = setup_logger(__name__)


def get_cosine_regressors(confounds_df):
    cosine_regressor_names = [
        col for col in confounds_df.columns if col.startswith("cosine")
    ]

    LGR.info(f"Name of cosine parameters: {cosine_regressor_names}")

    return (
        confounds_df[cosine_regressor_names].to_numpy(copy=True),
        cosine_regressor_names,
    )


def get_motion_regressors(confounds_df, n_motion_parameters):
    motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    if n_motion_parameters in [12, 18, 24]:
        derivatives = [f"{param}_derivative1" for param in motion_params]
        motion_params += derivatives

    if n_motion_parameters in [18, 24]:
        derivatives = [f"{param}_power2" for param in motion_params]
        motion_params += derivatives

    if n_motion_parameters == 24:
        power = [f"{param}_derivative1_power2" for param in motion_params]
        motion_params += power

    LGR.info(f"Using motion parameters: {motion_params}")

    return confounds_df[motion_params].to_numpy(copy=True), motion_params


def get_acompcor_component_names(confounds_json_data, n_components, strategy):
    if strategy == "separate":
        c_compcors = sorted(
            [k for k in confounds_json_data.keys() if "c_comp_cor" in k]
        )
        w_compcors = sorted(
            [k for k in confounds_json_data.keys() if "w_comp_cor" in k]
        )

        CSF = [c for c in c_compcors if confounds_json_data[c].get("Mask") == "CSF"][
            :n_components
        ]
        WM = [w for w in w_compcors if confounds_json_data[w].get("Mask") == "WM"][
            :n_components
        ]

        components_list = CSF + WM
    else:
        a_compcors = sorted(
            [k for k in confounds_json_data.keys() if "a_comp_cor" in k]
        )
        combined = [
            a for a in a_compcors if confounds_json_data[a].get("Mask") == "combined"
        ][:n_components]

        components_list = combined

    LGR.info(f"The following acompcor components will be used: {components_list}")

    return components_list


def get_global_signal_regressors(confounds_df, n_global_parameters):
    global_params = ["global_signal"]
    if n_global_parameters in [2, 3, 4]:
        global_params += ["global_signal_derivative1"]

    if n_global_parameters in [3, 4]:
        global_params += ["global_signal_power2"]

    if n_global_parameters == 4:
        global_params += ["global_signal_derivative1_power2"]

    LGR.info(f"Using global signal parameters: {global_params}")

    return confounds_df[global_params].to_numpy(copy=True), global_params


def percent_signal_change(
    subject_dir, afni_img_path, nifti_file, mask_file, censor_file
):
    mean_file = subject_dir / Path(nifti_file).name.replace("preproc_bold", "mean")
    percent_change_nifti_file = subject_dir / Path(nifti_file).name.replace(
        "preproc_bold", "percent_change"
    )
    censor_data = np.loadtxt(censor_file)
    kept_indices = np.where(censor_data == 1)[0]
    selector = ",".join(map(str, kept_indices))
    cmd_mean = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dTstat "
        f"-prefix {mean_file} "
        f"-mask {mask_file} "
        "-mean "
        f"-overwrite "
        f"'{nifti_file}[{selector}]'"
    )
    subprocess.run(cmd_mean, shell=True, check=True)
    # https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html
    # https://afni.nimh.nih.gov/pub/dist/HOWTO/howto/ht05_group/html/afni_howto5_subj.html
    # https://afni.nimh.nih.gov/pub/dist/edu/2011_03_one_day/afni_handouts/afni06_decon.pdf
    # c * min(200, a/b*100)
    cmd_calc = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dcalc "
        f"-a {nifti_file} -b {mean_file} -c {mask_file} "
        f"-expr 'c * max(-200, min(200, a/b*100))' -prefix {percent_change_nifti_file} -overwrite "
    )
    subprocess.run(cmd_calc, shell=True, check=True)

    return percent_change_nifti_file


def get_col_name(indx, regressor_positions):
    return regressor_positions[list(regressor_positions)[indx]]


def remove_collinear_columns(regressor_arr, regressor_positions, threshold=0.999):
    # To remove errors and warnings
    drop_columns = []
    for i in range(regressor_arr.shape[1]):
        for j in range(i + 1, regressor_arr.shape[1]):
            r = np.corrcoef(regressor_arr[:, i], regressor_arr[:, j])[0, 1]

            if r > threshold:
                col1 = get_col_name(i, regressor_positions)
                col2 = get_col_name(j, regressor_positions)
                drop_columns.append(j)

                LGR.critical(
                    f"Columns {col1} and {col2} are collinear ({r}), dropping {col2}."
                )

    return get_new_matrix_and_names(drop_columns, regressor_arr, regressor_positions)


def get_new_matrix_and_names(drop_columns, regressor_arr, regressor_positions):
    if not drop_columns:
        return regressor_arr, regressor_positions

    regressor_arr = np.delete(regressor_arr, drop_columns, axis=1)
    for indx in drop_columns:
        del regressor_positions[indx]

    regressor_positions = {
        indx: regressor_positions[key] for indx, key in enumerate(regressor_positions)
    }

    return regressor_arr, regressor_positions


def perform_spatial_smoothing(subject_dir, afni_img_path, nifti_file, mask_file, fwhm):
    smoothed_nifti_file = subject_dir / str(nifti_file).replace(
        "percent_change", "smoothed"
    )

    if smoothed_nifti_file.exists():
        smoothed_nifti_file.unlink()

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dBlurToFWHM "
        f"-input {nifti_file} "
        f"-mask {mask_file} "
        f"-FWHM {fwhm} "
        f"-prefix {smoothed_nifti_file} "
        "-overwrite"
    )
    LGR.info(f"Performing spatial smoothing with fwhm={fwhm}: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return smoothed_nifti_file
