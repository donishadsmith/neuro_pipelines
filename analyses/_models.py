import subprocess
from pathlib import Path

import numpy as np

from bidsaid.logging import setup_logger

from _first_level_utils import is_timing_file_empty, EVENT_RELATED_TASKS

LGR = setup_logger(__name__)


def create_design_matrix(
    subject_dir,
    afni_img_path,
    smoothed_nifti_file,
    mask_file,
    censor_file,
    deconvolve_cmd,
    cosine_regressor_names,
):
    design_matrix_file = subject_dir / str(smoothed_nifti_file).replace(
        "smoothed.nii.gz", "design_matrix.1D"
    )

    polort = 0 if cosine_regressor_names else "A"
    LGR.info(f"Using polort {polort} for 3dDeconvolve.")

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dDeconvolve "
        f"-input {smoothed_nifti_file} "
        f"-mask {mask_file} "
        f"-censor {censor_file} "
        f"-polort {polort} "
        "-local_times "
        f"{deconvolve_cmd['num_stimts']} "
        f"{deconvolve_cmd['args']} "
        f"-x1D {design_matrix_file} "
        "-x1D_stop "
        "-overwrite"
    )

    LGR.info(f"Running 3dDeconvolve to create design matrix: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return design_matrix_file


def perform_first_level(
    subject_dir,
    afni_img_path,
    design_matrix_file,
    smoothed_nifti_file,
    mask_file,
):
    stats_file_relm = subject_dir / Path(smoothed_nifti_file).name.replace(
        "smoothed", "stats"
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dREMLfit "
        f"-matrix {design_matrix_file} "
        f"-input {smoothed_nifti_file} "
        f"-mask {mask_file} "
        "-fout -tout "
        "-verb "
        f"-Rbuck {stats_file_relm} "
        "-overwrite"
    )

    LGR.info(
        f"Running 3dREMLfit for first level accounting for auto-correlation: {cmd}"
    )
    subprocess.run(cmd, shell=True, check=True)

    return stats_file_relm


def construct_deconvolve_cmd(
    analysis_type,
    stims,
    gltsyms,
    nuisance_regressors_file,
    seed_timeseries_file,
    format_string=True,
):
    deconvolve_cmd = {
        "num_stimts": "-num_stimts {num_labels} ",
        "args": f"{{stims}} -ortvec {nuisance_regressors_file} Nuisance {{gltsyms}}",
    }

    if analysis_type == "glm":
        stims = tuple(
            [
                stim
                for stim in stims
                if "PPI_" not in stim or not stim.startswith("-stim_file")
            ]
        )
        gltsyms = tuple([gltsym.replace("PPI_", "") for gltsym in gltsyms])
    else:
        seed_name = Path(seed_timeseries_file).name.split("_desc")[0]
        first_stim = (
            f"-stim_file {{label}} {seed_timeseries_file} -stim_label {{label}} {seed_name} ",
        )
        stims = first_stim + stims

    if not format_string:
        return deconvolve_cmd, stims, gltsyms

    deconvolve_cmd["num_stimts"] = deconvolve_cmd["num_stimts"].format(
        num_labels=len(stims)
    )

    stim_string = ""
    for label, stim in enumerate(stims, start=1):
        stim_string += stim.format(label=label)

    gltsyms_string = ""
    for sym_string in gltsyms:
        gltsyms_string += sym_string

    deconvolve_cmd["args"] = deconvolve_cmd["args"].format(
        stims=stim_string, gltsyms=gltsyms_string
    )

    return deconvolve_cmd


def get_task_deconvolve_kids_cmd(
    task,
    timing_dir,
    nuisance_regressors_file,
    analysis_type,
    ppi_dir=None,
    seed_timeseries_file=None,
):
    ppi_dir = ppi_dir or Path("")
    if task in EVENT_RELATED_TASKS:
        deconvolve_cmd = create_dynamic_deconvolve_cmd(
            task,
            timing_dir,
            nuisance_regressors_file,
            analysis_type,
            ppi_dir,
            seed_timeseries_file,
        )

        return deconvolve_cmd

    if task == "nback":
        stims = (
            f"-stim_times {{label}} {timing_dir / 'cue.1D'} 'BLOCK(2, 1)' -stim_label {{label}} cue ",
            f"-stim_times {{label}} {timing_dir / 'center.1D'} 'BLOCK(32, 1)' -stim_label {{label}} center ",
            f"-stim_times {{label}} {timing_dir / '1-back.1D'} 'BLOCK(32, 1)' -stim_label {{label}} 1-back ",
            f"-stim_times {{label}} {timing_dir / '2-back.1D'} 'BLOCK(32, 1)' -stim_label {{label}} 2-back ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_cue.1D'} -stim_label {{label}} PPI_cue ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_center.1D'} -stim_label {{label}} PPI_center ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_1-back.1D'} -stim_label {{label}} PPI_1-back ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_2-back.1D'} -stim_label {{label}} PPI_2-back ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_1-back -1*PPI_center' -glt_label 1 PPI_1-back_vs_PPI_center ",
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_center' -glt_label 2 PPI_2-back_vs_PPI_center ",
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_1-back' -glt_label 3 PPI_2-back_vs_PPI_1-back ",
        )
    elif task in ["mtle", "mtlr"]:
        mtl_type = "encoding" if task == "mtle" else "retrieval"
        stims = (
            f"-stim_times {{label}} {timing_dir / 'cue.1D'} 'BLOCK(2, 1)' -stim_label {{label}} cue ",
            f"-stim_times {{label}} {timing_dir / f'neutral_{mtl_type}.1D'} 'BLOCK(18, 1)' -stim_label {{label}} {f'neutral_{mtl_type}'} ",
            f"-stim_file {{label}} {ppi_dir / f'PPI_cue.1D'} -stim_label {{label}} PPI_cue ",
            f"-stim_file {{label}} {ppi_dir / f'PPI_neutral_{mtl_type}.1D'} -stim_label {{label}} {f'PPI_neutral_{mtl_type}'} ",
        )
        gltsyms = (" ",)
    elif task == "princess":
        stims = (
            f"-stim_times {{label}} {timing_dir / 'switch.1D'} 'BLOCK(52, 1)' -stim_label {{label}} switch ",
            f"-stim_times {{label}} {timing_dir / 'nonswitch.1D'} 'BLOCK(52, 1)' -stim_label {{label}} nonswitch ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_switch.1D'} -stim_label {{label}} PPI_switch ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_nonswitch.1D'} -stim_label {{label}} PPI_nonswitch ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_switch -1*PPI_nonswitch' -glt_label 1 PPI_switch_vs_PPI_nonswitch ",
        )

    return construct_deconvolve_cmd(
        analysis_type, stims, gltsyms, nuisance_regressors_file, seed_timeseries_file
    )


def get_task_deconvolve_adults_cmd(
    task,
    timing_dir,
    nuisance_regressors_file,
    analysis_type,
    ppi_dir=None,
    seed_timeseries_file=None,
):
    ppi_dir = ppi_dir or Path("")
    if task in EVENT_RELATED_TASKS:
        deconvolve_cmd = create_dynamic_deconvolve_cmd(
            task,
            timing_dir,
            nuisance_regressors_file,
            analysis_type,
            ppi_dir,
            seed_timeseries_file,
        )

        return deconvolve_cmd

    if task == "nback":
        stims = (
            f"-stim_times {{label}} {timing_dir / '0-back.1D'} 'BLOCK(30, 1)' -stim_label {{label}} 0-back ",
            f"-stim_times {{label}} {timing_dir / '2-back.1D'} 'BLOCK(30, 1)' -stim_label {{label}} 2-back ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_0-back.1D'} -stim_label {{label}} PPI_0-back ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_2-back.1D'} -stim_label {{label}} PPI_2-back ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_0-back' -glt_label 1 PPI_2-back_vs_PPI_0-back ",
        )
    elif task in ["mtle", "mtlr"]:
        mtl_type = "encoding" if task == "mtle" else "retrieval"
        stims = (
            f"-stim_times {{label}} {timing_dir / 'cue.1D'} 'BLOCK(2, 1)' -stim_label {{label}} cue ",
            f"-stim_times {{label}} {timing_dir / f'neutral_{mtl_type}.1D'} 'BLOCK(18, 1)' -stim_label {{label}} {f'neutral_{mtl_type}'} ",
            f"-stim_times {{label}} {timing_dir / f'aversive_{mtl_type}.1D'} 'BLOCK(18, 1)' -stim_label {{label}} {f'aversive_{mtl_type}'} ",
            f"-stim_file {{label}} {ppi_dir / 'PPI_cue.1D'} -stim_label {{label}} PPI_cue ",
            f"-stim_file {{label}} {ppi_dir / f'PPI_neutral_{mtl_type}.1D'} -stim_label {{label}} {f'PPI_neutral_{mtl_type}'} ",
            f"-stim_file {{label}} {ppi_dir / f'PPI_aversive_{mtl_type}.1D'} -stim_label {{label}} {f'PPI_aversive_{mtl_type}'} ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_aversive_encoding -1*PPI_neutral_encoding' -glt_label 1 PPI_aversive_encoding_vs_PPI_neutral_encoding ",
        )

    return construct_deconvolve_cmd(
        analysis_type, stims, gltsyms, nuisance_regressors_file, seed_timeseries_file
    )


def create_dynamic_deconvolve_cmd(
    task,
    timing_dir,
    nuisance_regressors_file,
    analysis_type,
    ppi_dir,
    seed_timeseries_file=None,
):
    # Dynamically create the flanker and nogo contrasts to avoid including contrasts that
    # have no data
    if task == "flanker":
        stims = (
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} congruent ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} incongruent ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} nogo ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} neutral ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} errors ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_congruent ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_incongruent ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_nogo ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_neutral ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_errors ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_incongruent -1*PPI_congruent' -glt_label {label} PPI_incongruent_vs_PPI_congruent ",
            "-gltsym 'SYM: +1*PPI_nogo -1*PPI_neutral' -glt_label {label} PPI_nogo_vs_PPI_neutral ",
        )
        files = [
            "congruent.1D",
            "incongruent.1D",
            "nogo.1D",
            "neutral.1D",
            "errors.1D",
        ]
    else:
        stims = (
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} go ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} nogo ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} errors ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_go ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_nogo ",
            "-stim_file {label} {ppi_file} -stim_label {label} PPI_errors ",
        )
        gltsyms = (
            "-gltsym 'SYM: +1*PPI_nogo -1*PPI_go' -glt_label {label} PPI_nogo_vs_PPI_go ",
        )
        files = ["go.1D", "nogo.1D", "errors.1D"]

    deconvolve_cmd, stims, gltsyms = construct_deconvolve_cmd(
        analysis_type,
        stims,
        gltsyms,
        nuisance_regressors_file,
        seed_timeseries_file,
        format_string=False,
    )

    empty_mask = np.array([is_timing_file_empty(timing_dir / file) for file in files])

    nonempty_files = np.array(files)[~empty_mask]
    keep_trial_regressors = [file.removesuffix(".1D") for file in nonempty_files]
    if analysis_type == "gPPI":
        keep_trial_regressors += [
            f"PPI_{trial_type}" for trial_type in keep_trial_regressors
        ]
        stim_string = f"{stims[0]} ".format(label=1)
        start_indx = 2
        stims = stims[1:]
    else:
        stim_string = ""
        start_indx = 1

    # Only keep stims without empty files
    for label, regressor in enumerate(keep_trial_regressors, start=start_indx):
        bool_list = [
            regressor == stim_string.rstrip().split(" ")[-1] for stim_string in stims
        ]

        stim = stims[bool_list.index(True)]
        if "PPI_" in stim:
            stim_string += stim.format(
                label=label, ppi_file=ppi_dir / f"{regressor}.1D"
            )
        else:
            stim_string += stim.format(
                label=label,
                timing_file=timing_dir / f"{regressor}.1D",
            )

    deconvolve_cmd["num_stimts"] = deconvolve_cmd["num_stimts"].format(num_labels=label)

    # Only keep gltsym with all conditions
    kept_gltsyms = []
    for gltsym in gltsyms:
        glt_label = gltsym.rstrip().split(" ")[-1]
        glt_label_parts = glt_label.split("_vs_")
        if all(
            glt_label_part.removeprefix("PPI_") in keep_trial_regressors
            for glt_label_part in glt_label_parts
        ):
            kept_gltsyms.append(gltsym)

    gltsyms_string = ""
    for label, gltsym in enumerate(kept_gltsyms, start=1):
        gltsyms_string += gltsym.format(label=label)

    deconvolve_cmd["args"] = deconvolve_cmd["args"].format(
        stims=stim_string, gltsyms=gltsyms_string
    )

    return deconvolve_cmd
