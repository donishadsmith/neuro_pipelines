import argparse
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.parsers import _is_float

from _utils import (
    create_condition_label_str,
    delete_dir,
    drop_dose_rows,
    get_beta_names,
    get_contrast_entity_key,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
    get_nontarget_dose,
    resample_seed_img,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the average beta for each cluster at "
            "the individual level for downstream analysis. Paper: https://www.nature.com/articles/nn.2303"
        )
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Root of directory containing the second level data table, "
            "cluster table results, and cluster table masks"
        ),
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--glm_dir",
        dest="glm_dir",
        required=False,
        default=None,
        help=(
            "Used only when ``analysis_type`` is gPPI. Used to compute the "
            "individual average beta coefficient from the glm for the clusters."
        ),
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=False,
        default=None,
        help=(
            "Path to the seed mask used as the seed for the gPPI. "
            "Used only when ``analysis_type`` is gPPI. "
            "Used to compute the average beta coefficient from the glm for the seed. "
            "This will only be used if `glm_dir` is not set to None."
        ),
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )
    parser.add_argument(
        "--method",
        dest="method",
        required=False,
        default="parametric",
        choices=["parametric", "nonparametric"],
        help="Whether parametric (3dlmer) or nonparametric (Palm) was used.",
    )
    parser.add_argument(
        "--save_excel_version",
        dest="save_excel_version",
        required=False,
        default=True,
        help=(
            "Save Excel version of the cluster tables "
            "to allow for certain Excel features such as highlighting"
        ),
    )

    return parser


def get_cluster_region_info(cluster_result_file, cluster_id, tail):
    df = pd.read_csv(cluster_result_file, sep=None, engine="python")
    cluster_id_mask = df["Cluster ID"].astype(str) == cluster_id
    tail_mask = df["Peak Stat"] > 0 if tail == "positive" else df["Peak Stat"] < 0
    mask = (cluster_id_mask) & (tail_mask)

    if "Region" in df.columns:
        region_name = df.loc[mask, "Region"].tolist()[0]
    else:
        region_name = cluster_id

    mni_coord_list = list(map(str, df.loc[mask, ["X", "Y", "Z"]].values.tolist()[0]))
    mni_coord = ", ".join(mni_coord_list)

    return region_name, mni_coord


def save_tabular_data(
    data_table,
    dst_dir,
    method,
    cluster_mask_filename,
    first_level_glt_label,
    second_level_glt_code,
    beta_name,
    add_condition_entity_key,
    save_excel_version,
):
    data_filename = (
        dst_dir
        / "individual_betas"
        / method
        / second_level_glt_code
        / first_level_glt_label
        / beta_name
        / cluster_mask_filename.name.replace(
            "cluster_mask.nii.gz", "individual_betas.csv"
        )
    )
    data_filename.parent.mkdir(parents=True, exist_ok=True)

    if add_condition_entity_key:
        data_filename = data_filename.parent / data_filename.name.replace(
            first_level_glt_label, f"{first_level_glt_label}_condition-{beta_name}"
        )

    data_table.to_csv(data_filename, sep=",", index=None)
    if save_excel_version:
        data_table.to_excel(str(data_filename).replace(".csv", ".xlsx"), index=False)


def get_individual_interpretations(
    data_table, beta_name, mask_origin, analysis_type, remove_PPI_prefix=False
):
    if remove_PPI_prefix:
        beta_name = beta_name.replace("PPI_", "")

    betas = data_table[
        f"{analysis_type.upper()} Individual {mask_origin.capitalize()} Beta"
    ].to_numpy(copy=True)
    if "_vs_" in beta_name:
        first_condition_label, second_condition_label = beta_name.split("_vs_")
        interpretations = np.where(
            np.isnan(betas) | (betas == 0),
            "NaN",
            np.where(
                betas > 0,
                f"{first_condition_label} > {second_condition_label}",
                f"{second_condition_label} > {first_condition_label}",
            ),
        )
    else:
        descriptions = (
            ("activation", "deactivation")
            if "PPI_" not in beta_name
            else (
                "increased_connectivity_with_seed_roi",
                "decreased_connectivity_with_seed_roi",
            )
        )
        interpretations = np.where(
            np.isnan(betas) | (betas == 0),
            "NaN",
            np.where(betas > 0, descriptions[0], descriptions[1]),
        )

    interpretations[interpretations == "NaN"] = np.nan

    return interpretations.tolist()


def add_info_to_data_table(
    analysis_dir,
    cluster_mask_filename,
    data_table,
    beta_name,
    mask_origin,
    analysis_type,
):

    tail = get_entity_value(cluster_mask_filename.name, entity="tail")
    file_desc = cluster_mask_filename.name.split(f"tail-{tail}_")[-1]
    file_desc = file_desc.replace("cluster_mask", "cluster_results").replace(
        ".nii.gz", ".csv"
    )
    cluster_id = get_entity_value(cluster_mask_filename.name, entity="clusterid")
    cluster_result_file = next(
        analysis_dir.rglob(
            f"{cluster_mask_filename.name.split('_clusterid-')[0]}_{file_desc}"
        )
    )

    data_table[f"{analysis_type.upper()} Individual Beta Interpretation"] = (
        get_individual_interpretations(
            data_table, beta_name, mask_origin, analysis_type
        )
    )

    data_table["Condition Label"] = create_condition_label_str(beta_name)

    second_level_glt_code_str = cluster_mask_filename.name.split("gltcode-")[-1].split(
        "_clusterid"
    )[0]

    if "_vs_" in second_level_glt_code_str:
        first_group_label, second_group_label = second_level_glt_code_str.split("_vs_")
        suffix = " mg MPH" if _is_float(first_group_label) else ""
        end_str = (
            "; greater activation"
            if analysis_type == "glm"
            else "; greater connectivity"
        )

        data_table[f"{analysis_type.upper()} Group Beta Interpretation"] = (
            f"{first_group_label}{suffix} > {second_group_label}{suffix}{end_str}"
            if tail == "positive"
            else f"{second_group_label}{suffix} > {first_group_label}{suffix}{end_str}"
        )
    else:
        interpretation = (
            "activation"
            if analysis_type == "glm"
            else "increased connectivity with seed ROI"
        )

        data_table[f"{analysis_type.upper()} Group Beta Interpretation"] = (
            f"mean {interpretation.removeprefix('increased')} across doses > 0"
            if tail == "positive"
            else f"mean {interpretation.removeprefix('increased')} across doses < 0"
        )

    if cluster_result_file:
        region_name, mni_coord = get_cluster_region_info(
            cluster_result_file, cluster_id, tail
        )
        data_table["Cluster Region ID"] = region_name
        data_table["Cluster MNI Coordinate"] = mni_coord
    else:
        data_table["Cluster Region ID"] = cluster_id


def get_subject_beta_filenames(
    data_table,
    first_level_glt_label,
    beta_name,
    parent_path=None,
):
    subject_beta_filenames = data_table["InputFile"].tolist()

    if first_level_glt_label == beta_name:
        return subject_beta_filenames

    subject_beta_filenames = [
        str(file).replace(f"_desc-{first_level_glt_label}", f"_desc-{beta_name}")
        for file in subject_beta_filenames
    ]

    if parent_path:
        subject_beta_filenames = [
            next(parent_path.rglob(f"*{Path(file).name}*"), None)
            for file in subject_beta_filenames
        ]
        subject_beta_filenames = [
            str(file) if file else float("NaN") for file in subject_beta_filenames
        ]

    return subject_beta_filenames


def compute_average_betas(
    data_table,
    subject_beta_filenames,
    mask_filename,
    mask_origin="cluster",
):
    subjects = data_table["Subj"].tolist()
    doses = data_table["dose"].tolist()
    average_betas = np.full(data_table.shape[0], np.nan)
    mask_img = nib.load(mask_filename)

    if mask_origin == "seed":
        mask_img = resample_seed_img(mask_img, nib.load(subject_beta_filenames[0]))

    for subject, dose, subject_beta_filename in zip(subjects, doses, subject_beta_filenames):
        subject_mask = (data_table["Subj"] == subject) & (data_table["dose"] == dose)
        if pd.isna(subject_beta_filename):
            average_betas[subject_mask] = float("NaN")
            continue

        subject_beta_filename = Path(subject_beta_filename)
        beta_img = nib.load(subject_beta_filename)
        beta_img_fdata = beta_img.get_fdata()
        average_beta = beta_img_fdata[mask_img.get_fdata() == 1].mean()
        average_betas[subject_mask] = average_beta

    return average_betas


def main(
    analysis_dir,
    dst_dir,
    glm_dir,
    seed_mask_path,
    cohort,
    task,
    analysis_type,
    method,
    save_excel_version,
):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)
    if glm_dir and Path(glm_dir).exists():
        glm_dir = Path(glm_dir)
    else:
        glm_dir = None

    if seed_mask_path and Path(seed_mask_path).exists():
        seed_mask_path = Path(seed_mask_path)
    else:
        seed_mask_path = None

    delete_dir(dst_dir / "individual_betas" / method)

    first_level_glt_labels = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="extract_individual_betas"
    )

    for first_level_glt_label in first_level_glt_labels:
        entity_key = get_contrast_entity_key(first_level_glt_label)
        filename = (
            f"task-{task}_{entity_key}-{first_level_glt_label}_desc-data_table.txt"
        )
        data_table_file = next(analysis_dir.rglob(filename))
        if not data_table_file:
            LGR.warning(
                f"The following data table could not be found in {analysis_dir}: {filename}"
            )
            continue

        data_table = pd.read_csv(data_table_file, sep=None, engine="python")
        data_table["Subj"] = data_table["Subj"].astype(str)
        if cohort == "kids":
            data_table["dose"] = data_table["dose"].astype(int)

        for second_level_glt_code in get_second_level_glt_codes(cohort):
            LGR.info(
                f"Creating tabular data for TASK: {task}, FIRST LEVEL GLTLABEL: "
                f"{first_level_glt_label}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
            )
            cluster_mask_filenames = list(
                analysis_dir.rglob(
                    f"*task-{task}_{entity_key}-{first_level_glt_label}"
                    f"_gltcode-{second_level_glt_code}*desc-{method}_cluster_mask.nii.gz"
                )
            )
            if not cluster_mask_filenames:
                LGR.info(
                    f"No cluster masks for TASK: {task}, FIRST LEVEL GLTLABEL: "
                    f"{first_level_glt_label}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
                )
                continue

            truncated_df = drop_dose_rows(
                data_table,
                get_nontarget_dose(second_level_glt_code, cohort),
                only_complete_cases=(method == "nonparametric"),
            )
            # The individual conditions for gPPI are main effects and should not be interpreted
            # since there is an interaction term in the model
            beta_names = get_beta_names(
                first_level_glt_label,
                create_sub_conditions=(analysis_type == "glm"),
            )
            for beta_name in beta_names:
                subject_beta_filenames = get_subject_beta_filenames(
                    truncated_df,
                    first_level_glt_label,
                    beta_name,
                )

                if not subject_beta_filenames:
                    LGR.warning(f"Skipping tabular data for {beta_name}.")
                    continue

                for cluster_mask_filename in cluster_mask_filenames:
                    beta_coefficient_df = truncated_df.copy(deep=True)
                    beta_coefficient_df[
                        f"{analysis_type.upper()} Individual Cluster Beta"
                    ] = compute_average_betas(
                        beta_coefficient_df,
                        subject_beta_filenames,
                        cluster_mask_filename,
                    )

                    add_info_to_data_table(
                        analysis_dir,
                        cluster_mask_filename,
                        beta_coefficient_df,
                        beta_name,
                        mask_origin="cluster",
                        analysis_type=analysis_type,
                    )

                    if analysis_type == "gPPI" and glm_dir:
                        glm_beta_name = beta_name.replace("PPI_", "")

                        glm_subject_beta_filenames = get_subject_beta_filenames(
                            beta_coefficient_df,
                            first_level_glt_label,
                            glm_beta_name,
                            parent_path=glm_dir,
                        )
                        if pd.Series(glm_subject_beta_filenames).isna().all():
                            continue

                        beta_coefficient_df["GLM Individual Cluster Beta"] = (
                            compute_average_betas(
                                beta_coefficient_df,
                                glm_subject_beta_filenames,
                                cluster_mask_filename,
                            )
                        )

                        beta_coefficient_df[
                            "GLM Individual Cluster Beta Interpretation"
                        ] = get_individual_interpretations(
                            beta_coefficient_df,
                            beta_name,
                            mask_origin="cluster",
                            analysis_type="glm",
                            remove_PPI_prefix=True,
                        )

                        if seed_mask_path:
                            possible_coordinate = get_coordinate_from_filename(
                                seed_mask_path
                            )
                            if possible_coordinate:
                                beta_coefficient_df["Seed MNI Coordinate"] = (
                                    possible_coordinate
                                )

                            LGR.info(
                                f"Using the following seed mask path: {seed_mask_path}"
                            )

                            beta_coefficient_df["GLM Individual Seed Beta"] = (
                                compute_average_betas(
                                    beta_coefficient_df,
                                    glm_subject_beta_filenames,
                                    seed_mask_path,
                                    mask_origin="seed",
                                )
                            )
                            beta_coefficient_df[
                                "GLM Individual Seed Beta Interpretation"
                            ] = get_individual_interpretations(
                                beta_coefficient_df,
                                beta_name,
                                mask_origin="seed",
                                analysis_type="glm",
                                remove_PPI_prefix=True,
                            )

                    if analysis_type == "gPPI":
                        beta_coefficient_df["GPPI Units of Beta Coefficient"] = (
                            "unitless"
                        )

                    if "GLM Individual Cluster Beta" in beta_coefficient_df.columns:
                        beta_coefficient_df["GLM Units of Beta Coefficient"] = (
                            "percent (percent signal change)"
                        )

                    if "InputFile" in beta_coefficient_df.columns:
                        beta_coefficient_df = beta_coefficient_df.drop(
                            columns=["InputFile"]
                        )

                    beta_coefficient_df["Analysis Method"] = f"{method} {analysis_type}"

                    add_condition_entity_key = beta_name != first_level_glt_label
                    save_tabular_data(
                        beta_coefficient_df,
                        dst_dir,
                        method,
                        cluster_mask_filename,
                        first_level_glt_label,
                        second_level_glt_code,
                        beta_name,
                        add_condition_entity_key,
                        save_excel_version,
                    )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
