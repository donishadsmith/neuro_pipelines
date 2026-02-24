import argparse
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import (
    get_beta_names,
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
    resample_seed_img,
)

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Extract the average beta for each cluster at the individual level for downstream analysis."
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help="Root of directory containing the second level data table, cluster table results, and cluster table masks.",
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
            "Used only when ``analysis_type`` is gPPI. Used to compute the individual average beta "
            "coefficient from the glm for the clusters."
        ),
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=False,
        default=None,
        help=(
            "Path to the seed mask used as the seed for the gPPI. Used only when ``analysis_type`` is gPPI. "
            "Used to compute the average beta coefficient from the glm for the seed. This will only be used "
            "if `glm_dir` is not set to None."
        ),
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
        default="nonparametric",
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


def get_nontarget_dose(second_level_glt_code):
    if second_level_glt_code == "mean":
        return None

    return list(
        {"0", "5", "10"}.difference(
            second_level_glt_code.replace("PPI_", "").split("_vs_")
        )
    )[0]


def drop_dose_rows(data_table, dose):
    return data_table[data_table["dose"] != int(dose)] if dose else data_table


def get_cluster_region_info(cluster_result_file, cluster_id, tail):
    df = pd.read_csv(cluster_result_file, sep=None, engine="python")
    cluster_id_mask = df["Cluster ID"].astype(str) == cluster_id
    tail_mask = df["Peak Stat"] > 0 if tail == "positive" else df["Peak_Stat"] < 0
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
    cluster_mask_filename,
    first_level_gltlabel,
    beta_name,
    add_condition_entity_key,
    save_excel_version,
):
    data_filename = dst_dir / cluster_mask_filename.name.replace(
        "cluster_mask.nii.gz", "individual_betas.csv"
    )

    if add_condition_entity_key:
        data_filename = str(data_filename).replace(
            first_level_gltlabel, f"{first_level_gltlabel}_condition-{beta_name}"
        )

    data_table.to_csv(data_filename, sep=",", index=None)
    if save_excel_version:
        data_table.to_excel(str(data_filename).replace(".csv", ".xlsx"), index=False)


def get_individual_interpretations(data_table, beta_name, mask_origin, analysis_type):
    betas = data_table[f"{analysis_type}_individual_{mask_origin}_beta"].to_numpy(
        copy=True
    )
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
            if "PPI_" in beta_name
            else ("increased_connectivity", "decreased_connectivity")
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

    data_table["units_of_beta_coefficient"] = "percent (percent_signal_change)"
    data_table[f"{analysis_type}_individual_beta_interpretation"] = (
        get_individual_interpretations(
            data_table, beta_name, mask_origin, analysis_type
        )
    )

    second_level_glt_code_str = cluster_mask_filename.name.split("gltcode-")[-1].split(
        "_clusterid"
    )[0]

    if "_vs_" in second_level_glt_code_str:
        first_group_label, second_group_label = second_level_glt_code_str.split("_vs_")
        data_table[f"{analysis_type}_group_beta_interpretation"] = (
            f"{first_group_label} > {second_group_label}"
            if tail == "positive"
            else f"{second_group_label} > {first_group_label}"
        )
    else:
        data_table[f"{analysis_type}_group_beta_interpretation"] = (
            f"Mean activation across doses > 0"
            if tail == "positive"
            else f"Mean activation across doses < 0"
        )

    if cluster_result_file:
        region_name, mni_coord = get_cluster_region_info(
            cluster_result_file, cluster_id, tail
        )
        data_table[f"cluster_region_id"] = region_name
        data_table[f"cluster_mni_coordinate"] = mni_coord
    else:
        data_table[f"cluster_region_id"] = cluster_id


def get_subject_beta_filenames(
    data_table, first_level_gltlabel, beta_name, parent_path=None
):
    subject_beta_filenames = data_table["InputFile"].tolist()

    if first_level_gltlabel == beta_name:
        return subject_beta_filenames

    subject_beta_filenames = [
        str(file).replace(f"_desc-{first_level_gltlabel}", f"_desc-{beta_name}")
        for file in subject_beta_filenames
    ]

    if parent_path:
        subject_beta_filenames = [
            parent_path / Path(file).name for file in subject_beta_filenames
        ]

    return subject_beta_filenames


def compute_average_betas(
    data_table,
    subject_beta_filenames,
    mask_filename,
    mask_origin="cluster",
):
    doses = data_table["dose"].tolist()
    average_betas = np.full(data_table.shape[-1], np.nan)
    mask = nib.load(mask_filename)

    if mask_origin == "seed":
        mask = resample_seed_img(mask, nib.load(subject_beta_filename[0]))

    for dose, subject_beta_filename in zip(doses, subject_beta_filenames):
        subject_beta_filename = Path(subject_beta_filename)
        subject = get_entity_value(
            subject_beta_filename.name, entity="sub", return_entity_prefix=True
        )
        contrast_img = nib.load(subject_beta_filename)
        beta_img_fdata = contrast_img.get_fdata()
        average_beta = beta_img_fdata[mask.get_fdata() == 1].mean()
        mask = (data_table["participant_id"] == subject) & (data_table["dose"] == dose)
        average_betas[mask] = average_beta

    return average_betas


def main(
    analysis_dir,
    dst_dir,
    glm_dir,
    seed_mask_path,
    task,
    analysis_type,
    method,
    save_excel_version,
):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)
    glm_dir = Path(glm_dir) or None
    seed_mask_path = Path(seed_mask_path) or None

    first_level_gltlabels = get_first_level_gltsym_codes(
        task, analysis_type, caller="extract_individual_betas"
    )

    for first_level_gltlabel in first_level_gltlabels:
        entity_key = get_contrast_entity_key(first_level_gltlabel)
        filename = (
            f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-data_table.txt"
        )
        data_table_file = next(analysis_dir.rglob(filename))
        if not data_table_file:
            LGR.critical(
                f"The following data table could not be found in {analysis_dir}: {filename}"
            )
            continue

        data_table = pd.read_csv(data_table_file, sep=None, engine="python")
        data_table["participant_id"] = data_table["participant_id"].astype(str)
        data_table["dose"] = data_table["dose"].astype(int)
        data_table[f"{analysis_type}_cluster_beta"] = np.nan
        for second_level_glt_code in get_second_level_glt_codes():
            LGR.info(
                f"Creating tabular data for TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_gltlabel}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
            )
            cluster_mask_filenames = list(
                analysis_dir.rglob(
                    f"*task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{second_level_glt_code}*desc-{method}_cluster_mask.nii.gz"
                )
            )
            if not cluster_mask_filenames:
                LGR.info(
                    f"No cluster masks for TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_gltlabel}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
                )
                continue

            truncated_df = drop_dose_rows(
                data_table, get_nontarget_dose(second_level_glt_code)
            )
            beta_names = get_beta_names(first_level_gltlabel)
            for beta_name in beta_names:
                add_condition_entity_key = beta_name != first_level_gltlabel
                subject_beta_filenames = get_subject_beta_filenames(
                    truncated_df, first_level_gltlabel, beta_name
                )

                if not subject_beta_filenames:
                    LGR.critical(f"Skipping tabular data for {beta_name}.")
                    continue

                for cluster_mask_filename in cluster_mask_filenames:
                    beta_coefficient_df = truncated_df.copy(deep=True)
                    beta_coefficient_df[f"{analysis_type}_individual_cluster_beta"] = (
                        compute_average_betas(
                            beta_coefficient_df,
                            subject_beta_filenames,
                            cluster_mask_filename,
                        )
                    )

                    beta_coefficient_df[
                        f"{analysis_type}_individual_cluster_beta_interpretation"
                    ] = get_individual_interpretations(
                        beta_coefficient_df,
                        beta_name,
                        mask_origin="cluster",
                        analysis_type=analysis_type,
                    )

                    add_info_to_data_table(
                        analysis_dir,
                        cluster_mask_filename,
                        beta_coefficient_df,
                        beta_name,
                        mask_origin="cluster",
                        analysis_type=analysis_type,
                    )

                    if "_vs_" in beta_name:
                        continue

                    if analysis_type == "gPPI" and glm_dir:
                        glm_beta_name = beta_name.removeprefix("PPI_")

                        glm_subject_beta_filenames = get_subject_beta_filenames(
                            beta_coefficient_df,
                            first_level_gltlabel,
                            glm_beta_name,
                            glm_dir,
                        )

                        beta_coefficient_df.drop(columns=["InputFile"])

                        beta_coefficient_df[f"glm_individual_cluster_beta"] = (
                            compute_average_betas(
                                beta_coefficient_df,
                                glm_subject_beta_filenames,
                                cluster_mask_filename,
                            )
                        )

                        beta_coefficient_df[
                            f"glm_individual_cluster_beta_interpretation"
                        ] = get_individual_interpretations(
                            beta_coefficient_df,
                            beta_name,
                            mask_origin="cluster",
                            analysis_type="glm",
                        )

                        if seed_mask_path:
                            if all(x in seed_mask_path.name for x in ["[", "]", ","]):
                                possible_coordinate = seed_mask_path.name.split("[")[
                                    1
                                ].split("]")[0]
                                suffix = "".join(seed_mask_path.suffixes)
                                possible_coordinate = possible_coordinate.removesuffix(
                                    suffix
                                )
                                if all(
                                    x.isdigit() for x in possible_coordinate.split(",")
                                ):
                                    beta_coefficient_df[f"seed_mni_coordinate"] = (
                                        seed_mask_path.name.split("[")[1].split("]")[0]
                                    )

                            LGR.info(
                                f"Using the following sphere mask path: {seed_mask_path}"
                            )
                            beta_coefficient_df[f"glm_individual_seed_beta"] = (
                                compute_average_betas(
                                    beta_coefficient_df,
                                    glm_subject_beta_filenames,
                                    seed_mask_path,
                                )
                            )
                            beta_coefficient_df[
                                f"glm_individual_seed_beta_interpretation"
                            ] = get_individual_interpretations(
                                beta_coefficient_df,
                                beta_name,
                                mask_origin="seed",
                                analysis_type="glm",
                            )

                    save_tabular_data(
                        beta_coefficient_df,
                        dst_dir,
                        cluster_mask_filename,
                        first_level_gltlabel,
                        beta_name,
                        add_condition_entity_key,
                        save_excel_version,
                    )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
