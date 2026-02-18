import argparse
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import (
    get_beta_names,
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
)

LGR = setup_logger(__name__)

GLT_CODES = ("5_vs_0", "10_vs_0", "10_vs_5")


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


def get_nontarget_dose(glt_code):
    return list(
        {"0", "5", "10"}.difference(glt_code.replace("PPI_", "").split("_vs_"))
    )[0]


def drop_dose_rows(data_table, dose):
    return data_table[data_table["dose"] != int(dose)]


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
        "cluster_mask.nii.gz", "individual_averaged_betas.csv"
    )

    if add_condition_entity_key:
        data_filename = str(data_filename).replace(
            first_level_gltlabel, f"{first_level_gltlabel}_condition-{beta_name}"
        )

    data_table.to_csv(data_filename, sep=",", index=None)
    if save_excel_version:
        data_table.to_excel(str(data_filename).replace(".csv", ".xlsx"), index=False)


def get_individual_interpretations(data_table, beta_name):
    averaged_cluster_betas = data_table["averaged_cluster_beta"].to_numpy(copy=True)
    if "_vs_" in beta_name:
        first_condition_label, second_condition_label = beta_name.split("_vs_")
        interpretations = np.where(
            np.isnan(averaged_cluster_betas) | (averaged_cluster_betas == 0),
            "NaN",
            np.where(
                averaged_cluster_betas > 0,
                f"{first_condition_label} > {second_condition_label}",
                f"{second_condition_label} > {first_condition_label}",
            ),
        )
    else:
        interpretations = np.where(
            np.isnan(averaged_cluster_betas) | (averaged_cluster_betas == 0),
            "NaN",
            np.where(averaged_cluster_betas > 0, "activation", "deactivation"),
        )

    interpretations[interpretations == "NaN"] = np.nan

    return interpretations.tolist()


def add_info_to_data_table(
    cluster_result_file, cluster_mask_filename, data_table, cluster_id, tail, beta_name
):
    data_table["units_of_beta_coefficient"] = "percent (percent_signal_change)"
    data_table["individual_interpretation"] = get_individual_interpretations(
        data_table, beta_name
    )
    data_table["gltlabel"] = beta_name
    data_table["tail"] = tail
    first_group_label, second_group_label = (
        cluster_mask_filename.name.split("gltcode-")[-1]
        .split("_clusterid")[0]
        .split("_vs_")
    )
    data_table["group_interpretation"] = (
        f"{first_group_label} > {second_group_label}"
        if tail == "positive"
        else f"{second_group_label} > {first_group_label}"
    )

    if cluster_result_file:
        region_name, mni_coord = get_cluster_region_info(
            cluster_result_file, cluster_id, tail
        )
        data_table["cluster_region_id"] = region_name
        data_table["mni_coordinate"] = mni_coord
    else:
        data_table["cluster_region_id"] = cluster_id

    return data_table


def get_subject_beta_filenames(
    data_table: pd.DataFrame, first_level_gltlabel, beta_name
):
    subject_beta_filenames = data_table["InputFile"].to_numpy(copy=True).tolist()

    if first_level_gltlabel == beta_name:
        return subject_beta_filenames

    return [
        str(file).replace(f"_desc-{first_level_gltlabel}", f"_desc-{beta_name}")
        for file in subject_beta_filenames
    ]


def create_tabular_data(
    analysis_dir,
    dst_dir,
    data_table,
    subject_beta_filenames,
    cluster_mask_filenames,
    beta_name,
    first_level_gltlabel,
    add_condition_entity_key,
    save_excel_version,
):
    doses = data_table["dose"].to_numpy(copy=True).tolist()
    for cluster_mask_filename in cluster_mask_filenames:
        cluster_mask = nib.load(cluster_mask_filename)
        tail = get_entity_value(cluster_mask_filename.name, entity="tail")
        for dose, subject_beta_filename in zip(doses, subject_beta_filenames):
            subject_beta_filename = Path(subject_beta_filename)
            subject = get_entity_value(
                subject_beta_filename.name, entity="sub", return_entity_prefix=True
            )
            contrast_img = nib.load(subject_beta_filename)
            beta_img_fdata = contrast_img.get_fdata()
            average_beta = beta_img_fdata[cluster_mask.get_fdata() == 1].mean()
            mask = (data_table["participant_id"] == subject) & (
                data_table["dose"] == dose
            )
            data_table.loc[mask, "averaged_cluster_beta"] = average_beta

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

        data_table["averaged_cluster_beta"] = data_table[
            "averaged_cluster_beta"
        ].astype(float)
        data_table = add_info_to_data_table(
            cluster_result_file,
            cluster_mask_filename,
            data_table,
            cluster_id,
            tail,
            beta_name,
        )

        save_tabular_data(
            data_table,
            dst_dir,
            cluster_mask_filename,
            first_level_gltlabel,
            beta_name,
            add_condition_entity_key,
            save_excel_version,
        )


def main(analysis_dir, dst_dir, task, analysis_type, method, save_excel_version):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)
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
        data_table["averaged_cluster_beta"] = np.nan
        for glt_code in GLT_CODES:
            LGR.info(
                f"Creating tabular data for TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_gltlabel}, GLTCODE: {glt_code}"
            )
            cluster_mask_filenames = list(
                analysis_dir.rglob(
                    f"*task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{glt_code}*desc-{method}_cluster_mask.nii.gz"
                )
            )
            if not cluster_mask_filenames:
                LGR.info(
                    f"No cluster masks for TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_gltlabel}, GLTCODE: {glt_code}"
                )
                continue

            df = drop_dose_rows(data_table, get_nontarget_dose(glt_code))
            beta_names = get_beta_names(first_level_gltlabel)
            for beta_name in beta_names:
                add_condition_entity_key = beta_name != first_level_gltlabel
                subject_beta_filenames = get_subject_beta_filenames(
                    df, first_level_gltlabel, beta_name
                )

                if not subject_beta_filenames:
                    LGR.critical(f"Skipping tabular data for {beta_name}.")
                    continue

                create_tabular_data(
                    analysis_dir,
                    dst_dir,
                    df.drop(columns=["InputFile"]),
                    subject_beta_filenames,
                    cluster_mask_filenames,
                    beta_name,
                    first_level_gltlabel,
                    add_condition_entity_key,
                    save_excel_version,
                )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
