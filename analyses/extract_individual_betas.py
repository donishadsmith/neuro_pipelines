import argparse
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import resample_to_img

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import get_task_contrasts, needs_resampling

LGR = setup_logger(__name__)

GLT_CODES = ("5_vs_0", "10_vs_0", "10_vs_5")


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Extract the average beta for each cluster at the individual level for downstream analysis."
    )
    parser.add_argument(
        "--second_level_dir",
        dest="second_level_dir",
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


def get_cluster_region_info(cluster_id, tail, cluster_result_file):
    df = pd.read_csv(cluster_result_file, sep=None, engine="python")
    cluster_id_mask = df["Cluster ID"].astype(str) == cluster_id
    tail_mask = df["Peak Stat"] > 0 if tail == "positive" else df["Peak_Stat"] < 0
    mask = (cluster_id_mask) & (tail_mask)
    region_name = df.loc[mask, "Region"].tolist()[0]
    mni_coord_list = list(map(str, df.loc[mask, ["X", "Y", "Z"]].values.tolist()[0]))
    mni_coord = ", ".join(mni_coord_list)

    return region_name, mni_coord


def save_tabular_data(df, dst_dir, cluster_mask_filename, save_excel_version):
    data_filename = dst_dir / cluster_mask_filename.name.replace(
        "cluster_mask.nii.gz", "individual_averaged_betas.csv"
    )
    df.to_csv(data_filename, sep=",", index=None)
    if save_excel_version:
        df.to_excel(str(data_filename).replace(".csv", ".xlsx"), index=False)


def add_info_to_data_table(
    cluster_result_file, cluster_mask_filename, data_table, cluster_id, tail
):
    data_table["tail"] = tail
    first_label, second_label = (
        cluster_mask_filename.name.split("gltcode-")[-1]
        .split("_clusterid")[0]
        .split("_vs_")
    )
    data_table["cluster_interpretation"] = (
        f"{first_label} > {second_label}"
        if tail == "positive"
        else f"{second_label} > {first_label}"
    )

    if cluster_result_file:
        region_name, mni_coord = get_cluster_region_info(
            cluster_id, tail, cluster_result_file
        )
        data_table["cluster_region_id"] = region_name
        data_table["mni_coordinate"] = mni_coord
    else:
        data_table["cluster_region_id"] = cluster_id

    return data_table


def create_tabular_data(
    second_level_dir,
    dst_dir,
    data_table,
    cluster_mask_filenames,
    save_excel_version,
):
    subject_contrast_filenames = data_table["InputFile"].to_numpy(copy=True)
    data_table = data_table.drop(columns=["InputFile"])
    for cluster_mask_filename in cluster_mask_filenames:
        cluster_mask = nib.load(cluster_mask_filename)
        tail = get_entity_value(cluster_mask_filename.name, entity="tail")
        dose = (
            data_table["dose"].values.max()
            if tail == "positive"
            else data_table["dose"].values.min()
        )
        for subject_contrast_filename in subject_contrast_filenames:
            subject_contrast_filename = Path(subject_contrast_filename)
            subject = get_entity_value(subject_contrast_filename.name, entity="sub")
            contrast_img = nib.load(subject_contrast_filename)
            # Likely doesnt need resampling but just in case, resample on the first subject
            if needs_resampling(cluster_mask, contrast_img):
                cluster_mask = resample_to_img(
                    cluster_mask,
                    contrast_img,
                    interpolation="nearest",
                    copy_header=True,
                )

            contrast_img_fdata = contrast_img.get_fdata()
            average_beta = contrast_img_fdata[cluster_mask.get_fdata() == 1].mean()
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
            second_level_dir.rglob(
                f"{cluster_mask_filename.name.split('_clusterid-')[0]}_{file_desc}"
            )
        )

        data_table = add_info_to_data_table(
            cluster_result_file, cluster_mask_filename, data_table, cluster_id, tail
        )

        save_tabular_data(
            data_table, dst_dir, cluster_mask_filename, save_excel_version
        )


def main(second_level_dir, dst_dir, task, analysis_type, method, save_excel_version):
    second_level_dir = Path(second_level_dir)
    dst_dir = Path(dst_dir)
    contrasts = get_task_contrasts(
        task, analysis_type, caller="extract_individual_betas"
    )
    for contrast in contrasts:
        filename = f"task-{task}_contrast-{contrast}_desc-data_table.txt"
        data_table_file = next(second_level_dir.rglob(filename))
        if not data_table_file:
            LGR.critical(
                f"The following data table could not be found in {second_level_dir}: {filename}"
            )
            continue

        data_table = pd.read_csv(data_table_file, sep=None, engine="python")
        data_table["participant_id"] = data_table["participant_id"].astype(str)
        data_table["dose"] = data_table["dose"].astype(int)
        data_table["averaged_cluster_beta"] = np.nan
        for glt_code in GLT_CODES:
            LGR.info(
                f"Creating tabular data for TASK: {task}, CONTRAST: {contrast}, GLTCODE: {glt_code}"
            )
            cluster_mask_filenames = list(
                second_level_dir.rglob(
                    f"*task-{task}_contrast-{contrast}_gltcode-{glt_code}*desc-{method}_cluster_mask.nii.gz"
                )
            )
            if not cluster_mask_filenames:
                LGR.info(
                    f"No cluster masks for ASK: {task}, CONTRAST: {contrast}, GLTCODE: {glt_code}"
                )
                continue

            df = drop_dose_rows(data_table, get_nontarget_dose(glt_code))
            create_tabular_data(
                second_level_dir,
                dst_dir,
                df.copy(deep=True),
                cluster_mask_filenames,
                save_excel_version,
            )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
