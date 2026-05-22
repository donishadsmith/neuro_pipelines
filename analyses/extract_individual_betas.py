"""
Useful Notes for the Group Contrasts
------------------------------------
For all group contrasts (Group A - Group B, so does not include the mean contrast),
since significant clusters and the subsequent averaged betas of these clusters are
derived from the group contrast, Those averagef cluster betas will be highly correlated group (predictor)
variable (i.e., dose). Consequently, they should not be used in analysis, where
one variable is regressed onto the other (e.g., linear regression, correlation, mediation).
This is circularity and will result in inflated correlation/beta values.

https://www.nature.com/articles/nn.2303

Interaction effects (i.e. group * average_cluster_beta -> Y) can be done for exploratory behavioral analyses.
The variance of average_cluster_beta will be larger due to the group differences, which would technically
reduce the standard error of the beta coefficient. However, since group and averaged cluster betas will be collinear, the variance inflation factor
(VIF) will likely be higher, which will inflate the beta coefficient standard errors by (SE) * sqrt(VIF).
This will make it more difficult to detect effects.

https://en.wikipedia.org/wiki/Variance_inflation_factor
"""

import argparse
from pathlib import Path

import pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.parsers import _is_float

from _utils import (
    compute_average_betas,
    create_condition_label_str,
    delete_dir,
    drop_dose_rows,
    format_beta_df,
    get_beta_names,
    get_contrast_entity_key,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    get_individual_interpretations,
    get_second_level_glt_codes,
    get_nontarget_dose,
    get_subject_beta_filenames,
    needs_complete_cases,
    save_tabular_data,
    validate_optional_path,
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
        default=False,
        help=(
            "Save Excel version of the cluster tables "
            "to allow for certain Excel features such as highlighting"
        ),
    )

    return parser


def get_cluster_region_info(cluster_results_df, cluster_id, tail):
    cluster_id_mask = cluster_results_df["Cluster ID"].astype(str) == cluster_id
    tail_mask = (
        cluster_results_df["Peak Stat"] > 0
        if tail == "positive"
        else cluster_results_df["Peak Stat"] < 0
    )
    mask = (cluster_id_mask) & (tail_mask)

    if "Region" in cluster_results_df.columns:
        region_name = cluster_results_df.loc[mask, "Region"].tolist()[0]
    else:
        region_name = cluster_id

    mni_coord_list = list(
        map(str, cluster_results_df.loc[mask, ["X", "Y", "Z"]].values.tolist()[0])
    )
    mni_coord = ", ".join(mni_coord_list)

    return region_name, mni_coord


def add_info_to_data_table(
    cluster_results_df,
    cluster_mask_filename,
    data_table,
    beta_name,
    mask_origin,
    analysis_type,
):

    tail = get_entity_value(cluster_mask_filename.name, entity="tail")
    cluster_id = get_entity_value(cluster_mask_filename.name, entity="clusterid")

    data_table[f"{analysis_type.upper()}_Individual_Beta_Interpretation"] = (
        get_individual_interpretations(
            data_table, beta_name, mask_origin, analysis_type
        )
    )

    data_table["Condition_Label"] = create_condition_label_str(beta_name)

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

        data_table[f"{analysis_type.upper()}_Group_Beta_Interpretation"] = (
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

        data_table[f"{analysis_type.upper()}_Group_Beta_Interpretation"] = (
            f"mean {interpretation.removeprefix('increased')} across doses > 0"
            if tail == "positive"
            else f"mean {interpretation.removeprefix('increased')} across doses < 0"
        )

    region_name, mni_coord = get_cluster_region_info(
        cluster_results_df, cluster_id, tail
    )
    data_table["Cluster_Region_ID"] = region_name
    data_table["Cluster_MNI_Coordinate"] = mni_coord


def get_cluster_results_df(analysis_dir, cluster_mask_filename):
    tail = get_entity_value(cluster_mask_filename.name, entity="tail")
    file_desc = cluster_mask_filename.name.split(f"tail-{tail}_")[-1]
    file_desc = file_desc.split("_cluster_mask")[0] + "_cluster_results.csv"
    cluster_result_file = next(
        analysis_dir.rglob(
            f"{cluster_mask_filename.name.split('_clusterid-')[0]}_{file_desc}"
        )
    )

    return pd.read_csv(cluster_result_file, sep=None, engine="python")


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
    glm_dir = validate_optional_path(glm_dir)
    seed_mask_path = validate_optional_path(seed_mask_path)

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
                    f"_gltcode-{second_level_glt_code}*desc-{method}_cluster_mask_*.nii.gz"
                )
            )
            if not cluster_mask_filenames:
                LGR.info(
                    f"No cluster masks for TASK: {task}, FIRST LEVEL GLTLABEL: "
                    f"{first_level_glt_label}, SECOND LEVEL GLTCODE: {second_level_glt_code}"
                )
                continue

            cluster_results_df = get_cluster_results_df(
                analysis_dir, cluster_mask_filenames[0]
            )

            truncated_df = drop_dose_rows(
                data_table,
                get_nontarget_dose(second_level_glt_code, cohort),
                only_complete_cases=needs_complete_cases(method),
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
                        f"{analysis_type.upper()}_Individual_Cluster_Beta"
                    ] = compute_average_betas(
                        beta_coefficient_df,
                        subject_beta_filenames,
                        cluster_mask_filename,
                    )

                    add_info_to_data_table(
                        cluster_results_df,
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
                        if not pd.Series(glm_subject_beta_filenames).isna().all():
                            beta_coefficient_df["GLM_Individual_Cluster_Beta"] = (
                                compute_average_betas(
                                    beta_coefficient_df,
                                    glm_subject_beta_filenames,
                                    cluster_mask_filename,
                                )
                            )

                            beta_coefficient_df[
                                "GLM_Individual_Cluster_Beta_Interpretation"
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
                                    beta_coefficient_df["Seed_MNI_Coordinate"] = (
                                        possible_coordinate
                                    )

                                LGR.info(
                                    f"Using the following seed mask path: {seed_mask_path}"
                                )

                                beta_coefficient_df["GLM_Individual_Seed_Beta"] = (
                                    compute_average_betas(
                                        beta_coefficient_df,
                                        glm_subject_beta_filenames,
                                        seed_mask_path,
                                        mask_origin="seed",
                                    )
                                )
                                beta_coefficient_df[
                                    "GLM_Individual_Seed_Beta_Interpretation"
                                ] = get_individual_interpretations(
                                    beta_coefficient_df,
                                    beta_name,
                                    mask_origin="seed",
                                    analysis_type="glm",
                                    remove_PPI_prefix=True,
                                )

                    beta_coefficient_df = format_beta_df(
                        analysis_type, beta_coefficient_df, method
                    )

                    add_condition_entity_key = beta_name != first_level_glt_label
                    output_dir = (
                        dst_dir
                        / "individual_betas"
                        / method
                        / second_level_glt_code
                        / first_level_glt_label
                        / beta_name
                    )
                    output_csv_name = cluster_mask_filename.name.replace(
                        "_cluster_mask_", "_individual_betas_"
                    ).replace(".nii.gz", ".csv")
                    save_tabular_data(
                        beta_coefficient_df,
                        output_dir,
                        output_csv_name,
                        first_level_glt_label,
                        beta_name,
                        add_condition_entity_key,
                        save_excel_version,
                    )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
