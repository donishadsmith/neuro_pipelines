import argparse, itertools, subprocess, sys
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import threshold_img
from nilearn.reporting import get_clusters_table
from nilearn.plotting import plot_stat_map
from scipy.stats import norm

from nifti2bids.logging import setup_logger

from _utils import get_contrast_entity_key, get_first_level_gltsym_codes

LGR = setup_logger(__name__)

# For nonparametric approach which is already thresholded
ZERO_STAT_THRESHOLD = 0
ZERO_CLUSTER_SIZE = 0

GLT_CODES = ("5_vs_0", "10_vs_0", "10_vs_5")


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=(
            "If parametric, apply NN1 2-sided cluster correction to data "
            "and identify significant clusters. If nonparametric, just identifies significant clusters."
        )
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Root path to directory containing second level stats "
            "and cluster correction tables."
        ),
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=False,
        default=None,
        help="Path to Apptainer image of Afni with R.",
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
        "--connectivity",
        dest="connectivity",
        required=False,
        default="NN1",
        help="Connectivity to use for parametric. Will always use 2-sided/bisided version.",
    )
    parser.add_argument(
        "--voxel_correction_p",
        dest="voxel_correction_p",
        required=False,
        default=0.001,
        help=("P-value for voxel correction. Only used for the parametric approach."),
    )
    parser.add_argument(
        "--cluster_correction_p",
        dest="cluster_correction_p",
        required=False,
        default=0.05,
        help="P-value for cluster correction. Only used for the parametric approach.",
    )
    parser.add_argument(
        "--template_img_path",
        dest="template_img_path",
        required=False,
        default=None,
        help="Path to a template image to use for plotting.",
    )

    return parser


def p_to_z(p_value, two_sided=True):
    return norm.ppf(1 - p_value / (2 if two_sided else 1))


def get_zscore_map_and_mask(
    analysis_dir, afni_img_path, task, entity_key, first_level_gltlabel, glt_code
):
    stats_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-parametric_stats.nii.gz"
        )
    )
    zcore_map_filename = str(stats_filename).replace(
        "_desc-parametric_stats", f"_gltcode-{glt_code}_desc-parametric_z_map"
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
        f"{stats_filename}'[{glt_code} Z]' "
        f"-prefix {zcore_map_filename} "
        "-overwrite"
    )
    LGR.info(f"Extracting {glt_code} z score map: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception:
        LGR.critical(f"The following command failed: {cmd}", exc_info=True)

    group_mask_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-group_mask.nii.gz"
        )
    )

    return Path(zcore_map_filename), group_mask_filename


def get_cluster_correction_table(
    analysis_dir, task, entity_key, first_level_gltlabel, connectivity
):
    cluster_correction_filename = next(
        analysis_dir.rglob(
            f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-cluster_correction.{connectivity}_bisided.1D"
        )
    )
    cluster_correction_table = pd.DataFrame(
        np.loadtxt(cluster_correction_filename),
        columns=["pthr", ".10000", ".05000", ".02000", ".01000"],
    )

    return cluster_correction_table.astype(float)


def get_cluster_size(
    cluster_correction_table, voxel_correction_p, cluster_correction_p
):
    cluster_p_values = list(map(float, cluster_correction_table.columns[1:]))
    cluster_p_values_arr = np.array(cluster_p_values)
    clust_p_index = (
        np.where((cluster_p_values_arr == cluster_correction_p) == True)[0][0] + 1
    )
    cluster_p_str = cluster_correction_table.columns[clust_p_index]

    return int(
        np.ceil(
            cluster_correction_table.loc[
                cluster_correction_table["pthr"] == voxel_correction_p,
                cluster_p_str,
            ].to_numpy(copy=True)[0]
        )
    )


def get_interpretation_labels(glt_code):
    first_label, second_label = glt_code.split("_vs_")

    return first_label, second_label


def identify_clusters(
    analysis_dir,
    thresholded_img,
    method,
    stat_threshold,
    cluster_size,
    task,
    entity_key,
    first_level_gltlabel,
    glt_code,
):
    clusters_table, labels_map_list = get_clusters_table(
        thresholded_img,
        stat_threshold=(
            p_to_z(stat_threshold) if method == "parametric" else stat_threshold
        ),
        cluster_threshold=cluster_size,
        two_sided=True,
        return_label_maps=True,
    )

    cluster_table_filename = (
        analysis_dir
        / "cluster_results"
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{glt_code}_desc-{method}_cluster_results.csv"
    )
    cluster_table_filename.parent.mkdir(parents=True, exist_ok=True)

    if not clusters_table.empty:
        n_rows = clusters_table.shape[0]
        clusters_table["Statistic Name"] = (
            ["Z-score"] * n_rows if method == "parametric" else ["T-score"] * n_rows
        )
        first_label, second_label = get_interpretation_labels(glt_code)

        clusters_table["Cluster ID"] = clusters_table["Cluster ID"].astype(str)

        peaks = clusters_table["Peak Stat"].astype(float).to_numpy(copy=True)
        mask_primary = clusters_table["Cluster ID"].str.isdigit()
        mask_pos = mask_primary & (peaks > 0)
        mask_neg = mask_primary & (peaks < 0)

        clusters_table.loc[mask_pos, "Interpretation"] = (
            f"{first_label} mg MPH > {second_label} mg MPH"
        )
        clusters_table.loc[mask_neg, "Interpretation"] = (
            f"{second_label} mg MPH > {first_label} mg MPH"
        )

        clusters_table.to_csv(cluster_table_filename, sep=",", index=False)

        # Get label map
        label_base_dir = analysis_dir / "cluster_masks"
        label_base_dir.mkdir(parents=True, exist_ok=True)
        for label_map in labels_map_list:
            label_ids = np.unique(label_map.get_fdata()[label_map.get_fdata() != 0])
            if label_ids.shape[0] == 0:
                continue

            label_map_fdata = label_map.get_fdata()
            thresholded_img_fdata = thresholded_img.get_fdata()

            voxel_data = thresholded_img_fdata[label_map_fdata != 0].ravel()
            voxel_stats = voxel_data[voxel_data != 0]
            peak_voxel_val = voxel_stats[np.argmax(np.abs(voxel_stats))]
            tail = "positive" if peak_voxel_val > 0 else "negative"

            peak_stats = clusters_table["Peak Stat"].to_numpy(copy=True)
            if tail == "positive":
                peak_stats_indices = clusters_table.loc[
                    peak_stats > 0, "Peak Stat"
                ].index.tolist()
            else:
                peak_stats_indices = clusters_table.loc[
                    peak_stats < 0, "Peak Stat"
                ].index.tolist()

            cluster_ids = clusters_table.loc[peak_stats_indices, "Cluster ID"].to_numpy(
                copy=True
            )
            cluster_ids = [label for label in cluster_ids if label.isdigit()]
            cluster_id_map = {
                index: label for index, label in enumerate(cluster_ids, start=1)
            }

            for label_id in label_ids:
                cluster_id = cluster_id_map[int(label_id)]

                label_mask_fdata = np.zeros_like(label_map.get_fdata())
                label_mask_fdata[label_map.get_fdata() == label_id] = 1

                label_mask_img = nib.nifti1.Nifti1Image(
                    label_mask_fdata, label_map.affine, label_map.header
                )
                label_mask_filename = label_base_dir / (
                    f"task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{glt_code}_clusterid-{cluster_id}"
                    f"_tail-{tail}_desc-{method}_cluster_mask.nii.gz"
                )

                nib.save(label_mask_img, label_mask_filename)

    return cluster_table_filename


def plot_thresholded_img(
    analysis_dir,
    thresholded_img,
    template_img_path,
    task,
    entity_key,
    first_level_gltlabel,
    glt_code,
    method,
):
    kwargs = {"stat_map_img": thresholded_img}
    if template_img_path:
        bg_img = nib.load(template_img_path)
        kwargs.update({"bg_img": bg_img})

    if first_level_gltlabel not in ["seen", "indoor"]:
        first_level_code = first_level_gltlabel.replace("_vs_", " > ")
    else:
        first_level_code = first_level_gltlabel

    first_label, second_label = get_interpretation_labels(glt_code)
    first_group = f"{first_label} mg MPH"
    second_group = f"{second_label} mg MPH"
    title = f"TASK: {task} FIRST LEVEL GLTLABEL: {first_level_code} GROUP CONTRAST: {first_group} > {second_group}"

    for mode in ["ortho", "x", "y", "z"]:
        display = plot_stat_map(**kwargs, display_mode=mode)

        display.title(title, bgcolor="black", color="white", size=10)
        statistic = "Z-score" if method == "parametric" else "T-score"
        display._cbar.set_label(f"{statistic} Intensity")

        plot_filename = (
            analysis_dir
            / "stat_plots"
            / f"task-{task}_{entity_key}-{first_level_code}_gltcode-{glt_code}_displaymode-{mode}_desc-{method}_cluster_plot.png"
        )
        plot_filename.parent.mkdir(parents=True, exist_ok=True)

        display.savefig(plot_filename, dpi=720)

        display.close()


def main(
    analysis_dir,
    afni_img_path,
    task,
    analysis_type,
    method,
    connectivity,
    voxel_correction_p,
    cluster_correction_p,
    template_img_path,
):
    analysis_dir = Path(analysis_dir)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    first_level_gltlabels = get_first_level_gltsym_codes(
        task, analysis_type, caller="get_cluster_results"
    )
    first_level_gltlabel_list = list(
        itertools.product(first_level_gltlabels, GLT_CODES)
    )

    for first_level_gltlabel, glt_code in first_level_gltlabel_list:
        entity_key = get_contrast_entity_key(first_level_gltlabel)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_gltlabel}, GLTCODE: {glt_code}")

        if method == "parametric":
            if not afni_img_path:
                LGR.critical("afni_img_path is required when method is parametric.")
                sys.exit(1)

            zcore_map_filename, group_mask_filename = get_zscore_map_and_mask(
                analysis_dir,
                afni_img_path,
                task,
                entity_key,
                first_level_gltlabel,
                glt_code,
            )
            if not zcore_map_filename.exists():
                LGR.warning(
                    f"Skipping the following glt code due to file not existing: {glt_code}"
                )
                continue

            cluster_correction_table = get_cluster_correction_table(
                analysis_dir, task, entity_key, first_level_gltlabel, connectivity
            )
            cluster_size = get_cluster_size(
                cluster_correction_table, voxel_correction_p, cluster_correction_p
            )

            thresholded_img = threshold_img(
                nib.load(zcore_map_filename),
                mask_img=nib.load(group_mask_filename),
                threshold=p_to_z(voxel_correction_p),
                cluster_threshold=cluster_size,
            )
            thresholded_filename = str(zcore_map_filename).replace(
                "-z_map", "-parametric_cluster_corrected"
            )
            LGR.info(f"Saving thresholded image to: {thresholded_filename}")
            nib.save(thresholded_img, thresholded_filename)
        else:
            try:
                thresholded_filename = next(
                    analysis_dir.rglob(
                        f"task-{task}_{entity_key}-{first_level_gltlabel}_gltcode-{glt_code}_desc-nonparametric_thresholded_bisided.nii.gz"
                    )
                )
            except Exception:
                LGR.critical(f"Skipping {glt_code}: no thresholded file found")
                continue

            thresholded_img = nib.load(thresholded_filename)

        cluster_table_filename = identify_clusters(
            analysis_dir,
            thresholded_img,
            method,
            ZERO_STAT_THRESHOLD,
            ZERO_CLUSTER_SIZE,
            task,
            entity_key,
            first_level_gltlabel,
            glt_code,
        )

        base_str = f"TASK: {task}, FIRST LEVEL GLTLABEL: {first_level_gltlabel} GLTCODE: {glt_code}"
        if not cluster_table_filename.exists():
            LGR.info(f"No significant clusters for {base_str}")
            continue
        else:
            LGR.info(f"Significant clusters found for {base_str}")

        plot_thresholded_img(
            analysis_dir,
            thresholded_img,
            template_img_path,
            task,
            entity_key,
            first_level_gltlabel,
            glt_code,
            method,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
