import argparse, itertools, subprocess
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import threshold_img
from nilearn.reporting import get_clusters_table
from nilearn.plotting import plot_stat_map
from scipy.stats import norm

from nifti2bids.logging import setup_logger

from _utils import get_task_contrasts

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply NN1 2-sided cluster correction to data "
            "and identify significant clusters."
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
        required=True,
        help="Path to Singularity image of Afni with R.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--voxel_correction_p",
        dest="voxel_correction_p",
        required=False,
        default=0.001,
        help="P-value for voxel correction.",
    )
    parser.add_argument(
        "--cluster_correction_p",
        dest="cluster_correction_p",
        required=False,
        default=0.05,
        help="P-value for cluster correction.",
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


def get_zscore_map_and_mask(analysis_dir, afni_img_path, task, contrast, glt_code):
    stats_filename = next(
        analysis_dir.rglob(f"task-{task}_contrast-{contrast}_desc-stats.nii.gz")
    )
    zcore_map_filename = str(stats_filename).replace(
        "_desc-stats", f"_gltcode-{glt_code}_desc-z_map"
    )

    cmd = (
        f"singularity exec -B /projects:/projects {afni_img_path} 3dbucket "
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
        analysis_dir.rglob(f"task-{task}_contrast-{contrast}_desc-group_mask.nii.gz")
    )

    return Path(zcore_map_filename), group_mask_filename


def get_cluster_correction_table(analysis_dir, task, contrast):
    cluster_correction_filename = next(
        analysis_dir.rglob(
            f"task-{task}_contrast-{contrast}_desc-cluster_correction.NN1_bisided.1D"
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
    clust_p_indx = np.where(cluster_p_values_arr == cluster_correction_p)[0][0] + 1
    cluster_p_str = cluster_correction_table.columns[clust_p_indx]

    return int(
        np.ceil(
            cluster_correction_table.loc[
                cluster_correction_table["pthr"] == voxel_correction_p, cluster_p_str
            ].to_numpy(copy=True)[0]
        )
    )


def identify_clusters(
    analysis_dir,
    thresholded_img,
    voxel_correction_p,
    cluster_size,
    task,
    contrast,
    glt_code,
):
    clusters_table, labels_map_list = get_clusters_table(
        thresholded_img,
        stat_threshold=p_to_z(voxel_correction_p),
        cluster_threshold=cluster_size,
        two_sided=True,
        return_label_maps=True,
    )

    cluster_table_filename = (
        analysis_dir
        / "cluster_results"
        / f"task-{task}_contrast-{contrast}_gltcode-{glt_code}_desc-cluster_results.csv"
    )
    cluster_table_filename.parent.mkdir(parents=True, exist_ok=True)
    if not clusters_table.empty:
        # Add interpretation
        first_label, second_label = glt_code.split("_vs_")

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
        # Make no assumption about labels_map order,
        # one map is positive, the other is negative
        label_base_dir = analysis_dir / "cluster_masks"
        label_base_dir.mkdir(parents=True, exist_ok=True)
        for label_map in labels_map_list:
            # Note: returns sorted array
            label_ids = np.unique(label_map.get_fdata()[label_map.get_fdata() != 0])
            if label_ids.shape[0] == 0:
                continue

            # Label ids may not correspond to cluster ids in the table
            label_map_fdata = label_map.get_fdata()
            thresholded_img_fdata = thresholded_img.get_fdata()

            voxel_data = thresholded_img_fdata[label_map_fdata != 0].ravel()
            voxel_stats = voxel_data[voxel_data != 0]
            peak_voxel_val = voxel_stats[np.argmax(np.abs(voxel_stats))]
            tail = "positive" if peak_voxel_val > 0 else "negative"

            peak_stats = clusters_table["Peak Stat"].to_numpy(copy=True)
            if tail == "positive":
                peak_stats_indices = clusters_table.loc[
                    np.where(peak_stats > 0), "Peak Stat"
                ].index.tolist()
            else:
                peak_stats_indices = clusters_table.loc[
                    np.where(peak_stats < 0), "Peak Stat"
                ].index.tolist()

            cluster_ids = clusters_table.loc[peak_stats_indices, "Cluster ID"].to_numpy(
                copy=True
            )
            # The cluster IDs are integers or an integer with a letter, if changed possibly use
            # not any(e.isalpha() for e in str(label)
            cluster_ids = [label for label in cluster_ids if label.isdigit()]
            # In source code, ids in label map go from 1 to N with 1 being the highest peak
            # DataFrame is sorted in highest abolute peak for positive and negative tail
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
                    f"task-{task}_contrast-{contrast}_gltcode-{glt_code}_clusterid-{cluster_id}"
                    f"_tail-{tail}_desc-cluster_mask.nii.gz"
                )

                nib.save(label_mask_img, label_mask_filename)

    return cluster_table_filename


def plot_thresholded_img(
    analysis_dir, thresholded_img, template_img_path, task, contrast, glt_code
):
    kwargs = {"stat_map_img": thresholded_img}
    if template_img_path:
        bg_img = nib.load(template_img_path)
        kwargs.update({"bg_img": bg_img})

    if contrast not in ["seen", "indoor"]:
        contrast_name = contrast.replace("_vs_", " > ")
    else:
        contrast_name = contrast

    group_names = [f"{dose} mg MPH" for dose in glt_code.split("_vs_")]
    first_group, second_group = group_names
    title = (
        f"Task: {task} Contrast: {contrast_name} Group: {first_group} > {second_group}"
    )

    for mode in ["ortho", "x", "y", "z"]:
        display = plot_stat_map(**kwargs, display_mode=mode)

        display.title(title, bgcolor="black", color="white", size=10)

        plot_filename = (
            analysis_dir
            / "stat_plots"
            / f"task-{task}_contrast-{contrast}_gltcode-{glt_code}_displaymode-{mode}_desc-plot.png"
        )
        plot_filename.parent.mkdir(parents=True, exist_ok=True)

        display.savefig(plot_filename, dpi=720)


def main(
    analysis_dir,
    afni_img_path,
    task,
    voxel_correction_p,
    cluster_correction_p,
    template_img_path,
):
    analysis_dir = Path(analysis_dir)

    LGR.info(f"TASK: {task}")

    glt_codes = ["5_vs_0", "10_vs_0", "10_vs_5"]
    contrasts = get_task_contrasts(task, caller="get_cluster_results")
    contrasts_glts_list = list(itertools.product(contrasts, glt_codes))
    for contrast, glt_code in contrasts_glts_list:
        LGR.info(f"CONTRAST: {contrast}, GLTCODE: {glt_code}")
        zcore_map_filename, group_mask_filename = get_zscore_map_and_mask(
            analysis_dir, afni_img_path, task, contrast, glt_code
        )
        if not zcore_map_filename.exists():
            LGR.warning(
                f"Skipping the following glt code due to file not existing: {glt_code}"
            )
            continue

        cluster_correction_table = get_cluster_correction_table(
            analysis_dir, task, contrast
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
            "-z_map", "-cluster_corrected"
        )
        LGR.info(f"Saving thresholded image to: {thresholded_filename}")
        nib.save(thresholded_img, thresholded_filename)

        cluster_table_filename = identify_clusters(
            analysis_dir,
            thresholded_img,
            voxel_correction_p,
            cluster_size,
            task,
            contrast,
            glt_code,
        )

        base_str = f"TASK: {task}, CONTRAST: {contrast} GLTCODE: {glt_code}"
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
            contrast,
            glt_code,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
