"""
Better to use git submodule/lfs or fetch via OSF or AWS; however, this increases technical complexity
which defeats the purpose of the app and there may be firewall issues associated with downloading.
"""

from pathlib import Path

import nibabel as nib, numpy as np
from nilearn.masking import _unmask_3d
from nilearn.maskers import nifti_spheres_masker
from nilearn.plotting import plot_roi
from matplotlib.colors import ListedColormap

from bidsaid.files import get_entity_value
from bidsaid.parsers import _is_float

COHORT_MAP = {
    "kids": {
        "template_mask_path": Path(__file__).parent
        / "templates/kids/tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz",
        "template_img_path": Path(__file__).parent
        / "templates/kids/tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz",
    },
    "adults": {
        "template_mask_path": Path(__file__).parent
        / "templates/adults/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
        "template_img_path": Path(__file__).parent
        / "templates/adults/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
    },
}


def run_pipeline(
    dst_dir,
    cohort,
    mni_coordinate,
    sphere_radius,
):
    if not all([_is_float(x) for x in mni_coordinate]):
        mni_coordinate = list(map(str, mni_coordinate))
        raise ValueError(f"Invalid MNI coordinate: {','.join(mni_coordinate)}")

    mni_coordinate = list(map(float, mni_coordinate))
    template_mask_path = COHORT_MAP[cohort]["template_mask_path"]
    template_img_path = COHORT_MAP[cohort]["template_img_path"]

    template_mask = nib.load(template_mask_path)
    # https://neurostars.org/t/create-a-10mm-sphere-roi-mask-around-a-given-coordinate/28853/3
    _, A = nifti_spheres_masker.apply_mask_and_get_affinity(
        seeds=[tuple(mni_coordinate)],
        niimg=None,
        radius=sphere_radius,
        allow_overlap=False,
        mask_img=template_mask,
    )

    sphere_mask = _unmask_3d(
        X=A.toarray().flatten(), mask=template_mask.get_fdata().astype(bool)
    ).astype(np.int8)

    hdr = template_mask.header.copy()
    hdr.set_data_dtype(np.int8)

    sphere_mask = nib.nifti1.Nifti1Image(sphere_mask, template_mask.affine, hdr)

    coord_name = "_".join([str(x) for x in mni_coordinate])
    tpl = get_entity_value(template_mask_path, "tpl", return_entity_prefix=True)
    res = get_entity_value(template_mask_path, "res", return_entity_prefix=True)
    sphere_name = f"{tpl}_{res}_desc-sphere_mask_{coord_name}.nii.gz"
    sphere_filename = (Path(dst_dir) if dst_dir else Path().home()) / sphere_name
    sphere_filename.parent.mkdir(parents=True, exist_ok=True)

    nib.save(sphere_mask, sphere_filename)

    display = plot_roi(
        sphere_filename,
        bg_img=template_img_path,
        draw_cross=False,
        cmap=ListedColormap(["red"]),
        colorbar=False,
    )

    plot_filename = sphere_filename.parent / sphere_filename.name.replace(
        ".nii.gz", ".png"
    )
    display.savefig(plot_filename, dpi=720)

    return sphere_filename, plot_filename
