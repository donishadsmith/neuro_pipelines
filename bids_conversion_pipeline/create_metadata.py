import copy, json
from pathlib import Path

import nifti2bids.metadata as bids_meta
from nifti2bids.io import regex_glob
from nifti2bids.bids import get_entity_value

_BASE_JSON = {
    "Modality": "MR",
    "MagneticFieldStrength": 3,
    "Manufacturer": "Philips",
    "ManufacturersModelName": "Ingenia Elition X 5.7.1",
    "InstitutionName": "Johns Hopkins University",
}

_ANAT_JSON = copy.deepcopy(_BASE_JSON)
_ANAT_JSON.update(
    {
        "MRAcquisitionType": "3D",
        "SliceThickness": 1,
        "SpacingBetweenSlices": 0,
        "EchoTime": 0.00367,
    }
)

_FUNC_JSON = copy.deepcopy(_BASE_JSON)
_FUNC_JSON.update(
    {
        "InternalPulseSequenceName": "EPI",
        "MRAcquisitionType": "2D",
        "SliceThickness": 3,
        "SpacingBetweenSlices": 1.12,
        "EchoTime": 0.03,
        "EffectiveEchoSpacing": None,
        "TotalReadoutTime": None,
        "PhaseEncodingAxis": "j",
        "PhaseEncodingDirection": "j-",
        "RepetitionTime": None,
        "SliceTiming": None,
        "TaskName": None,
    }
)

WATER_FAT_SHIFT_PIXELS = 7.174
EPI_FACTOR = 27


def _create_json_sidecar_pipeline(bids_dir: Path) -> None:
    nifti_files = regex_glob(bids_dir, pattern=r"^.*\.(nii.gz)$", recursive=True)
    for nifti_file in nifti_files:
        modality = nifti_file.parent.name
        if modality == "anat":
            json_schema = _ANAT_JSON
        else:
            json_schema = copy.deepcopy(_FUNC_JSON)
            json_schema["EffectiveEchoSpacing"] = (
                bids_meta.compute_effective_echo_spacing(
                    WATER_FAT_SHIFT_PIXELS, EPI_FACTOR
                )
            )
            json_schema["TotalReadoutTime"] = bids_meta.compute_total_readout_time(
                json_schema["EffectiveEchoSpacing"],
                recon_matrix_pe=bids_meta.get_recon_matrix_pe(
                    nifti_file, phase_encoding_axis="j"
                ),
            )
            json_schema["RepetitionTime"] = bids_meta.get_tr(nifti_file)
            json_schema["SliceTiming"] = bids_meta.create_slice_timing(
                nifti_file,
                slice_acquisition_method="sequential",
                ascending=True,
                slice_axis="k",
            )
            json_schema["TaskName"] = get_entity_value(nifti_file, entity="task")

        json_filename = str(nifti_file).replace(".nii.gz", ".json")
        with open(json_filename, "w") as f:
            json.dump(json_schema, f, indent=2)
