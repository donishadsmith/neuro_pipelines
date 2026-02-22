import copy, json
from pathlib import Path

import nifti2bids.metadata as bids_meta
from nifti2bids.io import regex_glob
from nifti2bids.bids import get_entity_value

# https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html
# Note: Metadata obtained from the Philips Exam Cards
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
        "PhaseEncodingDirection": None,
        "RepetitionTime": None,
        "SliceEncodingDirection": None,
        "SliceTiming": None,
        "TaskName": None,
    }
)


SLICE_ENCODING_END = "S"
FAT_SHIFT_DIRECTION = "P"
WATER_FAT_SHIFT_PIXELS = 7.174
EPI_FACTOR = 27


def _create_json_sidecar_pipeline(bids_dir: Path) -> None:
    nifti_files = regex_glob(bids_dir, pattern=r"^.*\.nii\.gz$", recursive=True)
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

            # There is a mix of orientation in the data, mostly LAS and LPS for
            # functional images
            phase_axis, phase_index = bids_meta.direction_to_voxel_axis(
                nifti_file, anatomical_directions=["A", "P"]
            )

            json_schema["TotalReadoutTime"] = bids_meta.compute_total_readout_time(
                json_schema["EffectiveEchoSpacing"],
                recon_matrix_pe=bids_meta.get_recon_matrix_pe(
                    nifti_file, phase_encoding_axis=phase_axis
                ),
            )
            json_schema["RepetitionTime"] = bids_meta.get_tr(nifti_file)

            slice_axis, slice_index = bids_meta.direction_to_voxel_axis(
                nifti_file, anatomical_directions=["I", "S"]
            )

            json_schema["SliceTiming"] = bids_meta.create_slice_timing(
                nifti_file,
                slice_acquisition_method="sequential",
                ascending=True,
                slice_axis=slice_axis,
            )

            _, orientation = bids_meta.get_image_orientation(nifti_file)
            # The slice orientation is in the transverse plane and the transverse
            # slice order is from feet to head and the slice scan order is ascending.
            # The slice direction goes from Inferior -> Superior with this configuration
            # RAS and LPS are Inferior -> Superior so direction is k. If LPI direction is k-
            # since it goes from Superior -> Inferior.
            json_schema["SliceEncodingDirection"] = (
                slice_axis
                if orientation[slice_index] == SLICE_ENCODING_END
                else f"{slice_axis}-"
            )
            json_schema["TaskName"] = get_entity_value(nifti_file, entity="task")

            # https://neurostars.org/t/determining-phase-encoding-direction-and-total-read-out-time-from-philips-scans/25402/4
            # https://neurostars.org/t/bids-fmap-phase-encoding-direction-and-image-orientation-beginner/33274/7
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC4845159/
            # When the fat shift direction is A, susceptibility artifacts are shifted posteriorly
            # When fat shift direction is P, susceptibility artifacts are shifted anteriorly
            # Artifacts shift in opposite direction of the fat
            # Philips fat direction is "P", phase encoding is on the "A-P" axis
            # Note from 0 to N is P -> A for RAS and A -> P for LPS
            # Hence fat shift direction of A (phase encoding from A to P) is j- for RAS and j for LPS
            # and fat shift direction of P (phase encoding from P to A) is j for RAS and j- for LPS
            # SENSE=3 collects 1/3 of the data in k-space in the encoding direction which reduces sdc
            # Distortion in these images are minimal https://cds.ismrm.org/protected/07MProceedings/PDFfiles/01500.pdf
            json_schema["PhaseEncodingDirection"] = (
                phase_axis
                if orientation[phase_index] != FAT_SHIFT_DIRECTION
                else f"{phase_axis}-"
            )

        json_filename = str(nifti_file).replace(".nii.gz", ".json")
        with open(json_filename, "w") as f:
            json.dump(json_schema, f, indent=2)
