"""Extract data from Conners 4."""

import glob, os, re, shutil
from typing import Literal, Optional
from datetime import datetime

import pandas as pd
from pypdf import PdfReader  # 6.1.1 - 6.1.3

CSV_COLUMN_NAMES = [
    "SN",
    "Visit",
    "Snvisit",
    "INA/EDF T score",
    "INA/EDF %",
    "HYP T score",
    "HYP %",
    "IMP T score",
    "IMP %",
    "EMDYS T score",
    "EMDYS %",
    "DEP T score",
    "DEP %",
    "ANX T score",
    "ANX %",
    "SCHOOL T score",
    "SCHOOL %",
    "PEER T score",
    "PEER %",
    "FAMILY T score",
    "FAMILY %",
    "ADHD-I T score",
    "ADHD-I %",
    "ADHD-HI T score",
    "ADHD-HI %",
    "ADHD-TOT T score",
    "ADHD-TOT %",
    "ODD T score",
    "ODD %",
    "CD T score",
    "CD %",
    "Prob score %",
]
UNIQUE_DATA_FIELD_NAMES = [
    "Dysfunction",
    "Hyperactivity",
    "Impulsivity",
    "Emotional Dysregulation",
    "Depressed Mood",
    "Anxious Thoughts",
    "Schoolwork",
    "Peer Interactions",
    "Family Life",
    "Symptoms",
    "Total ADHD Symptoms",
    "Disorder Symptoms",
    "ADHD Index",
    "ADHD Inattentive",
    "ADHD Hyperactive/Impulsive",
    "Conduct Disorder",
]


def get_files(target_dir: str, ext: str) -> list[str]:
    """Gets files with a specific extension."""
    return glob.glob(os.path.join(target_dir, f"*.{ext}"))


def convert_list_to_dict(input_list: list[str]) -> dict[str, None]:
    "Converts a list to a dictionary where each key is mapped to ``None``."
    return {key: None for key in input_list}


def extract_pdf_text(pdf_file: str, page_number: int) -> list[str]:
    """Extract text from a single page of a PDF and removes lines with only whitespace."""
    reader = PdfReader(pdf_file)
    single_page = reader.pages[page_number].extract_text()
    stripped_page_list = [
        line.strip(" ")
        for line in [line for line in single_page.splitlines() if not line.isspace()]
    ]

    return stripped_page_list


def create_dirs(dir_name: str) -> None:
    """Checks if directory exists and creates it if it does not exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def determine_rater(single_page_list: list[str]) -> Literal["parent", "child"]:
    """
    Determines if the rater is a child or parent.

    Uses the first page (index 0) of the Conners 4 PDF
    to determine if the rater is a child or parent.
    For Conners 4, if the rater is a parent, there
    is a "Parent's/Guardian's" field that is not
    present when the child is the rater.

    Parameters
    ----------
    single_page_list: :obj:`list[str]`
        A single PDF page represented as a list where each element
        is a line of of text as a string.

    Returns
    -------
    Literal["parent", "child"]:
        A string denoting if the rater is a parent or child
    """
    has_parent_guardian_str = any(
        [line.startswith("Parent") for line in single_page_list]
    )

    return "parent" if has_parent_guardian_str else "child"


def standardize_date(date: str, date_format: str = "%B %d, %Y") -> str:
    "Standardizes the administration date."
    return datetime.strptime(date, date_format).date()


def standardize_pdf_filenames(pdf_dir: str) -> None:
    """
    Cleans PDF filenames.

    Standardizes the PDF filename from a random assortment of letters to
    sub-{sub_id}_date-{administation_date}_rater-{file_id}.
    Outputs these filenames in a subdirectory named "reformatted_filenames"
    under the directory specified by ``pdf_dir``. Dynamically creates additional
    "child" and "parent" subdirectories under "reformatted_filenames" if any PDF is
    detected as the rater being the child or parent.

    Parameters
    ----------
    pdf_dir: :obj:`str`
        Directory containing the unformatted Conners 4 PDF files

    Returns
    -------
    None
    """
    target_dir = os.path.join(pdf_dir, "reformatted_filenames")
    create_dirs(target_dir)

    pdf_files = get_files(pdf_dir, "pdf")
    for pdf_file in pdf_files:
        stripped_page_list = extract_pdf_text(pdf_file, 0)
        rater = determine_rater(stripped_page_list)

        sub_id = [line for line in stripped_page_list if line.startswith("Name/ID")][0]
        sub_id = sub_id.split(" ")[1]

        administration_date = [
            line
            for line in stripped_page_list
            if line.startswith("Administration Date")
        ]
        administration_date = (
            administration_date[0].removeprefix("Administration Date:").strip(" ")
        )
        administration_date = standardize_date(administration_date)

        new_filename = f"sub-{sub_id}_date-{administration_date}_rater-{rater}"
        output_dir = os.path.join(target_dir, rater)
        create_dirs(output_dir)

        base_pdf_name = os.path.basename(pdf_file)
        shutil.copyfile(pdf_file, os.path.join(output_dir, base_pdf_name))
        os.rename(
            os.path.join(output_dir, base_pdf_name),
            os.path.join(output_dir, f"{new_filename}.pdf"),
        )


def get_subject_ids(
    reformatted_pdf_files: list[str], subjects: Optional[list[str]]
) -> list[str]:
    """
    Gets subject IDs from the reformatted pdf files.

    Parameters
    ----------
    reformatted_pdf_files: :obj:`list[str]`
        Absolute path of PDF files with reformatted names in the
        form "sub-[subject_id]_date-[administration_date]_rater-[rater].pdf"

    Returns
    -------
        list[str]
            List of subject IDs.
    """
    subjects = subjects or []
    all_subjects_list = [
        re.search(r"sub-(\d+?)_", os.path.basename(file))[0]
        .removeprefix("sub-")
        .removesuffix("_")
        for file in reformatted_pdf_files
    ]

    return [subject for subject in all_subjects_list if subject in subjects]


def get_sessions(subject_id_list: list[str]) -> list[str]:
    """
    Returns a list of the sessions in the form "v1", "v2",
    "v3", etc for each subject id.
    """
    session_list = []
    unique_ids = list(dict.fromkeys(subject_id_list).keys())
    for unique_id in unique_ids:
        visit_num_list = [val + 1 for val in range(subject_id_list.count(unique_id))]
        session_list.extend([f"v{val}" for val in visit_num_list])

    return session_list


def get_sn_visit(subject_id_list: list[str], visit_list: list[str]) -> list[str]:
    """
    Gets the subject id and visit. Assumes ``subject_id_list`` and
    ``visit_list`` are the same length.
    """
    sn_visit_list = []
    subject_session_list = list(zip(subject_id_list, visit_list))
    for subject, session in subject_session_list:
        sn_visit_list.append(f"{subject}{session}")

    return sn_visit_list


def create_unique_column_dict(columns_names_list: list) -> dict[str, None]:
    """
    Converts a list of column names to dictionary containing the unique
    starting names. For instance, "ANX T score" and "ANX %" are reduced
    to "ANX". This is done to later map these unique name to the
    Conners data field that contains both of these values
    (when converted to text, each line of Conners data field contains raw score,
    t score, percentile, etc).
    """
    unique_column_names_dict = {}
    for name in columns_names_list:
        reduced_name = name.split(" ")[0]
        if reduced_name not in unique_column_names_dict:
            unique_column_names_dict[reduced_name] = None

    return unique_column_names_dict


def create_column_names_dict(columns_names_list: list) -> dict[str, list]:
    """
    Converts a list of column names to dictionary of each name mapped to
    an empty list.
    """
    column_names_dict = {}
    for name in columns_names_list:
        column_names_dict[name] = []

    return column_names_dict


def replace_data_field_names(single_page_list: list[str]) -> list[str]:
    """
    Replaces lines starting with "Symptoms" with their respective data field.

    Conners 4 scores are on the fourth page of the PDF (index 3). When converted
    to text, "ADHD Inattentive", "ADHD Hyperactive/Impulsive", and "Conduct Disorder"
    are isolated onto their own lines with the subsequent line starting with
    "Symptoms" and containing the score information.

    Parameters
    ----------
    single_page_list: :obj:`list[str]`
        A single PDF page represented as a list where each element is a sentence.

    Returns
    -------
    list[str]:
        A single PDF page represented as list where each element is a sentence.
        If ``single_page_list`` was page 4 of Conners, then the lines starting with
        "Symptoms" are replaced with their proper data field name (
        "ADHD Inattentive", "ADHD Hyperactive/Impulsive", or "Conduct Disorder").
    """
    single_page_list = [
        line
        for line in single_page_list
        if any(line.startswith(x) for x in UNIQUE_DATA_FIELD_NAMES)
    ]

    target_field_names = [
        "ADHD Inattentive",
        "ADHD Hyperactive/Impulsive",
        "Conduct Disorder",
    ]
    remove_indxs_list = []
    for indx, line in enumerate(single_page_list):
        field_name = line.strip(" ")
        if field_name in target_field_names:
            if single_page_list[indx + 1].startswith("Symptoms"):
                single_page_list[indx + 1] = single_page_list[indx + 1].replace(
                    "Symptoms", field_name
                )
                remove_indxs_list.append(indx)

    return [
        line
        for indx, line in enumerate(single_page_list)
        if indx not in remove_indxs_list
    ]


def filter_list(single_page_list: list[str], data_field_name: str) -> str | None:
    """
    Filters ``single_page_list`` to only return the line starting with a specific
    data field name.
    """
    filtered_list = [
        line for line in single_page_list if line.startswith(data_field_name)
    ]

    return filtered_list if filtered_list else None


def remove_prefix(
    input_str: str, prefix: str, remove_left_whitespace: bool = True
) -> str:
    """Removes prefix from a string."""
    filtered_string = input_str.removeprefix(f"{prefix}")

    return filtered_string.lstrip(" ") if remove_left_whitespace else filtered_string


def remove_nondigits(input_str: str) -> str:
    """Remove all non-digits in a string"""
    return re.sub(r"\D", "", input_str)


def get_score(filtered_str: str, score_type: str) -> float:
    """Get Conners score from filtered string."""
    score = filtered_str[3] if score_type == "%" else filtered_str[1]
    score = remove_nondigits(score)

    return float(score)


def extract_conners_datafields(
    reformatted_pdf_files: str, conners_data_fields_dict: dict[str, str]
) -> dict[str, list[float]]:
    """Extracts the ADHD data from the pdf files."""
    column_names_dict = create_column_names_dict(CSV_COLUMN_NAMES)

    for pdf_file in reformatted_pdf_files:
        stripped_text_list = replace_data_field_names(
            extract_pdf_text(pdf_file=pdf_file, page_number=3)
        )

        for key in conners_data_fields_dict.keys():
            data_field_name = conners_data_fields_dict[key]
            if not (
                filtered_text_list := filter_list(stripped_text_list, data_field_name)
            ):
                data_field_name = f"{data_field_name.rstrip(' ')}**"
                filtered_text_list = filter_list(stripped_text_list, data_field_name)

            filtered_data_list = remove_prefix(
                filtered_text_list[0], data_field_name
            ).split(" ")

            if key != "Prob":
                for score_type in ["T score", "%"]:
                    column_names_dict[f"{key} {score_type}"].append(
                        get_score(filtered_data_list, score_type)
                    )
            else:
                column_names_dict["Prob score %"].append(
                    get_score(filtered_data_list, "Prob")
                )

    return column_names_dict


def run_pipeline(
    pdf_dir: str, csv_file_path: str, subjects: Optional[list[str]] = None
) -> None:
    """Main function to reformat filenames and extract Conners data."""
    pdf_dir = rf"{pdf_dir}"
    standardize_pdf_filenames(pdf_dir)

    reformatted_pdf_files = sorted(
        get_files(os.path.join(pdf_dir, "reformatted_filenames", "child"), "pdf")
    )

    data_fields_dict = convert_list_to_dict(CSV_COLUMN_NAMES)
    data_fields_dict["SN"] = get_subject_ids(reformatted_pdf_files, subjects)
    data_fields_dict["Visit"] = get_sessions(data_fields_dict["SN"])
    data_fields_dict["Snvisit"] = get_sn_visit(
        data_fields_dict["SN"], data_fields_dict["Visit"]
    )

    conners_data_fields_dict = create_unique_column_dict(CSV_COLUMN_NAMES)
    for key in list(conners_data_fields_dict.keys()):
        if key in ["SN", "Visit", "Snvisit"]:
            del conners_data_fields_dict[key]

    stripped_text_list = extract_pdf_text(
        pdf_file=reformatted_pdf_files[0], page_number=3
    )
    stripped_text_list = replace_data_field_names(stripped_text_list)

    extracted_data_fields_list = []
    for line in stripped_text_list:
        extracted_data_fields_list.append(re.search(r"^(\D*)", line)[0])

    for key, item in zip(conners_data_fields_dict.keys(), extracted_data_fields_list):
        conners_data_fields_dict[key] = item

    extracted_conners_data_dict = extract_conners_datafields(
        reformatted_pdf_files, conners_data_fields_dict
    )

    for key in ["SN", "Visit", "Snvisit"]:
        extracted_conners_data_dict[key] = data_fields_dict[key]

    file_name = (
        os.path.join(pdf_dir, "reformatted_filenames", "conners_data.csv")
        if csv_file_path is None
        else csv_file_path
    )

    df = pd.DataFrame(extracted_conners_data_dict)
    df.to_csv(file_name, sep=",", index=False)
