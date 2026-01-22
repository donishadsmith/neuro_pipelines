"Shared utilities"


def get_task_contrasts(task, caller):
    if task == "nback":
        contrasts = (
            "1-back_vs_0-back",
            "2-back_vs_0-back",
            "2-back_vs_1-back",
        )
    elif task == "mtle":
        contrasts = ("indoor",)
    elif task == "mtlr":
        contrasts = ("seen",)
    elif task == "princess":
        contrasts = ("switch_vs_nonswitch",)
    else:
        contrasts = (
            "congruent_vs_neutral",
            "incongruent_vs_neutral",
            "nogo_vs_neutral",
            "congruent_vs_incongruent",
            "congruent_vs_nogo",
            "incongruent_vs_nogo",
        )

    return (
        (f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )
