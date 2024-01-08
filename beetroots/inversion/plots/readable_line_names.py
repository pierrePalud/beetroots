r"""Former version of the ``latex-ism-emission-lines`` package by L. Einig available at  https://github.com/einigl/latex-ism-emission-lines/tree/main.
"""

import re
from typing import Optional, Sequence, Tuple, Union

__all__ = ["lines_to_latex"]


# Molecular names with underscore
_underscore_prefixes = {
    "13c_o": "13co",
    "c_18o": "c-18o",
    "13c_18o": "13c-18o",
    "h2_18o": "h2-18o",
    "c_c3h2": "c-c3h2",
}

# Molecular names to latex
_molecules_to_latex = {
    "h": "H",
    "h2": "H_2",
    "hd": "HD",
    "co": "CO",
    "13co": "^{13}CO",
    "c-18o": "C^{18}O",
    "13c-18o": "^{13}C^{18}O",
    "c": "C",
    "n": "N",
    "o": "O",
    "s": "S",
    "si": "Si",
    "cs": "CS",
    "cn": "CN",
    "hcn": "HCN",
    "hnc": "HNC",
    "oh": "OH",
    "h2o": "H_2O",
    "h2-18o": "H_2^{18}O",
    "c2h": "C_2H",
    "c-c3h2": "c-C_3H_2",
    "so": "SO",
    "cp": "C^+",
    "sp": "S^+",
    "hcop": "HCO^+",
    "chp": "CH^+",
    "ohp": "OH^+",
    "shp": "SH^+",
}

# Energy to LaTeX
_energy_to_latex = {
    "j": "J",
    "v": "\\nu",
    "f": "f",
    "n": "n",
    "ka": "k_a",
    "kc": "k_c",
}


def _transition(
    name: Optional[str], high_lvl: str, low_lvl: str
) -> Tuple[Union[str, Tuple[str, str]], bool]:
    """
    Returns a LaTeX string representing a non electronic transition.

    Parameters
    ----------
    name : str
        Energy name.
    high_lvl : str
        Higher energy level.
    low_lvl : str
        Lower energy level. Can be the same as `high_lvl`.

    Returns
    -------
    str or tuple of str
        If it is a transition: higher energy level and lower energy level. Else: energy level.
    bool
        True if it is a transition, else False
    """
    if name:
        if name in _energy_to_latex:
            name_latex = _energy_to_latex[name]
        else:
            name_latex = name
        if high_lvl == low_lvl:
            return "${}={}$".format(name_latex, low_lvl), False
        return (
            "${}={}$".format(name_latex, high_lvl),
            "${}={}$".format(name_latex, low_lvl),
        ), True
    if high_lvl == low_lvl:
        return "", False
    return ("${}$".format(high_lvl), "${}$".format(low_lvl)), True


def _eltransition(high: str, low: str) -> Tuple[Union[str, Tuple[str, str]], bool]:
    """
    Returns a LaTeX string representing an electronic transition.

    Parameters
    ----------
    high : str
        Higher energy electronic configuration.
    low : str
        Lower energy electronic configuration. Can be the same as `high`.

    Returns
    -------
    str or tuple of str
        If it is a transition: higher energy electronic configuration and lower energy configuration. Else: energy electronic configuration.
    bool
        True if it is a transition, else False
    """
    if high == low:
        return "${}$".format(high), False
    return ("${}$".format(high), "${}$".format(low)), True


def _sort_transitions(
    names: Sequence[str], high_lvls: Sequence[int], low_lvls: Sequence[int]
) -> str:
    """
    Returns a LaTeX string representing the energy transitions.
    This string first display the constant energy levels and then the energy transitions.

    Parameters
    ----------
    names : Sequence of str
        Energies names.
    high_lvls : Sequence of int
        Sequence of higher level for each energy.
    low_lvls : Sequence of int.
        Sequence of lower level for each energy.

    Returns
    -------
    str
        String representing first the constant energy levels and then the energy transitions.
    """
    if len(high_lvls) != len(names) or len(low_lvls) != len(names):
        raise ValueError("names, high_lvls and low_lvls must have the same length")

    if len(names) == 0:
        return ""
    if len(names) == 1:
        return _transition(None, high_lvls[0], low_lvls[0])

    descr_0, descr_1a, descr_1b = "", "", ""
    for name, high, low in zip(names, high_lvls, low_lvls):
        if name == "el":
            descr, istrans = _eltransition(high, low)
        else:
            descr, istrans = _transition(name, high, low)
        if istrans:
            descr_1a += descr[0] + ", "
            descr_1b += descr[1] + ", "
        else:
            descr_0 += descr + " "
    return "{} ({} $\\to$ {})".format(descr_0.strip(), descr_1a[:-2], descr_1b[:-2])


def lines_to_latex(line_name: str) -> str:
    """
    Returns a well displayed version of the formatted line ``line_name``.

    Parameters
    ----------
    line_name : str
        Formatted line.

    Returns
    -------
    str
        LaTeX string representing ``line_name``.
    """

    # Check if the input is in the good format
    if not "_" in line_name:
        return line_name
    line_name = line_name.lower().strip()

    # # Replace the underscore in molecular names containing one
    # for pref in _underscore_prefixes:
    #     if pref in line_name:
    #         line_name = line_name.replace(pref, "-", 1)
    #         break

    # Replace the underscore in molecular names containing one
    for pref, new_pref in _underscore_prefixes.items():
        if line_name.startswith(pref):
            line_name = line_name.replace(pref, f"{new_pref}", 1)
            break

    # Split the line name in two parts
    prefix, suffix = line_name.split("_", maxsplit=1)

    # Convert the prefix in LaTeX
    if prefix in _molecules_to_latex:
        latex_prefix = "${}$".format(_molecules_to_latex[prefix])
    else:
        latex_prefix = prefix

    # Convert the suffix in LaTeX
    if re.match("\Aj\d*__j\d*\Z", suffix):
        # ??__j{}__j{}
        res = re.search("\Aj(.*?)__j(.*?)\Z", suffix)
        if res:
            latex_suffix = "$(J={} \\to J={})$".format(*res.group(1, 2))
        else:
            raise ValueError("Should never been here")
    elif re.match("\Av\d*_j\d*__v\d*_j\d*\Z", suffix):
        # ??_v{}_j{}__v{}_j{}
        res = re.search("\Av(.*?)_j(.*?)__v(.*?)_j(.*?)\Z", suffix)
        if res:
            names = ["v", "j"]
            high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
        else:
            raise ValueError("Should never been here")
    elif re.match("\An\d*_j\d*__n\d*_j\d*\Z", suffix):
        # ??_n{}_j{}__n{}_j{}
        res = re.search("\An(.*?)_j(.*?)__n(.*?)_j(.*?)\Z", suffix)
        if res:
            names = ["n", "j"]
            high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
        else:
            raise ValueError("Should never been here")
    elif re.match("\Aj\d*_f\d*__j\d*_f\d*\Z", suffix):
        # ??_j{}_f{}__j{}_f{}
        res = re.search("\Aj(.*?)_f(.*?)__j(.*?)_f(.*?)\Z", suffix)
        if res:
            names = ["j", "f"]
            high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
        else:
            raise ValueError("Should never been here")
    elif re.match("\Aj\d*_ka\d*_kc\d*__j\d*_ka\d*_kc\d*\Z", suffix):
        # ??_j{}_ka{}_kc{}__j{}_ka{}_kc{}
        res = re.search("\Aj(.*?)_ka(.*?)_kc(.*?)__j(.*?)_ka(.*?)_kc(.*?)\Z", suffix)
        if res:
            names = ["j", "ka", "kc"]
            high_lvls, low_lvls = res.group(1, 2, 3), res.group(4, 5, 6)
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
        else:
            raise ValueError("Should never been here")
    elif re.match("\Ael\d*\w_j\d*__el\d*\w_j\d*\Z", suffix):
        # ??_el{int+char}_j{int}__el{int+char}_j{int}
        res = re.search("\Ael(.*?)_j(.*?)__el(.*?)_j(.*?)\Z", suffix)
        if res:
            names = ["el", "j"]
            high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
    elif re.match("\Ael\d*\w_j\d*_2__el\d*\w_j\d*_2\Z", suffix):
        # ??_el{int+char}_j{int}_2__el{int+char}_j{int}_2
        res = re.search("\Ael(.*?)_j(.*?)_2__el(.*?)_j(.*?)_2\Z", suffix)
        if res:
            names = ["el", "j"]
            high_lvls = res.group(1), r"\frac{" + f"{res.group(2)}" + r"}{2}"
            low_lvls = res.group(3), r"\frac{" + f"{res.group(4)}" + r"}{2}"
            latex_suffix = _sort_transitions(names, high_lvls, low_lvls)
    else:
        # Not adressed formats

        # oh_j5_2_pp_fif2__j3_2_pm_fif2

        # c2h_n1d0_j1d5_f1d0__n0d0_j0d5_f1d0

        # cn_n1_j0d5__n0_j0d5
        # ohp_n1_j0_f0d5__n0_j1_f0d5
        # shp_n1_j0_f0d5__n0_j1_f0d5

        latex_suffix = suffix

    out = latex_prefix + " " + latex_suffix
    out = out.replace("  ", " ")
    return out
