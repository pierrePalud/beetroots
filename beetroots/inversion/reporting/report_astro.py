import os
from typing import Dict, List

from pylatex import (
    Command,
    Document,
    Figure,
    Itemize,
    Label,
    NewPage,
    NoEscape,
    Package,
    Section,
    Subsection,
)

from beetroots.inversion.plots.readable_line_names import lines_to_latex


class ReportAstro:
    author = "Pierre Palud"

    def __init__(
        self,
        D: int,
        D_sampling: int,
        path_folder_img: str,
        list_model_names: List[str],
        pixels_of_interest: Dict[int, str],
        to_run_optim_map: bool,
        to_run_mcmc: bool,
    ):
        self.doc = Document()

        self.doc.packages.append(Package("float"))
        self.doc.packages.append(Package("hyperref"))

        self.D = D
        self.D_sampling = D_sampling

        self.path_folder_img = path_folder_img
        self.list_model_names = list_model_names
        self.pixels_of_interest = pixels_of_interest

        self.to_run_optim_map = to_run_optim_map
        self.to_run_mcmc = to_run_mcmc

    def add_title(
        self,
        cloud_name: str,
        simu_name: str,
    ) -> None:
        self.doc.preamble.append(
            Command("title", NoEscape(simu_name.replace("_", " ")))
        )
        # self.doc.preamble.append(Command("subtitle", simu_name))
        self.doc.preamble.append(Command("author", self.author))
        # self.doc.preamble.append(Command("date", dt_string))
        self.doc.append(Command("maketitle"))

        self.doc.append(NoEscape("\\tableofcontents"))
        self.doc.append(NewPage())
        return

    def add_section_observations(
        self,
        list_lines: List[str],
        list_lines_valid: List[str],
    ) -> None:
        path_folder_obs = f"{self.path_folder_img}/observations"

        L = len(list_lines)
        ncols = 5 if L > 5 else L
        nrows = L // ncols + (L % ncols > 0)
        width = 0.98 * (1 / ncols)

        with self.doc.create(
            Section(
                "Observations",
                label=False,
            )
        ):

            with self.doc.create(
                Subsection(
                    "Observation maps",
                    label=False,
                )
            ):
                # label_figure_log = r"fig:obs_log"
                # self.doc.append(
                #     r"Fig.~\ref{"
                #     + label_figure_log
                #     + r"} shows the observation maps used for inversion in log scale."
                # )
                # label_figure_lin = r"fig:obs_lin"
                # self.doc.append(
                #     r"Fig.~\ref{"
                #     + label_figure_lin
                #     + r"} shows the same observation maps in linear scale."
                # )

                with self.doc.create(Figure(position="H")) as image:
                    ell = 0
                    for i in range(nrows):
                        for j in range(ncols):
                            if ell < L:
                                if ell % ncols == 0 and ell > 0:
                                    image.append(NoEscape(r"\\"))

                                image.add_image(
                                    f"{path_folder_obs}/observation_line_{ell}_{list_lines[ell]}.PNG",
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                ell += 1

                    image.add_caption("Observation maps (log scale).")
                    # image.append(Label(label_figure_log))

                with self.doc.create(Figure(position="H")) as image:
                    ell = 0
                    for i in range(nrows):
                        for j in range(ncols):
                            if ell < L:
                                if ell % ncols == 0 and ell > 0:
                                    image.append(NoEscape(r"\\"))

                                image.add_image(
                                    f"{path_folder_obs}/observation_line_{ell}_{list_lines[ell]}_linscale.PNG",
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                ell += 1
                    image.add_caption("Observation maps (linear scale).")
                    # image.append(Label(label_figure_lin))

            with self.doc.create(
                Subsection(
                    "Standard deviation maps",
                    label=False,
                )
            ):

                with self.doc.create(Figure(position="H")) as image:
                    ell = 0
                    for i in range(nrows):
                        for j in range(ncols):
                            if ell < L:
                                if ell % ncols == 0 and ell > 0:
                                    image.append(NoEscape(r"\\"))

                                image.add_image(
                                    f"{path_folder_obs}/add_err_std_line_{ell}_{list_lines[ell]}.PNG",
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                ell += 1

                    image.add_caption("Maps of additive noise standard deviation.")

            with self.doc.create(
                Subsection(
                    "Censorship maps",
                    label=False,
                )
            ):

                with self.doc.create(Figure(position="H")) as image:
                    ell = 0
                    for i in range(nrows):
                        for j in range(ncols):
                            if ell < L:
                                if ell % ncols == 0 and ell > 0:
                                    image.append(NoEscape(r"\\"))

                                image.add_image(
                                    f"{path_folder_obs}/censored_map_line_{ell}_{list_lines[ell]}.PNG",
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                ell += 1

                    image.add_image(
                        f"{path_folder_obs}/proportion_censored_lines.PNG",
                        width=NoEscape(f"{width}\\textwidth"),
                        placement=None,
                    )

                    image.add_caption("Maps of censorship.")

            with self.doc.create(
                Subsection(
                    "Signal to Noise Ratio maps",
                    label=False,
                )
            ):

                with self.doc.create(Figure(position="H")) as image:
                    ell = 0
                    for i in range(nrows):
                        for j in range(ncols):
                            if ell < L:
                                if ell % ncols == 0 and ell > 0:
                                    image.append(NoEscape(r"\\"))

                                image.add_image(
                                    f"{path_folder_obs}/snr_line_{ell}_{list_lines[ell]}.PNG",
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                ell += 1

                    image.add_caption("Maps of SNR.")

        self.doc.append(NewPage())
        return

    def add_section_model(self, model_name: str) -> None:
        with self.doc.create(
            Section(
                NoEscape(model_name.replace("_", " ")),
                label=False,
            )
        ):
            self.add_subsection_objective(model_name)
            self.add_subsection_accepted_freq(model_name)

            self.add_subsection_pvalues(model_name)
            self.add_subsection_estimators(model_name)
            self.add_subsection_pixels_of_interest(model_name)

        # self.doc.append(NewPage())
        return

    def add_subsection_pvalues(
        self,
        model_name: str,
    ):
        path_folder_p = f"{self.path_folder_img}/p-values/{model_name}"

        with self.doc.create(
            Subsection(
                NoEscape("p-values maps"),
                label=False,
            )
        ):

            with self.doc.create(Figure(position="H")) as image:
                image.add_image(
                    f"{path_folder_p}/pval_estim_mcmc_seed0.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                image.add_image(
                    f"{path_folder_p}/decision_pval_estim_mcmc_seed0.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                if os.path.isfile(f"{path_folder_p}/proba_reject_mcmc_seed0.PNG"):
                    image.append(NoEscape(r"\\"))

                    image.add_image(
                        f"{path_folder_p}/proba_reject_mcmc_seed0.PNG",
                        width=NoEscape("0.45\\textwidth"),
                    )
                    image.add_image(
                        f"{path_folder_p}/decision_from_bayes_pval_mcmc_seed0.PNG",
                        width=NoEscape("0.45\\textwidth"),
                    )

                image.add_caption("Results of the p-values analysis.")
        return

    def add_subsection_estimators(
        self,
        model_name: str,
    ):
        path_folder_true = f"{self.path_folder_img}/true"
        path_folder_est = f"{self.path_folder_img}/estimators/{model_name}"

        ncols = 4 if os.path.isdir(path_folder_true) else 3
        width = 0.98 * (1 / ncols)

        with self.doc.create(
            Subsection(
                "Reconstructed physical parameters maps",
                label=False,
            )
        ):

            with self.doc.create(Figure(position="H")) as image:
                for d in range(self.D):
                    if os.path.isfile(f"{path_folder_est}/MAP_mcmc/MAP_mcmc_{d}.PNG"):

                        if os.path.isdir(path_folder_true):
                            image.add_image(
                                f"{path_folder_true}/true_{d}.PNG",
                                width=NoEscape(f"{width}\\textwidth"),
                                placement=None,
                            )
                        image.add_image(
                            f"{path_folder_est}/MAP_mcmc/MAP_mcmc_{d}.PNG",
                            width=NoEscape(f"{width}\\textwidth"),
                            placement=None,
                        )
                        image.add_image(
                            f"{path_folder_est}/MMSE/MMSE_{d}.PNG",
                            width=NoEscape(f"{width}\\textwidth"),
                            placement=None,
                        )

                        filename = (
                            f"{path_folder_est}/CI68/68_CI_uncertainty_factor_d{d}.PNG"
                        )
                        if os.path.isfile(filename):
                            image.add_image(
                                filename,
                                width=NoEscape(f"{width}\\textwidth"),
                                placement=None,
                            )
                        image.append(NoEscape(r"\\"))

                image.add_caption("Reconstructed physical parameters maps (log scale).")

            with self.doc.create(Figure(position="H")) as image:
                for d in range(self.D):
                    if os.path.isdir(path_folder_true):
                        image.add_image(
                            f"{path_folder_true}/true_linscale_{d}.PNG",
                            width=NoEscape(f"{width}\\textwidth"),
                            placement=None,
                        )
                    image.add_image(
                        f"{path_folder_est}/MAP_mcmc/MAP_mcmc_linscale_{d}.PNG",
                        width=NoEscape(f"{width}\\textwidth"),
                        placement=None,
                    )
                    image.add_image(
                        f"{path_folder_est}/MMSE/MMSE_linscale_{d}.PNG",
                        width=NoEscape(f"{width}\\textwidth"),
                        placement=None,
                    )
                    filename = f"{path_folder_est}/CI68/68_CI_uncertainty_factor_linscale_d{d}.PNG"
                    if os.path.isfile(filename):
                        image.add_image(
                            filename,
                            width=NoEscape(f"{width}\\textwidth"),
                            placement=None,
                        )
                    image.append(NoEscape(r"\\"))

                image.add_caption(
                    "Reconstructed physical parameters maps (linear scale)."
                )

        return

    def add_subsection_pixels_of_interest(
        self,
        model_name: str,
    ):
        path_folder_hist = f"{self.path_folder_img}/mc/{model_name}_2D/hist"
        path_folder_proba = f"{self.path_folder_img}/mc/{model_name}_2D/proba_contours"

        path_folder_yfx = f"{self.path_folder_img}/distri_comp_yspace/{model_name}"

        width = 0.95 / (self.D_sampling - 1)

        for pix_idx, name in self.pixels_of_interest.items():
            pix_text = f" for pixel {name} (idx={pix_idx})"

            with self.doc.create(
                Subsection(
                    f"Pixel {name} (idx={pix_idx})",
                    label=False,
                )
            ):

                with self.doc.create(Figure(position="H")) as image:
                    for d2 in range(self.D - 1, 0, -1):
                        tot = 0
                        for d1 in range(0, d2):
                            filename = f"{path_folder_hist}/hist2D_n{pix_idx}_d1{d1}_d2{d2}_overall_chain.PNG"
                            if os.path.isfile(filename):
                                image.add_image(
                                    filename,
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                tot += 1

                        if tot > 0:
                            image.append(NoEscape(r"\\"))

                    image.add_caption(
                        NoEscape(
                            r"2D Histograms of inferred physical parameters" + pix_text
                        ),
                    )

                with self.doc.create(Figure(position="H")) as image:
                    for d2 in range(self.D - 1, 0, -1):
                        tot = 0
                        for d1 in range(0, d2):
                            filename = f"{path_folder_proba}/HPR_2D_n{pix_idx}_d1{d1}_d2{d2}_overall_chain.PNG"
                            if os.path.isfile(filename):
                                image.add_image(
                                    filename,
                                    width=NoEscape(f"{width}\\textwidth"),
                                    placement=None,
                                )
                                tot += 1
                        if tot > 0:
                            image.append(NoEscape(r"\\"))

                    image.add_caption(
                        NoEscape(
                            r"2D contours of High Probability regions of inferred physical parameters"
                            + pix_text
                        ),
                    )

                with self.doc.create(Figure(position="H")) as image:
                    image.add_image(
                        f"{path_folder_yfx}/distribution_comparison_pix_{pix_idx}_fit.PNG",
                        width=NoEscape("0.45\\textwidth"),
                    )
                    image.add_image(
                        f"{path_folder_yfx}/distribution_comparison_pix_{pix_idx}_valid.PNG",
                        width=NoEscape("0.45\\textwidth"),
                    )
                    image.add_caption(
                        NoEscape(
                            r"Comparison of distributions on $y$ and $f(\theta)$"
                            + pix_text
                        ),
                    )

    def add_subsection_objective(
        self,
        model_name: str,
    ):
        path_folder_obj = f"{self.path_folder_img}/objective/{model_name}"
        with self.doc.create(
            Subsection(
                "Objective",
                label=False,
            )
        ):
            # label_figure = f"fig:objective_{model_name}"
            # self.doc.append(
            #     r"Fig.~\ref{"
            #     + label_figure
            #     + r"} shows the evolution of objective during sampling."
            # )

            with self.doc.create(Figure(position="H")) as image:
                image.add_image(
                    f"{path_folder_obj}/mcmc_objective_with_bi_and_true.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                image.add_image(
                    f"{path_folder_obj}/mcmc_objective_no_bi_with_true.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                image.add_caption("Evolution of objective during sampling.")
                # image.append(Label(label_figure))
        return

    def add_subsection_accepted_freq(
        self,
        model_name,
    ) -> None:
        path_folder_obj = f"{self.path_folder_img}/accepted_freq/mcmc/{model_name}"

        with self.doc.create(
            Subsection(
                "Accept frequency",
                label=False,
            )
        ):

            with self.doc.create(Figure(position="H")) as image:
                image.add_image(
                    f"{path_folder_obj}/freq_accept_seed0_MTM.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                image.add_image(
                    f"{path_folder_obj}/freq_accept_seed0_PMALA.PNG",
                    width=NoEscape("0.45\\textwidth"),
                )
                image.add_caption("Evolution of accept frequency during sampling.")

        return

    def main(
        self,
        cloud_name: str,
        simu_name: str,
        list_lines: List[str],
        list_lines_valid: List[str],
    ) -> None:
        print("\nstarting report...")
        self.add_title(cloud_name, simu_name)

        self.add_section_observations(list_lines, list_lines_valid)

        for model_name in self.list_model_names:
            self.add_section_model(model_name)

        self.doc.generate_pdf(
            f"{self.path_folder_img}/../report",
            clean_tex=False,
        )
        print("report exported to pdf.")
        return
