import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Graphs
import LatexLabels

####
# A map of rgb points in your distribution
# [distance, (r, g, b)]
# distance is percentage from left edge
# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python/50784012#50784012
heatmap = [
    [0.0, (0.1, 0.1, 1.0)],
    #  [0.20, (0, 0, .5)],
    #  [0.40, (0, .5, 0)],
    [0.40, (0.1, 1.0, 0.1)],
    #   [0.80, (.75, .75, 0)],
    #   [0.90, (1.0, .75, 0)],
    [1.00, (1.0, 0.1, 0.1)],
]


class Report:
    def __init__(self):
        self.df = pd.DataFrame()
        self.formats = []
        self.latex_labels = []
        self.head_size = 10

    def __init__(
        self,
        df,
        formats,
        latex_labels,
        formats_devices,
        cpu_threads_numbers,
        head_size=10,
    ):
        self.df = df
        self.formats = formats
        self.latex_labels = latex_labels
        self.formats_devices = formats_devices
        self.cpu_threads_numbers = cpu_threads_numbers
        self.head_size = head_size

    def effective_bw_profile(self):
        """
        Writes figures and HTML tables of effective bandwidth profiles.
        """
        if not os.path.exists("BW-profile"):
            os.mkdir("BW-profile")
        profiles = {}
        color_idx = 0
        print(self.formats)
        df = self.df
        for format in self.formats:
            print(f"Writing BW profile of {format}")
            if format == "CSR":
                df.sort_values(
                    by=[("CSR", "CPU", "1 threads", "bandwidth")],
                    inplace=True,
                    ascending=False,
                )
                profiles[format] = df[("CSR", "CPU", "1 threads", "bandwidth")].copy()
            else:
                df.sort_values(
                    by=[(format, "GPU", "bandwidth", "")], inplace=True, ascending=False
                )
                profiles[format] = df[(format, "GPU", "bandwidth", "")].copy()
            print(f"Writing BW profile of {format}")
            Graphs.draw_graphs(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} performance",
                ylabel="Effective bandwidth in GB/sec",
                filename=f"BW-profile/{format}.pdf",
                legend_loc="upper right",
                bar="none",
                yscale="linear",
                latex_labels=self.latex_labels,
            )
            Graphs.draw_graphs(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} performance",
                ylabel="Effective bandwidth in GB/sec",
                filename=f"BW-profile/{format}-log.pdf",
                legend_loc="upper right",
                bar="none",
                yscale="log",
                latex_labels=self.latex_labels,
            )
            copy_df = df.copy()
            for f in self.formats:
                if not f in ["cusparse", "CSR", format]:
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            copy_df.to_html(f"BW-profile/{format}.html")

        # Draw ellpack formats profiles
        current_formats = []
        xlabel = "Matrix number - sorted by particular formats effective bandwidth"
        ylabel = "Effective bandwidth in GB/sec"
        for format in self.formats:
            if (
                (
                    "Ellpack" in format
                    and not "Binary" in format
                    and not "Symmetric" in format
                    and not "Legacy" in format
                )
                or format == "CSR"
                or format == "cusparse"
            ):
                current_formats.append(format)
        Graphs.draw_graphs(
            current_formats,
            profiles,
            xlabel,
            ylabel,
            filename="ellpack-profiles-bw.pdf",
            legend_loc="lower left",
            bar="none",
            yscale="linear",
            latex_labels=self.latex_labels,
        )

        # Draw CSR formats profiles
        current_formats.clear()
        for format in self.formats:
            if (
                "CSR" in format
                and not "Binary" in format
                and not "Symmetric" in format
                and not "Legacy" in format
                and not "Hybrid" in format
            ) or format == "cusparse":
                current_formats.append(format)
        Graphs.draw_graphs(
            current_formats,
            profiles,
            xlabel,
            ylabel,
            filename="csr-profiles-bw.pdf",
            legend_loc="lower left",
            bar="none",
            yscale="linear",
            latex_labels=self.latex_labels,
        )

    def cusparse_comparison(self):
        """
        Writes figures and HTML tables comparing formats with cuSPARSE by the effective bandwidth.
        """
        df = self.df
        if "cusparse" in self.formats:
            if not os.path.exists("Cusparse-bw"):
                os.mkdir("Cusparse-bw")
            ascend_df = df.copy()
            df.sort_values(
                by=[("cusparse", "GPU", "bandwidth", "")], inplace=True, ascending=False
            )
            ascend_df.sort_values(
                by=[("cusparse", "GPU", "bandwidth", "")], inplace=True, ascending=True
            )
            profiles = {}
            profiles["cusparse"] = df[("cusparse", "GPU", "bandwidth", "")].copy()
            for format in self.formats:
                if not format in ["cusparse", "CSR"]:
                    print(f"Writing comparison of {format} and cuSPARSE")
                    filtered_df = df.dropna(subset=[(format, "GPU", "bandwidth", "")])
                    filtered_ascend_df = ascend_df.dropna(
                        subset=[(format, "GPU", "bandwidth", "")]
                    )
                    profiles[format] = filtered_df[
                        (format, "GPU", "bandwidth", "")
                    ].copy()
                    Graphs.draw_graphs(
                        [format, "cusparse"],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. cuSPARSE performance",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"Cusparse-bw/{format}.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="linear",
                        latex_labels=self.latex_labels,
                    )
                    Graphs.draw_graphs(
                        [format, "cusparse"],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. cuSPARSE performance",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"Cusparse-bw/{format}-log.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="log",
                        latex_labels=self.latex_labels,
                    )
                    copy_df = df.copy()
                    for f in self.formats:
                        if not f in ["cusparse", "CSR", format]:
                            copy_df.drop(
                                labels=f, axis="columns", level=0, inplace=True
                            )
                    copy_df.sort_index(inplace=True)
                    copy_df.to_html(f"Cusparse-bw/{format}.html")

    def csr_comparison(self):
        if not os.path.exists("CSR-bw"):
            os.mkdir("CSR-bw")
        df = self.df
        profiles = {}
        profiles["CSR"] = df[("CSR", "CPU", "1 threads", "bandwidth")].copy()
        for device in ["CPU", "GPU"]:
            for format in self.formats:
                if not format in ["cusparse", "CSR"]:
                    print(f"Writing comparison of {format} and CSR on CPU")
                    try:
                        df.sort_values(
                            by=[(format, device, "bandwidth", "")],
                            inplace=True,
                            ascending=False,
                        )
                    except:
                        continue
                    profiles[format] = df[(format, device, "bandwidth", "")].copy()
                    Graphs.draw_graphs(
                        [format, "CSR"],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"CSR-bw/{format}.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="linear",
                        latex_labels=self.latex_labels,
                    )

                    Graphs.draw_graphs(
                        [format, "CSR"],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"CSR-bw/{format}-log.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="log",
                        latex_labels=self.latex_labels,
                    )

                    copy_df = df.copy()
                    for f in self.formats:
                        if not f in ["cusparse", "CSR", format]:
                            copy_df.drop(
                                labels=f, axis="columns", level=0, inplace=True
                            )
                    copy_df.sort_index(inplace=True)
                    copy_df.to_html(f"CSR-bw/{format}-{device}.html")

    def legacy_formats_comparison(self):
        """
        Comparison of legacy formats with new ones by the effective bandwidth.
        """
        if not os.path.exists("Legacy-bw"):
            os.mkdir("Legacy-bw")
        df = self.df
        profiles = {}
        for ref_format, legacy_format in [
            ("Ellpack Ellpack", "Ellpack Legacy"),
            ("SlicedEllpack SlicedEllpack", "SlicedEllpack Legacy"),
            ("ChunkedEllpack ChunkedEllpack", "ChunkedEllpack Legacy"),
            ("BiEllpack BiEllpack", "BiEllpack Legacy"),
            ("CSR Adaptive", "CSR Legacy Adaptive"),
            ("CSR Scalar", "CSR Legacy Scalar"),
            ("CSR Vector", "CSR Legacy Vector"),
        ]:
            if ref_format in self.formats and legacy_format in self.formats:
                print(f"Writing comparison of {ref_format} and {legacy_format}")
                ascend_df = self.df.copy()
                df.sort_values(
                    by=[(ref_format, "GPU", "bandwidth", "")],
                    inplace=True,
                    ascending=False,
                )
                ascend_df.sort_values(
                    by=[(ref_format, "GPU", "bandwidth", "")],
                    inplace=True,
                    ascending=True,
                )
                profiles[ref_format] = df[(ref_format, "GPU", "bandwidth", "")].copy()
                profiles[legacy_format] = df[
                    (legacy_format, "GPU", "bandwidth", "")
                ].copy()

                Graphs.draw_graphs(
                    [ref_format, legacy_format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(ref_format)}",
                    ylabel="Effective bandwidth in GB/sec",
                    filename=f"Legacy-bw/{ref_format}.pdf",
                    legend_loc="upper right",
                    bar="none",
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

                copy_df = df.copy()
                for f in self.formats:
                    if not f in ["cusparse", "CSR", legacy_format]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                copy_df.to_html(f"Legacy-bw/{legacy_format}.html")

    def csr_speedup_comparison(self):
        """
        Comparison of speed-up w.r.t. CSR
        """
        if not os.path.exists("CSR-speed-up"):
            os.mkdir("CSR-speed-up")
        df = self.df
        for device in ["GPU"]:
            for format in self.formats:
                profiles = {}
                if not format in ["cusparse", "CSR"]:
                    print(f"Writing comparison of speed-up of {format} compared to CSR")
                    df["tmp"] = df[(format, device, "bandwidth", "")]
                    filtered_df = self.df.dropna(subset=[("tmp", "", "", "")])
                    try:
                        filtered_df.sort_values(
                            by=[(format, device, "speed-up", "CSR CPU")],
                            inplace=True,
                            ascending=False,
                        )
                    except:
                        continue

                    profiles[format] = filtered_df[
                        (format, device, "speed-up", "CSR CPU")
                    ].copy()
                    Graphs.draw_graphs(
                        [format],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                        ylabel="Speedup",
                        filename=f"CSR-speed-up/{format}.pdf",
                        legend_loc="upper right",
                        bar="CSR CPU",
                        yscale="linear",
                        latex_labels=self.latex_labels,
                    )
                    Graphs.draw_graphs(
                        [format],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                        ylabel="Speedup",
                        filename=f"CSR-speed-up/{format}-log.pdf",
                        legend_loc="upper right",
                        bar="CSR CPU 1 thread",
                        yscale="log",
                        latex_labels=self.latex_labels,
                    )

                    copy_df = df.copy()
                    for f in self.formats:
                        if not f in ["cusparse", "CSR", format]:
                            copy_df.drop(
                                labels=f, axis="columns", level=0, inplace=True
                            )
                    copy_df.sort_index(inplace=True)
                    copy_df.to_html(f"CSR-speed-up/{format}-{device}.html")

    ####
    # Comparison of speed-up w.r.t. Cusparse
    def cusparse_and_hypre_speedup_comparison(self):
        df = self.df
        for comparison in ["cusparse", "Hypre"]:
            if comparison in self.formats:
                if not os.path.exists(f"{comparison}-speed-up"):
                    os.mkdir(f"{comparison}-speed-up")
                profiles = {}
                for format in self.formats:
                    if not format in ["cusparse", "CSR", "Hypre"]:
                        print(
                            f"Writing comparison of speed-up of {format} ({LatexLabels.latex_label(format)}) compared to {comparison}"
                        )
                        df["tmp"] = df[(format, "GPU", "bandwidth", "")]
                        filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                        filtered_df.sort_values(
                            by=[(format, "GPU", "speed-up", comparison)],
                            inplace=True,
                            ascending=False,
                        )
                        profiles[format] = filtered_df[
                            (format, "GPU", "speed-up", comparison)
                        ].copy()
                        Graphs.draw_graphs(
                            [format],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                            ylabel="Speedup",
                            filename=f"{comparison}-speed-up/{format}.pdf",
                            legend_loc="upper right",
                            bar=comparison,
                            yscale="linear",
                            latex_labels=self.latex_labels,
                        )

                        Graphs.draw_graphs(
                            [format],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. performance of {LatexLabels.latex_label(format)}",
                            ylabel="Speedup",
                            filename=f"{comparison}-speed-up/{format}-log.pdf",
                            legend_loc="upper right",
                            bar=comparison,
                            yscale="log",
                            latex_labels=self.latex_labels,
                        )

                        # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                        # size = len(
                        #     filtered_df[(format, "GPU", "speed-up", comparison)].index
                        # )
                        # t = np.arange(size)
                        # bar = np.full(size, 1)
                        # axs.plot(
                        #     t,
                        #     filtered_df[(format, "GPU", "speed-up", comparison)],
                        #     "-o",
                        #     ms=1,
                        #     lw=1,
                        # )
                        # axs.plot(t, bar, "-", ms=1, lw=1)
                        # axs.legend(
                        #     [LatexLabels.latex_label(format), comparison], loc="upper right"
                        # )
                        # axs.set_ylabel("Speedup")
                        # axs.set_xlabel(
                        #     f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up"
                        # )
                        # plt.rcParams.update(
                        #     {
                        #         "text.usetex": True,
                        #         "font.family": "sans-serif",
                        #         "font.sans-serif": ["Helvetica"],
                        #     }
                        # )
                        # plt.savefig(f"{comparison}-speed-up/{format}.pdf")
                        # plt.close(fig)

                        # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                        # axs.set_yscale("log")
                        # axs.plot(
                        #     t,
                        #     filtered_df[(format, "GPU", "speed-up", comparison)],
                        #     "-o",
                        #     ms=1,
                        #     lw=1,
                        # )
                        # axs.plot(t, bar, "-", ms=1, lw=1)
                        # axs.legend(
                        #     [LatexLabels.latex_label(format), comparison], loc="lower left"
                        # )
                        # axs.set_xlabel(
                        #     f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up"
                        # )
                        # axs.set_ylabel("Speedup")
                        # plt.savefig(f"{comparison}-speed-up/{format}-log.pdf")
                        # plt.close(fig)
                        copy_df = df.copy()
                        for f in self.formats:
                            if not f in ["cusparse", "CSR", format, "Hypre"]:
                                copy_df.drop(
                                    labels=f, axis="columns", level=0, inplace=True
                                )
                        copy_df.sort_index(inplace=True)
                        copy_df.to_html(f"{comparison}-speed-up/{format}.html")

            # Draw Ellpack formats profiles
            xlabel = f"Matrix number - sorted by particular formats speedup compared to {comparison}"
            ylabel = "Speedup"
            current_formats = []
            for format in self.formats:
                if (
                    "Ellpack" in format
                    and not "Symmetric" in format
                    and not "Binary" in format
                    and not "Legacy" in format
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"ellpack-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

            current_formats.clear()
            for format in self.formats:
                if (
                    "Ellpack" in format
                    and "Symmetric" in format
                    and not "Binary" in format
                    and not "Legacy" in format
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"symmetric-ellpack-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

            current_formats.clear()
            for format in self.formats:
                if (
                    "Ellpack" in format
                    and not "Symmetric" in format
                    and "Binary" in format
                    and not "Legacy" in format
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"binary-ellpack-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

            current_formats.clear()
            for format in self.formats:
                if (
                    "Ellpack" in format
                    and "Symmetric" in format
                    and "Binary" in format
                    and not "Legacy" in format
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"symmetric-binary-ellpack-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

            # Draw CSR formats profiles
            current_formats.clear()
            for format in self.formats:
                if (
                    "CSR" in format
                    and not "Symmetric" in format
                    and not "Binary" in format
                    and not "Legacy" in format
                    and not "Hybrid" in format
                    and format != "CSR"
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"csr-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
            current_formats.clear()
            for format in self.formats:
                if (
                    "CSR" in format
                    and "Symmetric" in format
                    and not "Binary" in format
                    and not "Legacy" in format
                    and not "Hybrid" in format
                    and format != "CSR"
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"symmetric-csr-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
            current_formats.clear()

            for format in self.formats:
                if (
                    "CSR" in format
                    and not "Symmetric" in format
                    and "Binary" in format
                    and not "Legacy" in format
                    and not "Hybrid" in format
                    and format != "CSR"
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"binary-csr-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
            current_formats.clear()

            for format in self.formats:
                if (
                    "CSR" in format
                    and "Symmetric" in format
                    and "Binary" in format
                    and not "Legacy" in format
                    and not "Hybrid" in format
                    and format != "CSR"
                ):
                    current_formats.append(format)
            if current_formats:
                Graphs.draw_graphs(
                    current_formats,
                    profiles,
                    xlabel,
                    ylabel,
                    filename=f"symmetric-binary-csr-profiles-{comparison}-speedup.pdf",
                    legend_loc="upper right",
                    bar=comparison,
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
            current_formats.clear()

    def binary_matrices_comparison(self):
        """
        Comparison of binary matrices
        """
        df = self.df
        if not os.path.exists("binary-speed-up"):
            os.mkdir("binary-speed-up")
        for format in self.formats:
            profiles = {}
            if "Binary" in format:
                non_binary_format = format.replace("Binary ", "")
                print(
                    f"Writing comparison of speed-up of {format} vs {non_binary_format}"
                )
                # df['tmp'] = df[(format, 'GPU','speed-up','non-binary')]
                filtered_df = df.dropna(
                    subset=[(format, "GPU", "speed-up", "non-binary")]
                )  # ('tmp','','','')])
                # print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-binary')]}" )
                ascend_df = filtered_df.copy()
                filtered_df.sort_values(
                    by=[(format, "GPU", "speed-up", "non-binary")],
                    inplace=True,
                    ascending=False,
                )
                ascend_df.sort_values(
                    by=[(format, "GPU", "speed-up", "non-binary")],
                    inplace=True,
                    ascending=True,
                )
                profiles[format] = filtered_df[
                    (format, "GPU", "speed-up", "non-binary")
                ].copy()
                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"binary-speed-up/{format}.pdf",
                    legend_loc="upper right",
                    bar=f"{LatexLabels.latex_label(non_binary_format)}",
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"binary-speed-up/{format}-log.pdf",
                    legend_loc="upper right",
                    bar=f"{LatexLabels.latex_label(non_binary_format)}",
                    yscale="log",
                    latex_labels=self.latex_labels,
                )

                copy_df = filtered_df.copy()
                for f in self.formats:
                    if not f in ["cusparse", "CSR", format, non_binary_format]:
                        # print( f"Droping {f}..." )
                        # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                # head_df.to_html( f"Binary-speed-up/{format}-head.html" )
                copy_df.to_html(f"binary-speed-up/{format}.html")

    def symmetric_matrices_comparison(self):
        """
        Comparison of symmetric matrices
        """
        df = self.df
        if not os.path.exists("symmetric-speed-up"):
            os.mkdir("symmetric-speed-up")
        for format in self.formats:
            profiles = {}
            if "Symmetric" in format:
                non_symmetric_format = format.replace("Symmetric ", "")
                print(
                    f"Writing comparison of speed-up of {format} vs {non_symmetric_format}"
                )
                # df['tmp'] = df[(format, 'GPU','speed-up','non-symmetric')]
                filtered_df = df.dropna(
                    subset=[(format, "GPU", "speed-up", "non-symmetric")]
                )  # ('tmp','','','')])
                # ascend_df = filtered_df.copy()
                # print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-symmetric')]}" )
                filtered_df.sort_values(
                    by=[(format, "GPU", "speed-up", "non-symmetric")],
                    inplace=True,
                    ascending=False,
                )
                # ascend_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=True)

                cusparse_filtered_df = df.dropna(
                    subset=[(format, "GPU", "speed-up", "cusparse")]
                )  # ('tmp','','','')])
                cusparse_filtered_df.sort_values(
                    by=[(format, "GPU", "speed-up", "cusparse")],
                    inplace=True,
                    ascending=False,
                )
                profiles[format] = filtered_df[
                    (format, "GPU", "speed-up", "non-symmetric")
                ]
                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"symmetric-speed-up/{format}.pdf",
                    legend_loc="upper right",
                    bar=f"{LatexLabels.latex_label(non_symmetric_format)}",
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )
                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"symmetric-speed-up/{format}-log.pdf",
                    legend_loc="upper right",
                    bar=f"{LatexLabels.latex_label(non_symmetric_format)}",
                    yscale="log",
                    latex_labels=self.latex_labels,
                )

                profiles[format] = cusparse_filtered_df[
                    (format, "GPU", "speed-up", "cusparse")
                ]
                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"symmetric-speed-up/{format}-cusparse.pdf",
                    legend_loc="upper right",
                    bar="cuSPARSE",
                    yscale="linear",
                    latex_labels=self.latex_labels,
                )

                Graphs.draw_graphs(
                    [format],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                    ylabel="Speedup",
                    filename=f"symmetric-speed-up/{format}-cusparse-log.pdf",
                    legend_loc="upper right",
                    bar="cuSPARSE",
                    yscale="log",
                    latex_labels=self.latex_labels,
                )

                copy_df = df.copy()
                for f in self.formats:
                    if not f in ["cusparse", "CSR", format, non_symmetric_format]:
                        # print( f"Droping {f}..." )
                        # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                copy_df.sort_index(inplace=True)
                # head_df.to_html( f"Symmetric-speed-up/{format}-head.html" )
                copy_df.sort_values(
                    by=[(format, "GPU", "speed-up", "non-symmetric")],
                    inplace=True,
                    ascending=False,
                )
                copy_df.to_html(f"symmetric-speed-up/{format}.html")
                # copy_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,descending=True)
                # copy_df.to_html( f"Symmetric-speed-up/{format}-sort.html" )

    def csr_light_speedup_comparison(self):
        """
        Comparison of speed-up w.r.t. LightSpMV
        """
        df = self.df
        format = "CSR Ligh Automatic Light"
        print(f"Writing comparison of speed-up of CSR Light compared to LightSPMV")
        if (format, "GPU") in self.formats_devices:
            profiles = {}
            profiles[format] = df[
                (format, "GPU", "speed-up", "LightSpMV Vector")
            ].copy()
            Graphs.draw_graphs(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                ylabel="Speedup",
                filename="LightSpMV-speed-up.pdf",
                legend_loc="upper right",
                bar="LightSpMV",
                yscale="linear",
                latex_labels=self.latex_labels,
            )

            Graphs.draw_graphs(
                [format],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
                ylabel="Speedup",
                filename="LightSpMV-speed-up-log.pdf",
                legend_loc="upper right",
                bar="LightSpMV",
                yscale="log",
                latex_labels=self.latex_labels,
            )

            # df["tmp"] = df[(format, "GPU", "bandwidth", "")]
            # filtered_df = df.dropna(subset=[("tmp", "", "", "")])
            # ascend_df = filtered_df.copy()
            # filtered_df.sort_values(
            #     by=[(format, "GPU", "speed-up", "LightSpMV Vector")],
            #     inplace=True,
            #     ascending=False,
            # )
            # ascend_df.sort_values(
            #     by=[(format, "GPU", "speed-up", "LightSpMV Vector")],
            #     inplace=True,
            #     ascending=True,
            # )
            # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
            # size = len(filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")].index)
            # t = np.arange(size)
            # bar = np.full(size, 1)
            # axs.plot(
            #     t,
            #     filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")],
            #     "-o",
            #     ms=1,
            #     lw=1,
            # )
            # axs.plot(t, bar, "-", ms=1, lw=1)
            # axs.legend([LatexLabels.latex_label(format), "LightSpMV"], loc="upper right")
            # axs.set_ylabel("Speedup")
            # axs.set_xlabel(
            #     f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up"
            # )
            # plt.rcParams.update(
            #     {
            #         "text.usetex": True,
            #         "font.family": "sans-serif",
            #         "font.sans-serif": ["Helvetica"],
            #     }
            # )
            # # for Palatino and other serif fonts use:
            # # plt.rcParams.update({
            # #   "text.usetex": True,
            # #   "font.family": "serif",
            # #   "font.serif": ["Palatino"],
            # # })
            # plt.savefig(f"LightSpMV-speed-up.pdf")
            # plt.close(fig)

            # fig, axs = plt.subplots(1, 1, figsize=(6, 4))
            # axs.set_yscale("log")
            # axs.plot(
            #     t,
            #     filtered_df[(format, "GPU", "speed-up", "LightSpMV Vector")],
            #     "-o",
            #     ms=1,
            #     lw=1,
            # )
            # axs.plot(t, bar, "-", ms=1, lw=1)
            # axs.legend([LatexLabels.latex_label(format), "LightSpMV"], loc="lower left")
            # axs.set_xlabel(
            #     f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up"
            # )
            # axs.set_ylabel("Speedup")
            # plt.savefig(f"LightSpMV-speed-up-log.pdf")
            # plt.close(fig)
            # head_df = filtered_df.head( head_size )
            # bottom_df = ascend_df.head( head_size )
            copy_df = df.copy()
            for f in self.formats:
                if not f in ["cusparse", "CSR", format]:
                    # print( f"Droping {f}..." )
                    # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            # head_df.to_html( f"LightSpMV-speed-up-head.html" )
            copy_df.to_html(f"LightSpMV-speed-up-bottom.html")

    def csr_hypre_cpu_scalability(self):
        df = self.df
        cpu_formats = ["CSR"]
        if ("Hypre", "CPU") in self.formats_devices:
            cpu_formats.append("Hypre")
        for format in cpu_formats:
            for threads in self.threads_num_list:
                if threads == 1:
                    continue
                for metrics in ["speed-up", "eff."]:
                    threads_str = str(threads) + " threads"
                    print(f"Writing scalability of {format} on CPU")
                    df["tmp"] = df[(format, "CPU", threads_str, metrics)]
                    filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                    try:
                        filtered_df.sort_values(
                            by=[(format, "CPU", threads_str, metrics)],
                            inplace=True,
                            ascending=False,
                        )
                    except:
                        continue
                    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                    size = len(filtered_df[(format, "CPU", threads_str, metrics)].index)
                    t = np.arange(size)
                    bar = np.full(size, 1)
                    axs.plot(
                        t,
                        filtered_df[(format, "CPU", threads_str, metrics)],
                        "-o",
                        ms=1,
                        lw=1,
                    )
                    axs.plot(t, bar, "-", ms=1, lw=1)
                    axs.legend([LatexLabels.latex_label(format)], loc="upper right")
                    axs.set_ylabel(metrics)
                    axs.set_xlabel(
                        f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} {metrics}"
                    )
                    plt.rcParams.update(
                        {
                            "text.usetex": True,
                            "font.family": "sans-serif",
                            "font.sans-serif": ["Helvetica"],
                        }
                    )
                    plt.savefig(f"{format}-{threads}-threads-{metrics}.pdf")
                    plt.close(fig)

    def hypre_cpu_tnl_speedup_scalability(self):
        df = self.df
        if ("Hypre", "CPU") in self.formats_devices:
            for threads in self.threads_num_list:
                threads_str = str(threads) + " threads"
                print(f"Writing scalability of {format} on CPU")
                df["tmp"] = df[("Hypre", "CPU", threads_str, "TNL speed-up")]
                filtered_df = df.dropna(subset=[("tmp", "", "", "")])
                try:
                    filtered_df.sort_values(
                        by=[("Hypre", "CPU", threads_str, "TNL speed-up")],
                        inplace=True,
                        ascending=False,
                    )
                except:
                    continue
                fig, axs = plt.subplots(1, 1, figsize=(6, 4))
                size = len(
                    filtered_df[("Hypre", "CPU", threads_str, "TNL speed-up")].index
                )
                t = np.arange(size)
                bar = np.full(size, 1)
                axs.plot(
                    t,
                    filtered_df[("Hypre", "CPU", threads_str, "TNL speed-up")],
                    "-o",
                    ms=1,
                    lw=1,
                )
                axs.plot(t, bar, "-", ms=1, lw=1)
                axs.legend(["Hypre"], loc="upper right")
                axs.set_ylabel("Speedup")
                axs.set_xlabel(f"Matrix number - sorted w.r.t. Hypre TNL speed-up")
                plt.rcParams.update(
                    {
                        "text.usetex": True,
                        "font.family": "sans-serif",
                        "font.sans-serif": ["Helvetica"],
                    }
                )
                plt.savefig(f"Hypre-{threads}-threads-TNL-speed-up.pdf")
                plt.close(fig)

    def analyze_light_csr(self):
        """
        Analyze mapping of CUDA thredads in Light CSR
        """
        df = self.df
        sort_df = df.sort_values(
            by=[
                ("CSR Light Best", "GPU", "threads per row", ""),
                ("nonzeros per row", "", "", ""),
            ],
            inplace=False,
            ascending=True,
        )
        for f in self.formats:
            if not f in ["CSR Light Best"]:
                sort_df.drop(labels=f, axis="columns", level=0, inplace=True)
        sort_df.sort_index(inplace=True)
        sort_df.to_html(f"LightSpMV-Threads-per-row-best.html")
        # Graphs.draw_graphs(
        #     [format],
        #     profiles,
        #     xlabel=f"Matrix number - sorted w.r.t. {LatexLabels.latex_label(format)} speed-up",
        #     ylabel="Speedup",
        #     filename="LightSpMV-speed-up-log.pdf",
        #     legend_loc="upper right",
        #     bar="LightSpMV",
        #     yscale="log",
        #     latex_labels=latex_labels,
        # )

        size = len(sort_df[("nonzeros per row", "", "", "")].index)
        t = np.arange(size)
        fig, axs = plt.subplots(1, 1)
        axs.set_yticks(np.arange(0, 100, step=10))
        axs.plot(t, sort_df[("nonzeros per row", "", "", "")], "-o", ms=1, lw=1)
        axs.plot(
            t,
            sort_df[("CSR Light Best", "GPU", "threads per row", "")],
            "-o",
            ms=1,
            lw=1,
        )
        axs.legend(["Nonzeros per row", "Threads per row"], loc="upper right")
        axs.set_ylabel("CSR Light analysis")
        axs.set_ylim([0, 200])
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
            }
        )
        plt.savefig(f"LightSpMV-threads-mapping.pdf")
        plt.close(fig)

        profiles = {}
        sort_df = df.sort_values(
            by=[("nonzeros per row", "", "", "")], inplace=False, ascending=True
        )
        formats_list = [
            "CSR Light 1",
            "CSR Light 2",
            "CSR Light 4",
            "CSR Light 8",
            "CSR Light 16",
            "CSR Light 32",
        ]
        if formats_list in self.formats:
            for format in formats_list:
                sort_df.drop(labels=format, axis="columns", level=0, inplace=True)
                profiles[format] = df[(format, "GPU", "bandwidth", "")].copy()
            sort_df.sort_index(inplace=True)
            sort_df.to_html(f"LightSpMV-Threads-per-row.html")
            Graphs.draw_graphs(
                formats_list,
                profiles,
                "non-zeros per row",
                "BW",
                "nonzeros-bw.pdf",
                "lower right",
                "none",
                "linear",
                latex_labels=self.latex_labels,
            )

    def write_colormap(file, max_bw, size, x_position, y_position, standalone=False):
        if standalone:
            file.write("\\documentclass{standalone}\n")
            file.write("\\usepackage[utf8]{inputenc}\n")
            file.write("\\usepackage{tikz}\n")
            file.write("\\begin{document}\n")
            file.write("\\begin{tikzpicture}\n")
        i = 0
        x = x_position
        while i <= max_bw:
            y = y_position + i / max_bw * size
            r, g, b = color_map(i, max_bw, map=heatmap)
            file.write(f"\\definecolor{{color_hm_{i}}}{{rgb}}{{ {r}, {g}, {b} }}; \n")
            file.write(f"\\filldraw[color_hm_{i}] ({x},{y}) circle (2pt); \n")
            i = i + 5
        i = 0
        while i <= max_bw:
            y = y_position + i / max_bw * size
            file.write(
                f"\\filldraw[black] ({x},{y}) circle (1pt) node[anchor=west] {{{i}}}; \n"
            )
            i = i + 400

        if standalone:
            file.write("\\end{tikzpicture}\n")
            file.write("\\end{document}\n")

    def write_performance_circle_latex_base(file_name):
        file = open(f"{file_name}-base.tex", "w")
        file.write("\\documentclass{standalone}\n")
        file.write("\\usepackage[utf8]{inputenc}\n")
        file.write("\\usepackage{tikz}\n")
        file.write("\\begin{document}\n")
        file.write("\\begin{tikzpicture}\n")
        file.write(f"\\input{{{file_name}.tex}}\n")
        file.write("\\end{tikzpicture}\n")
        file.write("\\end{document}\n")

    #####
    # Draw performance circle in tikz
    def write_performance_circle(
        self, circle_formats, file_name, scale=1, with_color_map=False
    ):
        df = self.df
        self.write_performance_circle_latex_base(file_name)
        file = open(f"{file_name}.tex", "w")
        formats_number = 0
        for format in circle_formats:
            if format in self.formats:
                formats_number += 1

        format_idx = 0
        pos_x = 5 * scale
        pos_y = 5 * scale
        rad = 5 * scale
        formats_pos_x = {}
        formats_pos_y = {}
        for format in circle_formats:
            if format in self.formats:
                format_angle = (
                    math.pi / 2
                    - 2 * math.pi / formats_number * format_idx
                    - math.pi / formats_number
                )
                if format_angle < 0:
                    format_angle = 2 * math.pi + format_angle
                x = pos_x + rad * math.cos(format_angle)
                y = pos_y + rad * math.sin(format_angle)
                formats_pos_x[format] = x
                formats_pos_y[format] = y
                anchor = ""
                if format_angle <= math.pi * 1 / 4 or format_angle > math.pi * 7 / 4:
                    anchor = "west"
                if format_angle <= math.pi * 3 / 4 and format_angle > math.pi * 1 / 4:
                    anchor = "south"
                if format_angle <= math.pi * 5 / 4 and format_angle > math.pi * 3 / 4:
                    anchor = "east"
                if format_angle <= math.pi * 7 / 4 and format_angle > math.pi * 5 / 4:
                    anchor = "north"
                # print( f'{format_angle} : {format} -> {anchor} \n' )
                file.write(
                    f"\\filldraw[black] ({x},{y}) circle (2pt) node[anchor={anchor}]{{{LatexLabels.latex_label(format)}}}; \n"
                )
                div_angle = format_angle + math.pi / formats_number
                div_x = pos_x + rad * math.cos(div_angle)
                div_y = pos_y + rad * math.sin(div_angle)
                file.write(
                    f"\\draw [dashed] ({div_x},{div_y}) -- ({pos_x},{pos_y}); \n"
                )
                format_idx += 1
        formats_count = format_idx
        line_idx = 0
        elim = 0
        while line_idx < len(df.index):
            # matrixName = df.iloc[line_idx]['Matrix name']
            sum_bw = 0
            formats_bw = {}
            max_bw = 0
            for format in circle_formats:
                if format in self.formats:
                    format_bw = df.iloc[line_idx][(format, "GPU", "bandwidth", "")]
                    formats_bw[format] = format_bw
                    # print( f'{matrixName} {format} -> {format_bw}')
                    # if format_bw > max_bw:
                    sum_bw = sum_bw + format_bw
                    if format_bw > max_bw:
                        max_bw = format_bw
            for format in circle_formats:
                if format in self.formats:
                    formats_bw[format] = formats_bw[format] / sum_bw
            format_pos_x = 0
            format_pos_y = 0
            for format in circle_formats:
                if format in self.formats:
                    format_pos_x = (
                        format_pos_x + formats_pos_x[format] * formats_bw[format]
                    )
                    format_pos_y = (
                        format_pos_y + formats_pos_y[format] * formats_bw[format]
                    )
            if (
                format_pos_x == format_pos_x and format_pos_y == format_pos_y
            ):  # check for NaN
                r, g, b = color_map(max_bw, 1200, map=heatmap)
                file.write(
                    f"\\definecolor{{color_{line_idx}}}{{rgb}}{{ {r}, {g}, {b} }} \n"
                )
                file.write(
                    f"\\filldraw[color_{line_idx},opacity=0.75] ({format_pos_x},{format_pos_y}) circle (1pt); \n"
                )
            else:
                elim = elim + 1
            line_idx += 1
        if with_color_map:
            self.write_colormap(
                file, 1200, 5, 13 * scale, 1.5 * scale, standalone=False
            )
        os.system(f"pdflatex {file_name}-base.tex")
        print(f"Eliminated formats: {elim}")

    def write_performance_circles(self):
        df = self.df
        self.write_performance_circle(
            df,
            self.formats,
            [
                "cusparse",
                "Ellpack",
                "SlicedEllpack",
                "ChunkedEllpack",
                "BiEllpack",
                "CSR< Scalar >",
                "CSR< Adaptive >",
                "CSR< Vector >",
                "CSR< Light > Automatic Light",
            ],
            "performance-graph",
        )

        scale = 0.6
        aux_df = df
        aux_df.sort_values(
            by=[("SlicedEllpack", "GPU", "bandwidth")], inplace=True, ascending=True
        )
        self.write_performance_circle(
            aux_df,
            formats,
            ["Ellpack", "ChunkedEllpack", "SlicedEllpack"],
            "performance-graph-ellpacks-1",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            formats,
            [
                "BiEllpack",
                "ChunkedEllpack",
                "SlicedEllpack",
            ],
            "performance-graph-ellpacks-2",
            scale,
            with_color_map=False,
        )
        # write_performance_circle( df, formats, ['CSR< Scalar >', 'CSR< Adaptive >', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-csr-1' )
        aux_df.sort_values(
            by=[("CSR< Light > Automatic Light", "GPU", "bandwidth")],
            inplace=True,
            ascending=True,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["CSR< Scalar >", "CSR< Vector >", "CSR< Light > Automatic Light"],
            "performance-graph-csr-1",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["CSR< Adaptive >", "CSR< Vector >", "CSR< Light > Automatic Light"],
            "performance-graph-csr-2",
            scale,
            with_color_map=False,
        )
        aux_df.sort_values(
            by=[("cusparse", "GPU", "bandwidth")], inplace=True, ascending=True
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["cusparse", "SlicedEllpack", "ChunkedEllpack"],
            "performance-graph-cusparse-ellpacks",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["cusparse", "CSR< Vector >", "CSR< Light > Automatic Light"],
            "performance-graph-cusparse-csr-1",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["cusparse", "CSR< Adaptive >", "CSR< Light > Automatic Light"],
            "performance-graph-cusparse-csr-2",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["cusparse", "CSR< Scalar >", "CSR< Light > Automatic Light"],
            "performance-graph-cusparse-csr-3",
            scale,
            with_color_map=False,
        )
        self.write_performance_circle(
            aux_df,
            self.formats,
            ["cusparse", "SlicedEllpack", "CSR< Light > Automatic Light"],
            "performance-graph-cusparse-csr-ellpack",
            scale,
            with_color_map=False,
        )
        with open("color-map.tex", "w") as file:
            self.write_colormap(file, 1200, 5, 13 * scale, 1.5 * scale, standalone=True)

    def write(self):
        """
        Write report with benchmark data analysis
        """

        # Generate tables and figures
        self.effective_bw_profile()
        self.cusparse_comparison()
        self.csr_comparison()
        self.legacy_formats_comparison()
        self.csr_speedup_comparison()
        self.cusparse_and_hypre_speedup_comparison()
        self.binary_matrices_comparison()
        self.symmetric_matrices_comparison()
        self.csr_light_speedup_comparison()
        # self.csr_hypre_cpu_scalability()
        # self.hypre_cpu_tnl_speedup_scalability()

        best = self.df[("TNL Best", "GPU", "format", "")].tolist()
        best_formats = list(set(best))
        sum = 0
        for format in self.formats:
            if (
                not "Binary" in format
                and not "Symmetric" in format
                and not "Legacy" in format
                and not "LightSpMV" in format
                and not "TNL Best" in format
            ):
                cases = best.count(format)
                print(f"{format} is best in {cases} cases.")
                sum += cases
        print(f"Total is {sum}.")
        print(f"Best formats {best_formats}.")

        # self.write_performance_circles()
        self.analyze_light_csr()
