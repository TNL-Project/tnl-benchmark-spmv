import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zmq import device
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


def extract_sorted(df, subset, ascending=False):
    filtered_df = df.dropna(subset=[subset]).copy()
    filtered_df.sort_values(
        by=[subset],
        inplace=True,
        ascending=ascending,
    )
    return filtered_df[subset].copy()


class Report:
    def __init__(self):
        self.df = pd.DataFrame()
        self.formats = []

    def __init__(
        self,
        df,
        formats,
        launch_configs,
        legacy_counterparts,
    ):
        self.df = df
        self.formats = formats
        self.launch_configs = launch_configs
        self.legacy_counterparts = legacy_counterparts

    def effective_bw_profile(self):
        """
        Writes figures and HTML tables of effective bandwidth profiles on CPU.
        It creates separate directory "Bandwidth" with subdirectories for CPU and GPU.
        In each subdirectory, it saves the graphs with effective memory bandwidth
        together with HTML table with corresponding numbers.
        """

        if not os.path.exists("Bandwidth"):
            os.mkdir("Bandwidth")
        if not os.path.exists("Bandwidth/CPU"):
            os.mkdir("Bandwidth/CPU")
        if not os.path.exists("Bandwidth/GPU"):
            os.mkdir("Bandwidth/GPU")
        profiles = {}
        color_idx = 0
        print(self.formats)
        df = self.df
        for format in self.formats:
            for device in ["CPU", "GPU"]:
                if not (format, device) in self.launch_configs:
                    continue
                for launch_config in self.launch_configs[(format, device)]:
                    label = f"{format} {launch_config}"
                    profiles[label] = extract_sorted(
                        df,
                        (format, device, launch_config, "bandwidth", ""),
                        ascending=False,
                    )
                    print(
                        f"Writing BW profile of {format} on {device} with '{launch_config}' launch config "
                    )
                    Graphs.draw_graphs(
                        [label],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. {format} {launch_config} performance",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"Bandwidth/{device}/{format}-{launch_config}.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="linear",
                    )
                    Graphs.draw_graphs(
                        [label],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. {format} {launch_config} performance",
                        ylabel="Effective bandwidth in GB/sec",
                        filename=f"Bandwidth/{device}/{format}-{launch_config}-log.pdf",
                        legend_loc="upper right",
                        bar="none",
                        yscale="log",
                    )
                    copy_df = df.copy()
                    for f in self.formats:
                        if not f in ["cusparse", "CSR", format]:
                            copy_df.drop(
                                labels=f, axis="columns", level=0, inplace=True
                            )
                    copy_df.to_html(f"Bandwidth/{device}/{format}-{launch_config}.html")

    def ellpack_bw_profiles(self):
        """
        Draw a graph with all Ellpack based formats and cuSparse for comparison.
        """
        current_formats = []
        profiles = {}
        xlabel = "Matrix number - sorted by particular formats effective bandwidth"
        ylabel = "Effective bandwidth in GB/sec"
        df = self.df
        print("Writing ellpack profiles")
        for format in self.formats:
            if (format, "GPU") in self.launch_configs:
                for launch_config in self.launch_configs[(format, "GPU")]:
                    if (
                        "Ellpack" in format
                        and not "Binary" in format
                        and not "Symmetric" in format
                        and not "Legacy" in format
                        and not "Sorted" in format
                    ) or format == "cusparse":
                        label = f"{format} {launch_config}"
                        current_formats.append(label)
                        profiles[label] = extract_sorted(
                            df,
                            (format, "GPU", launch_config, "bandwidth", ""),
                            ascending=False,
                        )

            Graphs.draw_graphs(
                current_formats,
                profiles,
                xlabel,
                ylabel,
                filename="Ellpack-profiles-bw.pdf",
                legend_loc="lower left",
                bar="none",
                yscale="linear",
            )

    def csr_bw_profiles(self):
        """
        Draw a graph with all launch configurations of the CSR format and compares them with cuSparse.
        """
        current_formats = []
        profiles = {}
        xlabel = "Matrix number - sorted by particular formats effective bandwidth"
        ylabel = "Effective bandwidth in GB/sec"
        df = self.df
        print("Writing CSR profiles")
        for format in self.formats:
            if (
                "CSR" in format
                and not "Binary" in format
                and not "Symmetric" in format
                and not "Legacy" in format
                and not "Sorted" in format
                and not "CSR Best" in format
            ) or format == "cusparse":
                print(f"Processing format {format}")
                if (format, "GPU") in self.launch_configs:
                    for launch_config in self.launch_configs[(format, "GPU")]:
                        if (
                            not "Hybrid" in launch_config
                            and not "Warp" in launch_config
                        ):
                            label = f"{format} {launch_config}"
                            current_formats.append(label)
                            profiles[label] = extract_sorted(
                                df,
                                (format, "GPU", launch_config, "bandwidth", ""),
                                ascending=False,
                            )
        Graphs.draw_graphs(
            current_formats,
            profiles,
            xlabel,
            ylabel,
            filename="CSR-profiles-bw.pdf",
            legend_loc="lower left",
            bar="none",
            yscale="linear",
        )

    def comparison_bandwidth(self):
        """
        Writes figures and HTML tables comparing formats with cuSPARSE by the effective bandwidth.
        """
        df = self.df
        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Bandwidth"):
            os.mkdir("Comparison/Bandwidth")

        comparisons = [(("cusparse", "GPU"), "Default"), (("CSR", "CPU"), "1 thread")]
        for (
            reference_format,
            reference_device,
        ), reference_launch_config in comparisons:
            if not (reference_format, reference_device) in self.launch_configs:
                continue
            reference_label = f"{reference_format} {reference_launch_config}"
            if reference_launch_config == "Default":
                reference_label = f"{reference_format}"

            if not os.path.exists(f"Comparison/Bandwidth/{reference_label}"):
                os.mkdir(f"Comparison/Bandwidth/{reference_label}")
            profiles = {}

            profiles[reference_label] = extract_sorted(
                df,
                (
                    reference_format,
                    reference_device,
                    reference_launch_config,
                    "bandwidth",
                    "",
                ),
                ascending=False,
            )
            for format in self.formats:
                if not format in ["cusparse", reference_format]:
                    if (format, reference_device) in self.launch_configs:
                        for launch_config in self.launch_configs[
                            (format, reference_device)
                        ]:
                            label = f"{format} {launch_config}"
                            print(
                                f"Writing comparison of {format} with '{launch_config}' launch config and cuSPARSE"
                            )
                            profiles[label] = extract_sorted(
                                df,
                                (
                                    format,
                                    reference_device,
                                    launch_config,
                                    "bandwidth",
                                    "",
                                ),
                                ascending=False,
                            )
                            Graphs.draw_graphs(
                                [label, reference_label],
                                profiles,
                                xlabel=f"Matrix number - sorted w.r.t. {reference_label} performance",
                                ylabel="Effective bandwidth in GB/sec",
                                filename=f"Comparison/Bandwidth/{reference_label}/{reference_label}-{label}-{reference_device}-bw-comparison.pdf",
                                legend_loc="upper right",
                                bar="none",
                                yscale="linear",
                            )
                            Graphs.draw_graphs(
                                [label, reference_label],
                                profiles,
                                xlabel=f"Matrix number - sorted w.r.t. {reference_label} performance",
                                ylabel="Effective bandwidth in GB/sec",
                                filename=f"Comparison/Bandwidth/{reference_label}/{reference_label}-{label}-{reference_device}-bw-comparison-log.pdf",
                                legend_loc="upper right",
                                bar="none",
                                yscale="log",
                            )
                            copy_df = df.copy()
                            for f in self.formats:
                                if not f in ["cusparse", reference_format, format]:
                                    copy_df.drop(
                                        labels=f, axis="columns", level=0, inplace=True
                                    )
                            copy_df.sort_index(inplace=True)
                            copy_df.to_html(
                                f"Comparison/Bandwidth/{reference_label}-{label}-{reference_device}-bw.html"
                            )

    def comparison_legacy_formats_bandwidth(self):
        """
        Comparison of legacy formats with new ones by the effective bandwidth.
        """

        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Bandwidth"):
            os.mkdir("Comparison/Bandwidth")
        if not os.path.exists("Comparison/Bandwidth/Legacy"):
            os.mkdir("Comparison/Bandwidth/Legacy")
        df = self.df
        profiles = {}
        for ref_format, ref_launch_config in self.legacy_counterparts:
            legacy_format = self.legacy_counterparts[(ref_format, ref_launch_config)]
            print(
                f"Legacy format profiles: {ref_format} with {ref_launch_config} and {legacy_format}"
            )
            if (
                not ref_format in self.formats
                or not legacy_format in self.formats
                or not ref_launch_config in self.launch_configs[(ref_format, "GPU")]
            ):
                continue
            legacy_launch_config = "Default"
            ref_label = f"{ref_format} {ref_launch_config}"
            legacy_label = f"{legacy_format} {legacy_launch_config}"
            ascend_df = self.df.copy()
            df.sort_values(
                by=[(ref_format, "GPU", ref_launch_config, "bandwidth", "")],
                inplace=True,
                ascending=False,
            )
            ascend_df.sort_values(
                by=[(ref_format, "GPU", ref_launch_config, "bandwidth", "")],
                inplace=True,
                ascending=True,
            )
            profiles[ref_label] = df[
                (ref_format, "GPU", ref_launch_config, "bandwidth", "")
            ].copy()
            profiles[legacy_label] = df[
                (legacy_format, "GPU", legacy_launch_config, "bandwidth", "")
            ].copy()

            Graphs.draw_graphs(
                [ref_label, legacy_label],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. performance of {ref_format}",
                ylabel="Effective bandwidth in GB/sec",
                filename=f"Comparison/Bandwidth/Legacy/{ref_format}-{ref_launch_config}.pdf",
                legend_loc="upper right",
                bar="none",
                yscale="linear",
            )

            copy_df = df.copy()
            for f in self.formats:
                if not f in ["cusparse", "CSR", legacy_format]:
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            copy_df.to_html(
                f"Comparison/Bandwidth/Legacy/{ref_format}-{ref_launch_config}.html"
            )

    def comparison_speedup(self):
        """
        Writes figures and HTML tables comparing formats with cuSPARSE by the effective bandwidth.
        """
        df = self.df
        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Speedup"):
            os.mkdir("Comparison/Speedup")

        comparisons = ["cusparse", "CSR", "Hypre", "Ginkgo"]
        for reference_format in comparisons:
            if not reference_format in self.formats:
                continue

            speedup = f"{reference_format}"
            if reference_format == "CSR":
                speedup = "CSR CPU"
            reference_label = f"{reference_format}"
            if reference_format == "CSR":
                reference_label = "CSR CPU 1 thread"
            if reference_format in ["Hypre", "Ginkgo"]:
                reference_label = f"{reference_format} GPU"

            if not os.path.exists(f"Comparison/Speedup/{reference_label}"):
                os.mkdir(f"Comparison/Speedup/{reference_label}")

            for format in self.formats:
                if format == reference_format or format == "cusparse":
                    continue
                device = "GPU"
                if (format, device) in self.launch_configs:
                    for launch_config in self.launch_configs[(format, device)]:
                        profiles = {}
                        label = f"{format} {launch_config}"
                        print(
                            f"Writing speedup: {reference_label} vs {format} with '{launch_config}'"
                        )
                        # profiles[label] = extract_sorted(
                        #    df,
                        #    (format, device, launch_config, "speed-up", speedup),
                        #    ascending=False,
                        # )
                        df.loc[:, "tmp"] = df[
                            (format, device, launch_config, "bandwidth", "")
                        ]
                        filtered_df = df.dropna(subset=[("tmp", "", "", "", "")]).copy()
                        filtered_df.sort_values(
                            by=[
                                (
                                    format,
                                    device,
                                    launch_config,
                                    "speed-up",
                                    speedup,
                                )
                            ],
                            inplace=True,
                            ascending=False,
                        )
                        # print(f"Adding format {format}")
                        profiles[f"{label}"] = filtered_df[
                            (format, device, launch_config, "speed-up", speedup)
                        ].copy()
                        profiles[f"{label} BW"] = filtered_df[
                            (format, device, launch_config, "bandwidth", "")
                        ].copy()

                        Graphs.draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. performance of {label} on {device}",
                            ylabel="Speedup",
                            filename=f"Comparison/Speedup/{reference_label}/{reference_label}-{label}-{device}.pdf",
                            legend_loc="upper right",
                            bar=reference_label,
                            yscale="linear",
                        )
                        Graphs.draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. performance of {label} on {device}",
                            ylabel="Speedup",
                            filename=f"Comparison/Speedup/{reference_label}/{reference_label}-{label}-{device}-log.pdf",
                            legend_loc="upper right",
                            bar=reference_label,
                            yscale="log",
                        )
                        Graphs.draw_dual_graphs(
                            [label, f"{label} BW"],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. performance of {label} on {device}",
                            ylabels=[f"Speedup vs. {reference_label}", "Bandwidth"],
                            filename=f"Comparison/Speedup/{reference_label}/{reference_label}-{label}-{device}-bw.pdf",
                            legend_loc="none",
                            bar=reference_label,
                            yscales=["log", "linear"],
                            left_y_limits=[1.0e-4, 10],
                            right_y_limits=[0, 3],
                        )

                        copy_df = df.copy()
                        for f in self.formats:
                            if not f in ["cusparse", reference_format, format]:
                                copy_df.drop(
                                    labels=f, axis="columns", level=0, inplace=True
                                )
                        # copy_df.sort_index(inplace=True)
                        copy_df.sort_values(
                            by=[
                                (
                                    format,
                                    device,
                                    launch_config,
                                    "speed-up",
                                    speedup,
                                )
                            ],
                            inplace=True,
                            ascending=False,
                        )
                        copy_df.to_html(
                            f"Comparison/Speedup/{reference_label}/{reference_label}-{label}-{device}.html"
                        )

    def comparison_speedup_ginkgo_cpu(self):
        """
        Comparison of Ginkgo with CPU formats by the effective bandwidth.
        """

        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Speedup"):
            os.mkdir("Comparison/Speedup")
        if not os.path.exists("Comparison/Speedup/Ginkgo CPU"):
            os.mkdir("Comparison/Speedup/Ginkgo CPU")
        df = self.df
        profiles = {}
        if not ("Ginkgo", "CPU") in self.launch_configs:
            return
        for launch_config in self.launch_configs[("Ginkgo", "CPU")]:
            threads = int(launch_config.split(" ")[0])
            df.sort_values(
                by=[("Ginkgo", "CPU", launch_config, "TNL speed-up", "")],
                inplace=True,
                ascending=False,
            )
            profiles["CSR"] = df[
                ("Ginkgo", "CPU", launch_config, "TNL speed-up", "")
            ].copy()
            print(f"Writing speedup: Ginkgo vs. TNL on CPU with {threads} threads")
            Graphs.draw_graphs(
                ["CSR"],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. performance of CSR",
                ylabel="Speedup",
                filename=f"Comparison/Speedup/Ginkgo CPU/Ginkgo-CPU-CSR-{threads}-threads.pdf",
                legend_loc="upper right",
                bar="Ginkgo CPU",
                yscale="linear",
            )
            Graphs.draw_graphs(
                ["CSR"],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. performance of CSR",
                ylabel="Speedup",
                filename=f"Comparison/Speedup/Ginkgo CPU/Ginkgo-CPU-CSR-{threads}-threads-log.pdf",
                legend_loc="upper right",
                bar="Ginkgo CPU",
                yscale="log",
            )

    def comparison_speedup_legacy_formats(self):
        """
        Comparison of legacy formats based on speedup compared to the current formats
        """
        df = self.df
        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Speedup"):
            os.mkdir("Comparison/Speedup")
        if not os.path.exists("Comparison/Speedup/Legacy"):
            os.mkdir("Comparison/Speedup/Legacy")

        for ref_format, ref_launch_config in self.legacy_counterparts:
            legacy_format = self.legacy_counterparts[(ref_format, ref_launch_config)]
            print(
                f"Writing comparison: {ref_format} with {ref_launch_config} vs {legacy_format}"
            )
            for device in ["GPU"]:
                if (
                    not (ref_format, device) in self.launch_configs
                    or not (ref_format, device) in self.launch_configs
                    or not ref_launch_config
                    in self.launch_configs[(ref_format, device)]
                ):
                    continue
                ref_label = f"{ref_format} {ref_launch_config}"
                legacy_label = f"{legacy_format} {ref_launch_config}"
                profiles = {}
                df.sort_values(
                    by=[
                        (
                            ref_format,
                            device,
                            ref_launch_config,
                            "speed-up",
                            legacy_format,
                        )
                    ],
                    inplace=True,
                    ascending=False,
                )
                profiles[ref_label] = df[
                    (ref_format, device, ref_launch_config, "speed-up", legacy_format)
                ].copy()
                # print(
                #    f"format={format} device={device} legacy_format = {legacy_format} >>> {profiles[format]}"
                # )
                Graphs.draw_graphs(
                    [ref_label],
                    profiles,
                    xlabel=f"Matrix number - sorted w.r.t. speed-up of {ref_label}",
                    ylabel="Speedup",
                    filename=f"Comparison/Speedup/Legacy/{ref_label}-{device}.pdf",
                    legend_loc="upper right",
                    bar=legacy_label,
                    yscale="linear",
                )
                copy_df = df.copy()
                for f in self.formats:
                    if not f in [ref_format, legacy_format]:
                        copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
                # copy_df.sort_index(inplace=True)
                copy_df.to_html(f"Comparison/Speedup/Legacy/{ref_label}-{device}.html")

    ####
    # Comparison of speed-up w.r.t. Cusparse
    def comparison_speedup_groups_vs_reference_formats(self):
        df = self.df
        for comparison in ["cusparse", "Hypre", "Ginkgo"]:
            profiles = {}
            print(f"Comparison with {comparison}")
            if comparison in self.formats:
                for format in self.formats:
                    print(f"  Processing format {format}")
                    if (
                        not format in ["cusparse", "Hypre", "Ginkgo"]
                        and (format, "GPU") in self.launch_configs
                    ):
                        for launch_config in self.launch_configs[(format, "GPU")]:
                            label = f"{format} {launch_config}"
                            print(
                                f"Writing speed-up: {comparison} vs. {format} {launch_config}"
                            )
                            df["tmp"] = df[
                                (format, "GPU", launch_config, "bandwidth", "")
                            ]
                            filtered_df = df.dropna(
                                subset=[("tmp", "", "", "", "")]
                            ).copy()
                            filtered_df.sort_values(
                                by=[
                                    (
                                        format,
                                        "GPU",
                                        launch_config,
                                        "speed-up",
                                        comparison,
                                    )
                                ],
                                inplace=True,
                                ascending=False,
                            )
                            # print(f"Adding format {format}")
                            profiles[f"{label}"] = filtered_df[
                                (format, "GPU", launch_config, "speed-up", comparison)
                            ].copy()
                            profiles[f"{label} BW"] = filtered_df[
                                (format, "GPU", launch_config, "bandwidth", "")
                            ].copy()

            if not profiles:
                continue

            # Draw Ellpack and CSR formats profiles
            xlabel = f"Matrix number - sorted by particular formats speedup compared to {comparison}"
            ylabel = "Speedup"
            current_labels = []
            for format_class in ["Ellpack", "CSR"]:
                for accepted_class in ["", "Symmetric", "Binary", "Legacy", "Sorted"]:
                    rejected_classes = ["Symmetric", "Binary", "Legacy", "Sorted"]
                    if accepted_class in rejected_classes:
                        rejected_classes.remove(accepted_class)

                    print(f"Processing {format_class} {accepted_class}")
                    for format in self.formats:
                        skip = False
                        for rejected_class in rejected_classes:
                            if rejected_class in format:
                                skip = True
                        if skip:
                            continue

                        if (
                            format_class in format
                            and accepted_class in format
                            and (format, "GPU") in self.launch_configs
                        ):
                            for launch_config in self.launch_configs[(format, "GPU")]:
                                current_labels.append(f"{format} {launch_config}")
                    file_name = f"{accepted_class}-{format_class}-profiles-{comparison}-speedup.pdf"
                    if accepted_class == "":
                        file_name = f"{format_class}-profiles-{comparison}-speedup.pdf"
                    if current_labels:
                        Graphs.draw_graphs(
                            current_labels,
                            profiles,
                            xlabel,
                            ylabel,
                            filename=file_name,
                            legend_loc="upper right",
                            bar=comparison,
                            yscale="linear",
                        )
                    current_labels.clear()

    def comparison_speedup_binary_symmetric_sorted_matrices(self):
        """
        Comparison of binary matrices
        """
        df = self.df
        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Speedup"):
            os.mkdir("Comparison/Speedup")

        for analyzed_class in ["Symmetric", "Binary", "Sorted"]:
            if not os.path.exists(f"Comparison/Speedup/{analyzed_class}"):
                os.mkdir(f"Comparison/Speedup/{analyzed_class}")

            rejected_classes = ["Symmetric", "Binary", "Sorted"]
            rejected_classes.remove(analyzed_class)
            for format in self.formats:
                profiles = {}
                if analyzed_class in format and (format, "GPU") in self.launch_configs:
                    counterpart_format = format.replace(f"{analyzed_class} ", "")
                    counterpart_label = f"non-{analyzed_class.lower()}"
                    for launch_config in self.launch_configs[(format, "GPU")]:
                        print(
                            f"Writing speed-up: {format} {launch_config} vs {counterpart_format}"
                        )
                        filtered_df = df.dropna(
                            subset=[
                                (
                                    format,
                                    "GPU",
                                    launch_config,
                                    "speed-up",
                                    counterpart_label,
                                )
                            ]
                        ).copy()

                        ascend_df = filtered_df.copy()
                        filtered_df.sort_values(
                            by=[
                                (
                                    format,
                                    "GPU",
                                    launch_config,
                                    "speed-up",
                                    counterpart_label,
                                )
                            ],
                            inplace=True,
                            ascending=False,
                        )
                        ascend_df.sort_values(
                            by=[
                                (
                                    format,
                                    "GPU",
                                    launch_config,
                                    "speed-up",
                                    counterpart_label,
                                )
                            ],
                            inplace=True,
                            ascending=True,
                        )
                        label = f"{format} {launch_config}"
                        profiles[label] = filtered_df[
                            (
                                format,
                                "GPU",
                                launch_config,
                                "speed-up",
                                counterpart_label,
                            )
                        ].copy()
                        profiles[f"{label} BW"] = filtered_df[
                            (format, "GPU", launch_config, "bandwidth", "")
                        ].copy()

                        Graphs.draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. {label} speed-up",
                            ylabel="Speedup",
                            filename=f"Comparison/Speedup/{analyzed_class}/{label}-{counterpart_label}.pdf",
                            legend_loc="upper right",
                            bar=f"{counterpart_format} {launch_config}",
                            yscale="linear",
                        )
                        Graphs.draw_graphs(
                            [label],
                            profiles,
                            xlabel=f"Matrix number - sorted w.r.t. {label} speed-up",
                            ylabel="Speedup",
                            filename=f"Comparison/Speedup/{analyzed_class}/{label}-{counterpart_label}-log.pdf",
                            legend_loc="upper right",
                            bar=f"{counterpart_format} {launch_config}",
                            yscale="log",
                        )

                        Graphs.draw_dual_graphs(
                            [label, f"{label} BW"],
                            profiles,
                            xlabel=f"{label}",
                            ylabels=["Speedup", "Bandwidth"],
                            filename=f"Comparison/Speedup/{analyzed_class}/{label}-{counterpart_label}-with-bw.pdf",
                            legend_loc="none",
                            bar="",
                            yscales=["log", "linear"],
                            left_y_limits=[0.4, 10],
                            right_y_limits=[0, 3],
                            fig_size=(8, 3),
                        )

                    copy_df = filtered_df.dropna(
                        subset=[
                            (
                                format,
                                "GPU",
                                launch_config,
                                "speed-up",
                                counterpart_label,
                            )
                        ]
                    ).copy()
                    for f in self.formats:
                        if not f in ["cusparse", "CSR", format, "Hypre"]:
                            copy_df.drop(
                                labels=f, axis="columns", level=0, inplace=True
                            )
                    copy_df.sort_values(
                        by=[
                            (
                                format,
                                "GPU",
                                launch_config,
                                "speed-up",
                                counterpart_label,
                            )
                        ],
                        inplace=True,
                        ascending=False,
                    )
                    copy_df.to_html(
                        f"Comparison/Speedup/{analyzed_class}/{format}-{launch_config}.html"
                    )

    def comparison_speedup_light_csr(self):
        """
        Comparison of speed-up w.r.t. LightSpMV
        """
        df = self.df
        if not os.path.exists("Comparison"):
            os.mkdir("Comparison")
        if not os.path.exists("Comparison/Speedup"):
            os.mkdir("Comparison/Speedup")

        print(f"Writing speed-up: Light CSR vs. LightSPMV")
        if ("CSR", "GPU") in self.launch_configs and "Light CSR" in self.launch_configs[
            ("CSR", "GPU")
        ]:
            profiles = {}
            profiles["Light CSR"] = extract_sorted(
                df,
                ("CSR", "GPU", "Light CSR", "speed-up", "LightSpMV Vector"),
                ascending=False,
            )

            Graphs.draw_graphs(
                ["Light CSR"],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. Light CSR speed-up",
                ylabel="Speedup",
                filename="Comparison/Speedup/LightSpMV-speed-up.pdf",
                legend_loc="upper right",
                bar="LightSpMV",
                yscale="linear",
            )

            Graphs.draw_graphs(
                ["Light CSR"],
                profiles,
                xlabel=f"Matrix number - sorted w.r.t. Light CSR speed-up",
                ylabel="Speedup",
                filename="Comparison/Speedup/LightSpMV-speed-up-log.pdf",
                legend_loc="upper right",
                bar="LightSpMV",
                yscale="log",
            )

            copy_df = df.copy()
            for f in self.formats:
                if not f in ["cusparse", "CSR", format]:
                    # print( f"Droping {f}..." )
                    # head_df.drop( labels=f, axis='columns', level=0, inplace=True )
                    copy_df.drop(labels=f, axis="columns", level=0, inplace=True)
            copy_df.sort_index(inplace=True)
            # head_df.to_html( f"LightSpMV-speed-up-head.html" )
            copy_df.to_html(f"LightSpMV-speed-up-bottom.html")

    def cpu_scalability_csr_hypre_ginkgo(self):
        df = self.df
        cpu_formats = ["CSR"]
        if ("Hypre", "CPU") in self.launch_configs:
            cpu_formats.append("Hypre")
        if ("Ginkgo", "CPU") in self.launch_configs:
            cpu_formats.append("Ginkgo")

        for format in cpu_formats:
            if (format, "CPU") not in self.launch_configs:
                continue
            for launch_config in self.launch_configs[(format, "CPU")]:
                if launch_config == "Default":
                    continue
                threads = int(launch_config.split(" ")[0])
                if threads == 1:
                    continue
                for metrics in ["speed-up", "eff."]:
                    print(f"Writing scalability of {format} on CPU")
                    profiles = {}
                    profiles[format] = extract_sorted(
                        df, (format, "CPU", launch_config, metrics, ""), ascending=False
                    )
                    Graphs.draw_graphs(
                        [format],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. {format} {metrics}",
                        ylabel=f"{metrics}",
                        filename=f"CPU-{metrics}-{format}-{launch_config}.pdf",
                        legend_loc="upper right",
                        bar="",
                        yscale="linear",
                    )

    def comparison_speedup_cpu_hypre_ginkgo(self):
        df = self.df
        for format in ["Ginkgo", "Hypre"]:
            if (format, "CPU") in self.launch_configs:
                for launch_config in self.launch_configs[(format, "CPU")]:
                    print(f"Writing scalability of {format} on CPU")
                    profiles = {}
                    profiles[f"CSR {launch_config}"] = extract_sorted(
                        df,
                        (format, "CPU", launch_config, "TNL speed-up", ""),
                    )
                    Graphs.draw_graphs(
                        [f"CSR {launch_config}"],
                        profiles,
                        xlabel=f"Matrix number - sorted w.r.t. speed-up of CSR {launch_config}",
                        ylabel=f"speed-up",
                        filename=f"{format}-{launch_config}-speed-up.pdf",
                        legend_loc="upper right",
                        bar=f"{format} {launch_config}",
                        yscale="linear",
                    )

    def analyze_light_csr(self):
        """
        Analyze mapping of CUDA threads in Light CSR
        """
        df = self.df
        sort_df = df.sort_values(
            by=[
                ("CSR Best", "GPU", "", "TPS", ""),
                ("nonzeros per row", "", "", "", ""),
            ],
            inplace=False,
            ascending=True,
        ).copy()
        for f in self.formats:
            if not f in ["CSR Best"]:
                sort_df.drop(labels=f, axis="columns", level=0, inplace=True)
        sort_df.sort_index(inplace=True)
        sort_df.to_html(f"LightSpMV-Threads-per-row-best.html")
        # Graphs.draw_graphs(
        #     [format],
        #     profiles,
        #     xlabel=f"Matrix number - sorted w.r.t. {format} speed-up",
        #     ylabel="Speedup",
        #     filename="LightSpMV-speed-up-log.pdf",
        #     legend_loc="upper right",
        #     bar="LightSpMV",
        #     yscale="log",
        # )

        size = len(sort_df[("nonzeros per row", "", "", "", "")].index)
        t = np.arange(size)
        fig, axs = plt.subplots(1, 1)
        axs.set_yticks(np.arange(0, 100, step=10))
        axs.plot(t, sort_df[("nonzeros per row", "", "", "", "")], "-o", ms=1, lw=1)
        axs.plot(
            t,
            sort_df[("CSR Best", "GPU", "", "TPS", "")],
            "-o",
            ms=1,
            lw=1,
        )
        axs.legend(["Nonzeros per row", "TPS"], loc="upper right")
        axs.set_ylabel("CSR on GPU analysis")
        axs.set_ylim([0, 200])
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                # "font.sans-serif": ["Helvetica"],
            }
        )
        plt.savefig(f"LightSpMV-threads-mapping.pdf")
        plt.close(fig)

        profiles = {}
        sort_df = df.sort_values(
            by=[("nonzeros per row", "", "", "", "")], inplace=False, ascending=True
        )
        for format in self.formats:
            if format != "CSR":
                sort_df.drop(labels=format, axis="columns", level=0, inplace=True)

        # sort_df.sort_index(inplace=True)
        sort_df.to_html(f"LightSpMV-Threads-per-row.html")

        launch_configs_list = []
        for launch_config in self.launch_configs[("CSR", "GPU")]:
            if "thread" in launch_config:
                # sort_df.drop(labels="CSR", axis="columns", level=0, inplace=True)
                print(f"Adding {launch_config} to profiles...")
                profiles[launch_config] = df[
                    ("CSR", "GPU", launch_config, "bandwidth", "")
                ].copy()
                launch_configs_list.append(launch_config)
        print(launch_configs_list)
        print(profiles)
        Graphs.draw_graphs(
            launch_configs_list,
            profiles,
            "non-zeros per row",
            "BW",
            "CSR-threads-per-segment-bw.pdf",
            "upper right",
            "none",
            "linear",
        )

    def best_kernels_table(self):
        """
        Write table with best kernels for each matrix
        """
        best = self.df[("TNL Best", "GPU", "format", "", "")].tolist()
        best_kernels = list(set(best))
        print(best_kernels)
        best_kernels_df = pd.DataFrame(
            columns=[
                "0",
                "10",
                "100",
                "1000",
                "10000",
                "100000",
                "1000000",
                "10000000",
            ],
            index=best_kernels,
        )
        df = self.df
        for drop in [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            df.drop(df[df[("rows", "", "", "", "")] < drop].index, inplace=True)
            best_kernels_count = self.count_best_kernels(df)
            for kernel in best_kernels:
                if kernel in best_kernels_count:
                    best_kernels_df.loc[kernel][str(drop)] = best_kernels_count[kernel]
                else:
                    best_kernels_df.loc[kernel][str(drop)] = 0
        best_kernels_df.to_html("best-results.html")

    def write(self):
        """
        Write report with benchmark data analysis
        """

        # Generate tables and figures

        self.effective_bw_profile()
        self.ellpack_bw_profiles()
        self.csr_bw_profiles()

        self.comparison_bandwidth()
        self.comparison_legacy_formats_bandwidth()

        self.comparison_speedup()
        self.comparison_speedup_ginkgo_cpu()
        self.comparison_speedup_legacy_formats()
        self.comparison_speedup_groups_vs_reference_formats()
        self.comparison_speedup_binary_symmetric_sorted_matrices()
        self.comparison_speedup_light_csr()

        self.cpu_scalability_csr_hypre_ginkgo()
        self.comparison_speedup_cpu_hypre_ginkgo()

        # self.analyze_light_csr()
