def divide_columns(df, in_colA, in_colB, out_col):
    """
    Compute out_col = in_colA / in_colB
    """
    in_colA_list = df[in_colA]
    in_colB_list = df[in_colB]
    out_col_list = []

    for A, B in zip(in_colA_list, in_colB_list):
        div = 0
        try:
            div = A / B
        except:
            div = float("nan")
        out_col_list.append(div)
    df[out_col] = out_col_list


def divide_column_by_number(df, in_colA, number, out_col):
    """
    Compute out_col = in_colA / number
    """
    in_colA_list = df[in_colA]
    out_col_list = []

    for A in in_colA_list:
        div = 0
        try:
            div = A / number
        except:
            div = float("nan")
        out_col_list.append(div)
    df[out_col] = out_col_list


"""
Compute speed-up and efficiency of different formats and launch configurations
"""


class Speedup:
    def __init__(self, df, formats, launch_configs, legacy_counterparts):
        self.df = df
        self.formats = formats
        self.launch_configs = launch_configs
        self.legacy_counterparts = legacy_counterparts

    def compute_csr_cpu_speedup(self):
        """
        Compute speed-up and efficiency of the CSR format on CPU with different number of threads
        """
        for launch_config in self.launch_configs[("CSR", "CPU")]:
            if launch_config == "1 thread" or launch_config == "Default":
                continue
            print(f"Processing CSR CPU {launch_config}")
            threads = int(launch_config.split(" ")[0])
            divide_columns(
                self.df,
                ("CSR", "CPU", "1 thread", "time", ""),
                ("CSR", "CPU", launch_config, "time", ""),
                ("CSR", "CPU", launch_config, "speed-up", ""),
            )
            divide_column_by_number(
                self.df,
                ("CSR", "CPU", launch_config, "speed-up", ""),
                threads,
                ("CSR", "CPU", launch_config, "eff.", ""),
            )

        for format in self.formats:
            if "cusparse" in format:
                continue
            if not (format, "GPU") in self.launch_configs:
                continue
            for launch_config in self.launch_configs[(format, "GPU")]:
                divide_columns(
                    self.df,
                    ("CSR", "CPU", "1 thread", "time", ""),
                    (format, "GPU", launch_config, "time", ""),
                    (format, "GPU", launch_config, "speed-up", "CSR CPU"),
                )

    def compute_cusparse_speedup(self):
        """
        Compute speed-up of particular formats compared to Cusparse on GPU
        """
        if "cusparse" in self.formats:
            for device in ["GPU"]:
                for format in self.formats:
                    if not format in ["cusparse"]:
                        if (format, device) in self.launch_configs and (
                            format,
                            device,
                        ) in self.launch_configs:
                            for launch_config in self.launch_configs[(format, device)]:
                                divide_columns(
                                    self.df,
                                    ("cusparse", "GPU", "Default", "time", ""),
                                    (format, device, launch_config, "time", ""),
                                    (
                                        format,
                                        device,
                                        launch_config,
                                        "speed-up",
                                        "cusparse",
                                    ),
                                )

    def compute_hypre_and_ginkgo_speedup(self):
        """
        Compute speed-up of particular formats compared to Hypre and Ginkgo on GPU and CSR on CPU
        """
        for ref_format in ["Hypre", "Ginkgo"]:
            if ref_format in self.formats:
                for device in ["GPU"]:
                    for format in self.formats:
                        if not format in ["cusparse", "CSR", "Ginkgo", "Hypre"]:
                            if (format, device) in self.launch_configs and (
                                format,
                                device,
                            ) in self.launch_configs:
                                for launch_config in self.launch_configs[
                                    (format, device)
                                ]:
                                    divide_columns(
                                        self.df,
                                        (ref_format, "GPU", "Default", "time", ""),
                                        (format, device, launch_config, "time", ""),
                                        (
                                            format,
                                            device,
                                            launch_config,
                                            "speed-up",
                                            ref_format,
                                        ),
                                    )

                for launch_config in self.launch_configs[(ref_format, "CPU")]:
                    divide_columns(
                        self.df,
                        (ref_format, "CPU", launch_config, "time", ""),
                        ("CSR", "CPU", launch_config, "time", ""),
                        (ref_format, "CPU", launch_config, "TNL speed-up", ""),
                    )
                    threads = int(launch_config.split(" ")[0])
                    if threads != 1:
                        divide_columns(
                            self.df,
                            (ref_format, "CPU", "1 thread", "time", ""),
                            (ref_format, "CPU", launch_config, "time", ""),
                            (ref_format, "CPU", launch_config, "speed-up", ""),
                        )
                        divide_column_by_number(
                            self.df,
                            (ref_format, "CPU", launch_config, "speed-up", ""),
                            threads,
                            (ref_format, "CPU", launch_config, "eff.", ""),
                        )

    def compute_csr_light_speedup(self):
        """
        Compute speed-up of CSR Light Automatic and CSR Light Automatic Light compared to LightSpMV Vector
        """
        if "LightSpMV Vector" in self.formats:
            if ("CSR", "GPU") in self.launch_configs:
                if "Light CSR" in self.launch_configs[("CSR", "GPU")]:
                    divide_columns(
                        self.df,
                        ("CSR", "GPU", "Light CSR", "bandwidth", ""),
                        ("LightSpMV Vector", "GPU", "Default", "bandwidth", ""),
                        ("CSR", "GPU", "Light CSR", "speed-up", "LightSpMV Vector"),
                    )

    def compute_binary_speedup(self):
        """
        Compute speed-up of Binary formats compared to their non-binary counterparts
        """
        for format in self.formats:
            if "Binary" in format and (format, "GPU") in self.launch_configs:
                for launch_config in self.launch_configs[(format, "GPU")]:
                    non_binary_format = format.replace("Binary ", "")
                    divide_columns(
                        self.df,
                        (non_binary_format, "GPU", launch_config, "time", ""),
                        (format, "GPU", launch_config, "time", ""),
                        (format, "GPU", launch_config, "speed-up", "non-binary"),
                    )

    def compute_symmetric_speedup(self):
        """
        Compute speed-up of Symmetric formats compared to their non-symmetric counterparts
        """
        for format in self.formats:
            if "Symmetric" in format:
                if (format, "GPU") in self.launch_configs:
                    for launch_config in self.launch_configs[(format, "GPU")]:
                        non_symmetric_format = format.replace("Symmetric ", "")
                        divide_columns(
                            self.df,
                            (non_symmetric_format, "GPU", launch_config, "time", ""),
                            (format, "GPU", launch_config, "time", ""),
                            (format, "GPU", launch_config, "speed-up", "non-symmetric"),
                        )

    def compute_sorted_speedup(self):
        """
        Compute speed-up of Sorted formats compared to their non-sorted counterparts
        """
        for format in self.formats:
            if "Sorted" in format:
                if (format, "GPU") in self.launch_configs:
                    for launch_config in self.launch_configs[(format, "GPU")]:
                        non_symmetric_format = format.replace("Sorted ", "")
                        divide_columns(
                            self.df,
                            (non_symmetric_format, "GPU", launch_config, "time", ""),
                            (format, "GPU", launch_config, "time", ""),
                            (format, "GPU", launch_config, "speed-up", "non-sorted"),
                        )

    def compute_legacy_speedup(self):
        """
        Compute speed-up of formats compared to their legacy counterparts
        """
        for format in self.formats:
            if (format, "GPU") in self.launch_configs:
                for launch_config in self.launch_configs[(format, "GPU")]:
                    if not self.legacy_counterparts.get((format, launch_config)):
                        continue
                    legacy_format = self.legacy_counterparts[(format, launch_config)]
                    if legacy_format in self.formats:
                        if (format, "GPU") in self.launch_configs and (
                            legacy_format,
                            "GPU",
                        ) in self.launch_configs:
                            divide_columns(
                                self.df,
                                (legacy_format, "GPU", "Default", "time", ""),
                                (format, "GPU", launch_config, "time", ""),
                                (
                                    format,
                                    "GPU",
                                    launch_config,
                                    "speed-up",
                                    legacy_format,
                                ),
                            )

    def compute_speedup(self):

        self.compute_csr_cpu_speedup()
        self.compute_cusparse_speedup()
        self.compute_hypre_and_ginkgo_speedup()
        self.compute_csr_light_speedup()
        self.compute_binary_speedup()
        self.compute_symmetric_speedup()
        self.compute_sorted_speedup()
        self.compute_legacy_speedup()
        return self.df
