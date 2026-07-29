def divide_columns(df, in_colA, in_colB, out_col):
    """
    Compute out_col = in_colA / in_colB
    Division by zero yields NaN (matching the original try/except behavior).
    """
    if in_colA not in df.columns:
        raise Exception(f"Column {in_colA} not found in the dataframe")
    if in_colB not in df.columns:
        raise Exception(f"Column {in_colB} not found in the dataframe")
    if out_col not in df.columns:
        raise Exception(f"Column {out_col} not found in the dataframe")

    result = df[in_colA] / df[in_colB]
    df[out_col] = result.replace( [ float( "inf" ), float( "-inf" ) ], float( "nan" ) )


def divide_column_by_number(df, in_colA, number, out_col):
    """
    Compute out_col = in_colA / number
    Division by zero yields NaN (matching the original try/except behavior).
    """
    if in_colA not in df.columns:
        raise Exception(f"Column {in_colA} not found in the dataframe")
    if out_col not in df.columns:
        raise Exception(f"Column {out_col} not found in the dataframe")

    result = df[in_colA] / number
    df[out_col] = result.replace( [ float( "inf" ), float( "-inf" ) ], float( "nan" ) )


"""
Compute speed-up and efficiency of different formats and launch configurations
"""


class Speedup:
    def __init__(self, df, formats, launch_configs, legacy_counterparts, accelerator_devices):
        self.df = df
        self.formats = formats
        self.launch_configs = launch_configs
        self.legacy_counterparts = legacy_counterparts
        self.accelerator_devices = accelerator_devices

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
                ("CSR", "CPU", "1 thread", "time mean", ""),
                ("CSR", "CPU", launch_config, "time mean", ""),
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
            for device in self.accelerator_devices:
                if not (format, device) in self.launch_configs:
                    continue
                for launch_config in self.launch_configs[(format, device)]:
                    divide_columns(
                        self.df,
                        ("CSR", "CPU", "1 thread", "time mean", ""),
                        (format, device, launch_config, "time mean", ""),
                        (format, device, launch_config, "speed-up", "CSR CPU"),
                    )

    def compute_cusparse_speedup(self):
        """
        Compute speed-up of particular formats compared to Cusparse on the same accelerator device
        """
        if "cusparse" in self.formats:
            for device in self.accelerator_devices:
                if not ("cusparse", device) in self.launch_configs:
                    continue
                for format in self.formats:
                    if not format in ["cusparse"]:
                        if (format, device) in self.launch_configs:
                            for launch_config in self.launch_configs[(format, device)]:
                                divide_columns(
                                    self.df,
                                    ("cusparse", device, "Default", "time mean", ""),
                                    (format, device, launch_config, "time mean", ""),
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
        Compute speed-up of particular formats compared to Hypre and Ginkgo on each accelerator device and CSR on CPU
        """
        for ref_format in ["Hypre", "Ginkgo"]:
            if ref_format in self.formats:
                for device in self.accelerator_devices:
                    for format in self.formats:
                        if not format in ["cusparse", "CSR", "Ginkgo", "Hypre"]:
                            if (format, device) in self.launch_configs and (
                                format,
                                device,
                            ) in self.launch_configs:
                                for launch_config in self.launch_configs[
                                    (format, device)
                                ]:
                                    if (
                                        ref_format,
                                        device,
                                        "Default",
                                        "time mean",
                                        "",
                                    ) in self.df.columns:
                                        divide_columns(
                                            self.df,
                                            (ref_format, device, "Default", "time mean", ""),
                                            (format, device, launch_config, "time mean", ""),
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
                        (ref_format, "CPU", launch_config, "time mean", ""),
                        ("CSR", "CPU", launch_config, "time mean", ""),
                        (ref_format, "CPU", launch_config, "TNL speed-up", ""),
                    )
                    threads = int(launch_config.split(" ")[0])
                    if threads != 1:
                        divide_columns(
                            self.df,
                            (ref_format, "CPU", "1 thread", "time mean", ""),
                            (ref_format, "CPU", launch_config, "time mean", ""),
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
            for device in self.accelerator_devices:
                if ("CSR", device) in self.launch_configs:
                    if "Light CSR" in self.launch_configs[("CSR", device)]:
                        divide_columns(
                            self.df,
                            ("CSR", device, "Light CSR", "bandwidth", ""),
                            ("LightSpMV Vector", device, "Default", "bandwidth", ""),
                            ("CSR", device, "Light CSR", "speed-up", "LightSpMV Vector"),
                        )

    def compute_binary_speedup(self):
        """
        Compute speed-up of Binary formats compared to their non-binary counterparts
        """
        for format in self.formats:
            if "Binary" in format:
                for device in self.accelerator_devices:
                    if (format, device) in self.launch_configs:
                        for launch_config in self.launch_configs[(format, device)]:
                            non_binary_format = format.replace("Binary ", "")
                            divide_columns(
                                self.df,
                                (non_binary_format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "speed-up", "non-binary"),
                            )

    def compute_symmetric_speedup(self):
        """
        Compute speed-up of Symmetric formats compared to their non-symmetric counterparts
        """
        for format in self.formats:
            if "Symmetric" in format:
                for device in self.accelerator_devices:
                    if (format, device) in self.launch_configs:
                        for launch_config in self.launch_configs[(format, device)]:
                            non_symmetric_format = format.replace("Symmetric ", "")
                            divide_columns(
                                self.df,
                                (non_symmetric_format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "speed-up", "non-symmetric"),
                            )

    def compute_sorted_speedup(self):
        """
        Compute speed-up of Sorted formats compared to their non-sorted counterparts
        """
        for format in self.formats:
            if "Sorted" in format:
                for device in self.accelerator_devices:
                    if (format, device) in self.launch_configs:
                        for launch_config in self.launch_configs[(format, device)]:
                            non_symmetric_format = format.replace("Sorted ", "")
                            divide_columns(
                                self.df,
                                (non_symmetric_format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "time mean", ""),
                                (format, device, launch_config, "speed-up", "non-sorted"),
                            )

    def compute_legacy_speedup(self):
        """
        Compute speed-up of formats compared to their legacy counterparts
        """
        for format in self.formats:
            for device in self.accelerator_devices:
                if (format, device) in self.launch_configs:
                    for launch_config in self.launch_configs[(format, device)]:
                        if not self.legacy_counterparts.get((format, launch_config)):
                            continue
                        legacy_format = self.legacy_counterparts[(format, launch_config)]
                        if legacy_format in self.formats:
                            if (format, device) in self.launch_configs and (
                                legacy_format,
                                device,
                            ) in self.launch_configs:
                                divide_columns(
                                    self.df,
                                    (legacy_format, device, "Default", "time mean", ""),
                                    (format, device, launch_config, "time mean", ""),
                                    (
                                        format,
                                        device,
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
