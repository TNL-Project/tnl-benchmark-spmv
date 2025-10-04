import os
import numpy as np
import pandas as pd
import MultiindexCreator as mic


class BestFormats:

    def __init__(self, df, formats, launch_configs):
        self.df = df
        self.formats = formats
        self.launch_configs = launch_configs

        mc = mic.MultiindexCreator(4)
        mc.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])
        mc.add_entry(["CSR Best", "CPU", "threads", ""])
        mc.add_entry(["CSR Best", "CPU", "bandwidth", ""])
        mc.add_entry(["CSR Best", "CPU", "time", ""])
        mc.add_entry(["CSR Best", "GPU", "launch cfg.", ""])
        mc.add_entry(["CSR Best", "GPU", "bandwidth", ""])
        mc.add_entry(["CSR Best", "GPU", "time", ""])
        mc.add_entry(["CSR Best", "GPU", "diff.max", ""])
        mc.add_entry(["CSR Best", "GPU", "speed-up", "cusparse"])
        mc.add_entry(["CSR Best", "GPU", "speed-up", "CSR CPU"])
        mc.add_entry(["CSR Best", "GPU", "speed-up", "Hypre"])
        mc.add_entry(["CSR Best", "GPU", "speed-up", "Ginkgo"])

        mc.add_entry(["TNL Best", "format", "", ""])
        mc.add_entry(["TNL Best", "device", "", ""])
        mc.add_entry(["TNL Best", "launch cfg.", "", ""])
        mc.add_entry(["TNL Best", "bandwidth", "", ""])
        mc.add_entry(["TNL Best", "time", "", ""])
        mc.add_entry(["TNL Best", "speed-up", "CSR CPU", ""])
        mc.add_entry(["TNL Best", "speed-up", "cusparse", ""])
        mc.add_entry(["TNL Best", "speed-up", "2nd best", ""])

        mc.add_entry(["TNL 2nd Best", "format", "", ""])
        mc.add_entry(["TNL 2nd Best", "device", "", ""])
        mc.add_entry(["TNL 2nd Best", "launch cfg.", "", ""])
        mc.add_entry(["TNL 2nd Best", "bandwidth", "", ""])
        mc.add_entry(["TNL 2nd Best", "time", "", ""])
        mc.add_entry(["TNL 2nd Best", "speed-up", "CSR CPU", ""])
        mc.add_entry(["TNL 2nd Best", "speed-up", "cusparse", ""])

        mc.add_entry(["Total Best", "format", "", ""])
        mc.add_entry(["Total Best", "device", "", ""])
        mc.add_entry(["Total Best", "launch cfg.", "", ""])
        mc.add_entry(["Total Best", "bandwidth", "", ""])
        mc.add_entry(["Total Best", "time", "", ""])
        mc.add_entry(["Total Best", "speed-up", "2nd best", ""])

        mc.add_entry(["Total 2nd Best", "format", "", ""])
        mc.add_entry(["Total 2nd Best", "device", "", ""])
        mc.add_entry(["Total 2nd Best", "launch cfg.", "", ""])
        mc.add_entry(["Total 2nd Best", "bandwidth", "", ""])
        mc.add_entry(["Total 2nd Best", "time", "", ""])

        multicolumns, df_data = mc.get_multiindex()
        self.best_df = pd.DataFrame(
            df_data, columns=multicolumns, index=range(len(self.df))
        )
        self.best_df.loc[:, ("Matrix name", "", "", "")] = self.df.loc[
            :, ("Matrix name", "", "", "")
        ]
        self.best_df.loc[:, ("rows", "", "", "")] = self.df.loc[:, ("rows", "", "", "")]
        self.best_df.loc[:, ("columns", "", "", "")] = self.df.loc[
            :, ("columns", "", "", "")
        ]
        self.best_df.loc[:, ("nonzeros per row", "", "", "")] = self.df.loc[
            :, ("nonzeros per row", "", "", "")
        ]

    def get_best_csr(self):
        """
        Get the best formats for each matrix based on the given criteria.
        """
        if ("CSR", "CPU") in self.launch_configs:
            launch_config_list = []
            bandwidth_list = []
            time_list = []
            for idx, row in self.df.iterrows():
                max_bandwidth = float("-inf")
                best_config = None
                time = None
                diff_max = None
                for config in self.launch_configs[("CSR", "CPU")]:
                    col = ("CSR", "CPU", config, "bandwidth", "")
                    if col in self.df.columns:
                        bw = row[col]
                        if bw is not None and bw > max_bandwidth:
                            max_bandwidth = bw
                            best_config = config
                            time = row[("CSR", "CPU", config, "time", "")]
                launch_config_list.append(best_config)
                bandwidth_list.append(max_bandwidth)
                time_list.append(time)
            self.best_df[("CSR Best", "CPU", "threads", "")] = launch_config_list
            self.best_df[("CSR Best", "CPU", "bandwidth", "")] = bandwidth_list
            self.best_df[("CSR Best", "CPU", "time", "")] = time_list

        if ("CSR", "GPU") in self.launch_configs:
            launch_config_list = []
            bandwidth_list = []
            time_list = []
            diff_max_list = []
            speedup_cusparse_list = []
            speedup_csr_cpu_list = []
            speedup_hypre_list = []
            speedup_ginkgo_list = []
            for idx, row in self.df.iterrows():
                max_bandwidth = float("-inf")
                best_config = None
                time = None
                diff_max = None
                speedup_cusparse = None
                speedup_csr_cpu = None
                speedup_hypre = None
                speedup_ginkgo = None
                for config in self.launch_configs[("CSR", "GPU")]:
                    col = ("CSR", "GPU", config, "bandwidth", "")
                    if col in self.df.columns:
                        bw = row[col]
                        if bw is not None and bw > max_bandwidth:
                            max_bandwidth = bw
                            best_config = config
                            time = row[("CSR", "GPU", config, "time", "")]
                            diff_max = row[("CSR", "GPU", config, "diff.max", "")]
                            speedup_cusparse = row[
                                ("CSR", "GPU", config, "speed-up", "cusparse")
                            ]
                            speedup_csr_cpu = row[
                                ("CSR", "GPU", config, "speed-up", "CSR CPU")
                            ]
                            speedup_hypre = row[
                                ("CSR", "GPU", config, "speed-up", "Hypre")
                            ]
                            speedup_ginkgo = row[
                                ("CSR", "GPU", config, "speed-up", "Ginkgo")
                            ]
                launch_config_list.append(best_config)
                bandwidth_list.append(max_bandwidth)
                time_list.append(time)
                diff_max_list.append(diff_max)
                speedup_cusparse_list.append(speedup_cusparse)
                speedup_csr_cpu_list.append(speedup_csr_cpu)
                speedup_hypre_list.append(speedup_hypre)
                speedup_ginkgo_list.append(speedup_ginkgo)
            self.best_df[("CSR Best", "GPU", "launch cfg.", "")] = launch_config_list
            self.best_df[("CSR Best", "GPU", "bandwidth", "")] = bandwidth_list
            self.best_df[("CSR Best", "GPU", "time", "")] = time_list
            self.best_df[("CSR Best", "GPU", "diff.max", "")] = diff_max_list
            self.best_df[("CSR Best", "GPU", "speed-up", "cusparse")] = (
                speedup_cusparse_list
            )
            self.best_df[("CSR Best", "GPU", "speed-up", "CSR CPU")] = (
                speedup_csr_cpu_list
            )
            self.best_df[("CSR Best", "GPU", "speed-up", "Hypre")] = speedup_hypre_list
            self.best_df[("CSR Best", "GPU", "speed-up", "Ginkgo")] = (
                speedup_ginkgo_list
            )

    def get_best_tnl_format(self):
        """
        Get the best TNL format
        """
        format_list = []
        device_list = []
        launch_config_list = []
        bandwidth_list = []
        time_list = []
        diff_max_list = []
        speedup_csr_cpu_list = []
        speedup_cusparse_list = []
        for idx, row in self.df.iterrows():
            max_bandwidth = [float("-inf"), float("-inf")]
            best_format = ["", ""]
            best_launch_config = ["", ""]
            best_device = ["", ""]
            time = ["", ""]
            speedup_csr_cpu = ["", ""]
            speedup_cusparse = ["", ""]
            for format in self.formats:
                if (
                    "Binary" in format
                    or "Symmetric" in format
                    or "cusparse" in format
                    or "Hypre" in format
                    or "Ginkgo" in format
                ):
                    continue
                for device in ["CPU", "GPU"]:
                    if (format, device) not in self.launch_configs:
                        continue
                    for launch_config in self.launch_configs[(format, device)]:
                        col = (format, device, launch_config, "bandwidth", "")
                        if col in self.df.columns:
                            bw = row[col]
                            if bw is not None and bw > max_bandwidth[0]:
                                max_bandwidth[0] = bw
                                best_format[0] = format
                                best_device[0] = device
                                best_launch_config[0] = launch_config
                                time[0] = row[
                                    (format, device, launch_config, "time", "")
                                ]
                                if "cusparse" in self.formats and device == "GPU":
                                    speedup_cusparse[0] = (
                                        max_bandwidth[0]
                                        / row[
                                            (
                                                "cusparse",
                                                "GPU",
                                                "Default",
                                                "bandwidth",
                                                "",
                                            )
                                        ]
                                    )
                                if (
                                    "CSR",
                                    "CPU",
                                ) in self.launch_configs and device == "GPU":
                                    speedup_csr_cpu[0] = (
                                        max_bandwidth[0]
                                        / row[
                                            (
                                                "CSR",
                                                "CPU",
                                                "1 thread",
                                                "bandwidth",
                                                "",
                                            )
                                        ]
                                    )
            format_list.append(best_format[0])
            device_list.append(best_device[0])
            launch_config_list.append(best_launch_config[0])
            bandwidth_list.append(max_bandwidth[0])
            time_list.append(time[0])
            speedup_csr_cpu_list.append(speedup_csr_cpu[0])
            speedup_cusparse_list.append(speedup_cusparse[0])
        self.best_df[("TNL Best", "format", "", "")] = format_list
        self.best_df[("TNL Best", "device", "", "")] = device_list
        self.best_df[("TNL Best", "launch cfg.", "", "")] = launch_config_list
        self.best_df[("TNL Best", "bandwidth", "", "")] = bandwidth_list
        self.best_df[("TNL Best", "time", "", "")] = time_list
        self.best_df[("TNL Best", "speed-up", "CSR CPU", "")] = speedup_csr_cpu_list
        self.best_df[("TNL Best", "speed-up", "cusparse", "")] = speedup_cusparse_list

    def get_total_best_format(self):
        """
        Get the best format of all
        """
        format_list = []
        device_list = []
        launch_config_list = []
        bandwidth_list = []
        time_list = []
        diff_max_list = []
        speedup_csr_cpu_list = []
        speedup_cusparse_list = []
        for idx, row in self.df.iterrows():
            max_bandwidth = float("-inf")
            best_format = None
            best_launch_config = None
            best_device = None
            time = None
            for format in self.formats:
                if "Binary" in format or "Symmetric" in format:
                    continue
                for device in ["CPU", "GPU"]:
                    if (format, device) not in self.launch_configs:
                        continue
                    for launch_config in self.launch_configs[(format, device)]:
                        col = (format, device, launch_config, "bandwidth", "")
                        if col in self.df.columns:
                            bw = row[col]
                            if bw is not None and bw > max_bandwidth:
                                max_bandwidth = bw
                                best_format = format
                                best_device = device
                                best_config = launch_config
                                time = row[(format, device, launch_config, "time", "")]
            format_list.append(best_format)
            device_list.append(best_device)
            launch_config_list.append(best_config)
            bandwidth_list.append(max_bandwidth)
            time_list.append(time)
        self.best_df[("Total Best", "format", "", "")] = format_list
        self.best_df[("Total Best", "device", "", "")] = device_list
        self.best_df[("Total Best", "launch cfg.", "", "")] = launch_config_list
        self.best_df[("Total Best", "bandwidth", "", "")] = bandwidth_list
        self.best_df[("Total Best", "time", "", "")] = time_list

    def write(self):
        self.get_best_csr()
        self.get_best_tnl_format()
        self.get_total_best_format()
        print("Writing best formats to best_formats.html")
        self.best_df.to_html("best_formats.html")

    def count_best_formats(self):
        """
        Count best formats and print them to the console
        """
        best_formats = {}
        best_sorted = 0
        best_gpu = 0
        total = 0
        for idx, row in self.best_df.iterrows():
            best_format = row[("Total Best", "format", "", "")]
            best_device = row[("Total Best", "device", "", "")]
            best_launch_config = row[("Total Best", "launch cfg.", "", "")]
            if not best_format in best_formats:
                best_formats[best_format] = 0
            best_formats[best_format] += 1
            if "Sorted" in best_format:
                best_sorted += 1
            total += 1
            if best_device == "GPU":
                best_gpu += 1

        print("=======================================================")
        print("Best formats summary")
        print("=======================================================")
        print(f"Total matrices: {total}")
        print(f"Best on GPU: {best_gpu}")
        print("Best formats:")
        for format, count in best_formats.items():
            print(f" - {format}: {count}")
        print(f"Best sorted formats: {best_sorted}")
