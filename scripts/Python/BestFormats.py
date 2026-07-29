import os
import numpy as np
import pandas as pd
import MultiindexCreator as mic


class BestFormats:

    def __init__(self, df, formats, launch_configs, accelerator_devices):
        self.df = df
        self.formats = formats
        self.launch_configs = launch_configs
        self.accelerator_devices = accelerator_devices

        mc_csr = mic.MultiindexCreator(4)
        mc_csr.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])
        mc_csr.add_entry(["CSR Best", "CPU", "threads", ""])
        mc_csr.add_entry(["CSR Best", "CPU", "bandwidth", ""])
        mc_csr.add_entry(["CSR Best", "CPU", "time", ""])
        for device in accelerator_devices:
            mc_csr.add_entry(["CSR Best", device, "launch cfg.", ""])
            mc_csr.add_entry(["CSR Best", device, "bandwidth", ""])
            mc_csr.add_entry(["CSR Best", device, "time", ""])
            mc_csr.add_entry(["CSR Best", device, "diff.max", ""])
            mc_csr.add_entry(["CSR Best", device, "speed-up", "cusparse"])
            mc_csr.add_entry(["CSR Best", device, "speed-up", "CSR CPU"])
            mc_csr.add_entry(["CSR Best", device, "speed-up", "Hypre"])
            mc_csr.add_entry(["CSR Best", device, "speed-up", "Ginkgo"])
            mc_csr.add_entry(["CSR Best", device, "speed-up", "2nd best"])

            mc_csr.add_entry(["CSR 2nd Best", device, "launch cfg.", ""])
            mc_csr.add_entry(["CSR 2nd Best", device, "bandwidth", ""])
            mc_csr.add_entry(["CSR 2nd Best", device, "time", ""])
            mc_csr.add_entry(["CSR 2nd Best", device, "diff.max", ""])
            mc_csr.add_entry(["CSR 2nd Best", device, "speed-up", "cusparse"])
            mc_csr.add_entry(["CSR 2nd Best", device, "speed-up", "CSR CPU"])
            mc_csr.add_entry(["CSR 2nd Best", device, "speed-up", "Hypre"])
            mc_csr.add_entry(["CSR 2nd Best", device, "speed-up", "Ginkgo"])

        multicolumns, df_data = mc_csr.get_multiindex()
        self.best_csr_df = pd.DataFrame(
            df_data, columns=multicolumns, index=range(len(self.df))
        )
        self.best_csr_df.loc[:, ("Matrix name", "", "", "")] = self.df.loc[
            :, ("Matrix name", "", "", "")
        ]
        self.best_csr_df.loc[:, ("rows", "", "", "")] = self.df.loc[:, ("rows", "", "", "")]
        self.best_csr_df.loc[:, ("columns", "", "", "")] = self.df.loc[
            :, ("columns", "", "", "")
        ]
        self.best_csr_df.loc[:, ("nonzeros per row", "", "", "")] = self.df.loc[
            :, ("nonzeros per row", "", "", "")
        ]


        mc_tnl_best = mic.MultiindexCreator(4)
        mc_tnl_best.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])
        mc_tnl_best.add_entry(["TNL Best", "format", "", ""])
        mc_tnl_best.add_entry(["TNL Best", "device", "", ""])
        mc_tnl_best.add_entry(["TNL Best", "launch cfg.", "", ""])
        mc_tnl_best.add_entry(["TNL Best", "bandwidth", "", ""])
        mc_tnl_best.add_entry(["TNL Best", "time", "", ""])
        mc_tnl_best.add_entry(["TNL Best", "speed-up", "CSR CPU", ""])
        mc_tnl_best.add_entry(["TNL Best", "speed-up", "cusparse", ""])
        mc_tnl_best.add_entry(["TNL Best", "speed-up", "2nd best", ""])

        mc_tnl_best.add_entry(["TNL 2nd Best", "format", "", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "device", "", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "launch cfg.", "", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "bandwidth", "", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "time", "", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "speed-up", "CSR CPU", ""])
        mc_tnl_best.add_entry(["TNL 2nd Best", "speed-up", "cusparse", ""])

        multicolumns, df_data = mc_tnl_best.get_multiindex()
        self.best_tnl_df = pd.DataFrame(
            df_data, columns=multicolumns, index=range(len(self.df))
        )
        self.best_tnl_df.loc[:, ("Matrix name", "", "", "")] = self.df.loc[
            :, ("Matrix name", "", "", "")
        ]
        self.best_tnl_df.loc[:, ("rows", "", "", "")] = self.df.loc[:, ("rows", "", "", "")]
        self.best_tnl_df.loc[:, ("columns", "", "", "")] = self.df.loc[
            :, ("columns", "", "", "")
        ]
        self.best_tnl_df.loc[:, ("nonzeros per row", "", "", "")] = self.df.loc[
            :, ("nonzeros per row", "", "", "")
        ]

        mc_total_best = mic.MultiindexCreator(4)
        mc_total_best.add_entries([["Matrix name"], ["rows"], ["columns"], ["nonzeros per row"]])
        mc_total_best.add_entry(["Total Best", "format", "", ""])
        mc_total_best.add_entry(["Total Best", "device", "", ""])
        mc_total_best.add_entry(["Total Best", "launch cfg.", "", ""])
        mc_total_best.add_entry(["Total Best", "bandwidth", "", ""])
        mc_total_best.add_entry(["Total Best", "time", "", ""])
        mc_total_best.add_entry(["Total Best", "speed-up", "2nd best", ""])

        mc_total_best.add_entry(["Total 2nd Best", "format", "", ""])
        mc_total_best.add_entry(["Total 2nd Best", "device", "", ""])
        mc_total_best.add_entry(["Total 2nd Best", "launch cfg.", "", ""])
        mc_total_best.add_entry(["Total 2nd Best", "bandwidth", "", ""])
        mc_total_best.add_entry(["Total 2nd Best", "time", "", ""])

        multicolumns, df_data = mc_total_best.get_multiindex()
        self.best_total_df = pd.DataFrame(
            df_data, columns=multicolumns, index=range(len(self.df))
        )
        self.best_total_df.loc[:, ("Matrix name", "", "", "")] = self.df.loc[
            :, ("Matrix name", "", "", "")
        ]
        self.best_total_df.loc[:, ("rows", "", "", "")] = self.df.loc[:, ("rows", "", "", "")]
        self.best_total_df.loc[:, ("columns", "", "", "")] = self.df.loc[
            :, ("columns", "", "", "")
        ]
        self.best_total_df.loc[:, ("nonzeros per row", "", "", "")] = self.df.loc[
            :, ("nonzeros per row", "", "", "")
        ]

        self.best_csr_df.to_html("best_csr_formats_empty.html")

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
                            time = row[("CSR", "CPU", config, "time mean", "")]
                launch_config_list.append(best_config)
                bandwidth_list.append(max_bandwidth)
                time_list.append(time)
            self.best_csr_df[("CSR Best", "CPU", "threads", "")] = launch_config_list
            self.best_csr_df[("CSR Best", "CPU", "bandwidth", "")] = bandwidth_list
            self.best_csr_df[("CSR Best", "CPU", "time", "")] = time_list

        for device in self.accelerator_devices:
            if ("CSR", device) not in self.launch_configs:
                continue
            launch_config_list = [[], []]
            bandwidth_list = [[], []]
            time_list = [[], []]
            diff_max_list = [[], []]
            speedup_cusparse_list = [[], []]
            speedup_csr_cpu_list = [[], []]
            speedup_hypre_list = [[], []]
            speedup_ginkgo_list = [[], []]
            speedup_second_best_list = []
            for idx, row in self.df.iterrows():
                max_bandwidth = [float("-inf"), float("-inf")]
                best_config = [None, None]
                time = [0, 0]
                diff_max = [None, None]
                speedup_cusparse = [None, None]
                speedup_csr_cpu = [None, None]
                speedup_hypre = [None, None]
                speedup_ginkgo = [None, None]
                speedup_second_best = None
                for config in self.launch_configs[("CSR", device)]:
                    col = ("CSR", device, config, "bandwidth", "")
                    if col in self.df.columns:
                        bw = row[col]
                        if bw is not None and bw > max_bandwidth[0]:
                            max_bandwidth[1] = max_bandwidth[0]
                            best_config[1] = best_config[0]
                            time[1] = time[0]
                            diff_max[1] = diff_max[0]
                            speedup_cusparse[1] = speedup_cusparse[0]
                            speedup_csr_cpu[1] = speedup_csr_cpu[0]
                            speedup_hypre[1] = speedup_hypre[0]
                            speedup_ginkgo[1] = speedup_ginkgo[0]
                            max_bandwidth[0] = bw
                            best_config[0] = config
                            time[0] = row[("CSR", device, config, "time mean", "")]
                            diff_max[0] = row[("CSR", device, config, "diff.max", "")]
                            speedup_cusparse[0] = row[
                                ("CSR", device, config, "speed-up", "cusparse")
                            ]
                            speedup_csr_cpu[0] = row[
                                ("CSR", device, config, "speed-up", "CSR CPU")
                            ]
                            speedup_hypre[0] = row[
                                ("CSR", device, config, "speed-up", "Hypre")
                            ]
                            speedup_ginkgo[0] = row[
                                ("CSR", device, config, "speed-up", "Ginkgo")
                            ]
                            if time[0] != "" and time[1] != "":
                                speedup_second_best = time[1] / time[0]
                # 1st best
                launch_config_list[0].append(best_config[0])
                bandwidth_list[0].append(max_bandwidth[0])
                time_list[0].append(time[0])
                diff_max_list[0].append(diff_max[0])
                speedup_cusparse_list[0].append(speedup_cusparse[0])
                speedup_csr_cpu_list[0].append(speedup_csr_cpu[0])
                speedup_hypre_list[0].append(speedup_hypre[0])
                speedup_ginkgo_list[0].append(speedup_ginkgo[0])
                speedup_second_best_list.append(speedup_second_best)

                # 2nd best
                launch_config_list[1].append(best_config[1])
                bandwidth_list[1].append(max_bandwidth[1])
                time_list[1].append(time[1])
                diff_max_list[1].append(diff_max[1])
                speedup_cusparse_list[1].append(speedup_cusparse[1])
                speedup_csr_cpu_list[1].append(speedup_csr_cpu[1])
                speedup_hypre_list[1].append(speedup_hypre[1])
                speedup_ginkgo_list[1].append(speedup_ginkgo[1])

            # 1st best
            self.best_csr_df[("CSR Best", device, "launch cfg.", "")] = launch_config_list[0]
            self.best_csr_df[("CSR Best", device, "bandwidth", "")] = bandwidth_list[0]
            self.best_csr_df[("CSR Best", device, "time", "")] = time_list[0]
            self.best_csr_df[("CSR Best", device, "diff.max", "")] = diff_max_list[0]
            self.best_csr_df[("CSR Best", device, "speed-up", "cusparse")] = (
                speedup_cusparse_list[0]
            )
            self.best_csr_df[("CSR Best", device, "speed-up", "CSR CPU")] = (
                speedup_csr_cpu_list[0]
            )
            self.best_csr_df[("CSR Best", device, "speed-up", "Hypre")] = speedup_hypre_list[
                0
            ]
            self.best_csr_df[("CSR Best", device, "speed-up", "Ginkgo")] = (
                speedup_ginkgo_list[0]
            )
            self.best_csr_df[("CSR Best", device, "speed-up", "2nd best")] = (
                speedup_second_best_list[0]
            )

            # 2nd best
            self.best_csr_df[("CSR 2nd Best", device, "launch cfg.", "")] = (
                launch_config_list[1]
            )
            self.best_csr_df[("CSR 2nd Best", device, "bandwidth", "")] = bandwidth_list[1]
            self.best_csr_df[("CSR 2nd Best", device, "time", "")] = time_list[1]
            self.best_csr_df[("CSR 2nd Best", device, "diff.max", "")] = diff_max_list[1]
            self.best_csr_df[("CSR 2nd Best", device, "speed-up", "cusparse")] = (
                speedup_cusparse_list[1]
            )
            self.best_csr_df[("CSR 2nd Best", device, "speed-up", "CSR CPU")] = (
                speedup_csr_cpu_list[1]
            )
            self.best_csr_df[("CSR 2nd Best", device, "speed-up", "Hypre")] = (
                speedup_hypre_list[1]
            )
            self.best_csr_df[("CSR 2nd Best", device, "speed-up", "Ginkgo")] = (
                speedup_ginkgo_list[1]
            )

    def get_best_tnl_format(self):
        """
        Get the best TNL format
        """
        format_list = [[], []]
        device_list = [[], []]
        launch_config_list = [[], []]
        bandwidth_list = [[], []]
        time_list = [[], []]
        diff_max_list = [[], []]
        speedup_csr_cpu_list = [[], []]
        speedup_cusparse_list = [[], []]
        speedup_second_best_list = []
        for idx, row in self.df.iterrows():
            max_bandwidth = [float("-inf"), float("-inf")]
            best_format = ["", ""]
            best_launch_config = ["", ""]
            best_device = ["", ""]
            time = ["", ""]
            speedup_csr_cpu = ["", ""]
            speedup_cusparse = ["", ""]
            speedup_second_best = ""
            for format in self.formats:
                if (
                    "Binary" in format
                    or "Symmetric" in format
                    or "cusparse" in format
                    or "Hypre" in format
                    or "Ginkgo" in format
                ):
                    continue
                for device in ["CPU"] + self.accelerator_devices:
                    if (format, device) not in self.launch_configs:
                        continue
                    for launch_config in self.launch_configs[(format, device)]:
                        col = (format, device, launch_config, "bandwidth", "")
                        if col in self.df.columns:
                            bw = row[col]
                            if bw is not None and bw > max_bandwidth[0]:
                                max_bandwidth[1] = max_bandwidth[0]
                                best_format[1] = best_format[0]
                                best_device[1] = best_device[0]
                                best_launch_config[1] = best_launch_config[0]
                                time[1] = time[0]
                                speedup_csr_cpu[1] = speedup_csr_cpu[0]
                                speedup_cusparse[1] = speedup_cusparse[0]
                                max_bandwidth[0] = bw
                                best_format[0] = format
                                best_device[0] = device
                                best_launch_config[0] = launch_config
                                time[0] = row[
                                    (format, device, launch_config, "time mean", "")
                                ]
                                if time[0] != "" and time[1] != "":
                                    speedup_second_best = time[1] / time[0]
                                if (
                                    "cusparse" in self.formats
                                    and device != "CPU"
                                    and ("cusparse", device) in self.launch_configs
                                ):
                                    speedup_cusparse[0] = (
                                        max_bandwidth[0]
                                        / row[
                                            (
                                                "cusparse",
                                                device,
                                                "Default",
                                                "bandwidth",
                                                "",
                                            )
                                        ]
                                    )
                                if (
                                    "CSR",
                                    "CPU",
                                ) in self.launch_configs and device != "CPU":
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
            format_list[0].append(best_format[0])
            device_list[0].append(best_device[0])
            launch_config_list[0].append(best_launch_config[0])
            bandwidth_list[0].append(max_bandwidth[0])
            time_list[0].append(time[0])
            speedup_csr_cpu_list[0].append(speedup_csr_cpu[0])
            speedup_cusparse_list[0].append(speedup_cusparse[0])
            speedup_second_best_list.append(speedup_second_best)
            format_list[1].append(best_format[1])
            device_list[1].append(best_device[1])
            launch_config_list[1].append(best_launch_config[1])
            bandwidth_list[1].append(max_bandwidth[1])
            time_list[1].append(time[1])
            speedup_csr_cpu_list[1].append(speedup_csr_cpu[1])
            speedup_cusparse_list[1].append(speedup_cusparse[1])
        # 1st best
        self.best_tnl_df[("TNL Best", "format", "", "")] = format_list[0]
        self.best_tnl_df[("TNL Best", "device", "", "")] = device_list[0]
        self.best_tnl_df[("TNL Best", "launch cfg.", "", "")] = launch_config_list[0]
        self.best_tnl_df[("TNL Best", "bandwidth", "", "")] = bandwidth_list[0]
        self.best_tnl_df[("TNL Best", "time", "", "")] = time_list[0]
        self.best_tnl_df[("TNL Best", "speed-up", "CSR CPU", "")] = speedup_csr_cpu_list[0]
        self.best_tnl_df[("TNL Best", "speed-up", "cusparse", "")] = speedup_cusparse_list[
            0
        ]
        self.best_tnl_df[("TNL Best", "speed-up", "2nd best", "")] = (
            speedup_second_best_list
        )
        # 2nd best
        print("2nd best speed-ups:")
        self.best_tnl_df[("TNL 2nd Best", "format", "", "")] = format_list[1]
        self.best_tnl_df[("TNL 2nd Best", "device", "", "")] = device_list[1]
        self.best_tnl_df[("TNL 2nd Best", "launch cfg.", "", "")] = launch_config_list[1]
        self.best_tnl_df[("TNL 2nd Best", "bandwidth", "", "")] = bandwidth_list[1]
        self.best_tnl_df[("TNL 2nd Best", "time", "", "")] = time_list[1]
        self.best_tnl_df[("TNL 2nd Best", "speed-up", "CSR CPU", "")] = (
            speedup_csr_cpu_list[1]
        )
        self.best_tnl_df[("TNL 2nd Best", "speed-up", "cusparse", "")] = (
            speedup_cusparse_list[1]
        )

    def get_total_best_format(self):
        """
        Get the best format of all
        """
        format_list = [[], []]
        device_list = [[], []]
        launch_config_list = [[], []]
        bandwidth_list = [[], []]
        time_list = [[], []]
        diff_max_list = [[], []]
        speedup_csr_cpu_list = [[], []]
        speedup_cusparse_list = [[], []]
        speedup_second_best_list = []
        for idx, row in self.df.iterrows():
            max_bandwidth = [float("-inf"), float("-inf")]
            best_format = [None, None]
            best_launch_config = [None, None]
            best_device = [None, None]
            time = [0, 0]
            speedup_second_best = None
            for format in self.formats:
                if "Binary" in format or "Symmetric" in format:
                    continue
                for device in ["CPU"] + self.accelerator_devices:
                    if (format, device) not in self.launch_configs:
                        continue
                    for launch_config in self.launch_configs[(format, device)]:
                        col = (format, device, launch_config, "bandwidth", "")
                        if col in self.df.columns:
                            bw = row[col]
                            if bw is not None and bw > max_bandwidth[0]:
                                max_bandwidth[1] = max_bandwidth[0]
                                best_format[1] = best_format[0]
                                best_device[1] = best_device[0]
                                best_launch_config[1] = best_launch_config[0]
                                time[1] = time[0]
                                max_bandwidth[0] = bw
                                best_format[0] = format
                                best_device[0] = device
                                best_launch_config[0] = launch_config
                                time[0] = row[
                                    (format, device, launch_config, "time mean", "")
                                ]
                                if time[0] != "" and time[1] != "":
                                    speedup_second_best = time[1] / time[0]
            # 1st best
            format_list[0].append(best_format[0])
            device_list[0].append(best_device[0])
            launch_config_list[0].append(best_launch_config[0])
            bandwidth_list[0].append(max_bandwidth[0])
            time_list[0].append(time[0])
            speedup_second_best_list.append(speedup_second_best)
            # 2nd best
            format_list[1].append(best_format[1])
            device_list[1].append(best_device[1])
            launch_config_list[1].append(best_launch_config[1])
            bandwidth_list[1].append(max_bandwidth[1])
            time_list[1].append(time[1])

        # 1st best
        self.best_total_df[("Total Best", "format", "", "")] = format_list[0]
        self.best_total_df[("Total Best", "device", "", "")] = device_list[0]
        self.best_total_df[("Total Best", "launch cfg.", "", "")] = launch_config_list[0]
        self.best_total_df[("Total Best", "bandwidth", "", "")] = bandwidth_list[0]
        self.best_total_df[("Total Best", "time", "", "")] = time_list[0]
        self.best_total_df[("Total Best", "speed-up", "2nd best", "")] = (
            speedup_second_best_list
        )
        # 2nd best
        self.best_total_df[("Total 2nd Best", "format", "", "")] = format_list[1]
        self.best_total_df[("Total 2nd Best", "device", "", "")] = device_list[1]
        self.best_total_df[("Total 2nd Best", "launch cfg.", "", "")] = launch_config_list[1]
        self.best_total_df[("Total 2nd Best", "bandwidth", "", "")] = bandwidth_list[1]
        self.best_total_df[("Total 2nd Best", "time", "", "")] = time_list[1]

    def write(self):
        self.get_best_csr()
        self.get_best_tnl_format()
        self.get_total_best_format()
        print("Writing best formats to best_formats.html")
        self.best_csr_df.to_html("best_csr_formats.html")
        self.best_tnl_df.to_html("best_tnl_formats.html")
        self.best_total_df.to_html("best_total_formats.html")

    def count_best_formats(self, file_name):
        """
        Count best formats and print them to the console
        """
        best_formats = {}
        best_sorted = 0
        best_accelerator = 0
        total = 0
        for idx, row in self.best_total_df.iterrows():
            best_format = row[("Total Best", "format", "", "")]
            best_device = row[("Total Best", "device", "", "")]
            best_launch_config = row[("Total Best", "launch cfg.", "", "")]
            if not best_format in best_formats:
                best_formats[best_format] = 0
            best_formats[best_format] += 1
            if "Sorted" in best_format:
                best_sorted += 1
            total += 1
            if best_device != "CPU":
                best_accelerator += 1

        with open(file_name, "w") as f:
            f.write("=======================================================\n")
            f.write("Best formats summary\n")
            f.write("=======================================================\n")
            f.write(f"Total matrices: {total}\n")
            f.write(f"Best on accelerator: {best_accelerator}\n")
            f.write("Best formats:\n")
            for format, count in best_formats.items():
                f.write(f" - {format}: {count}\n")
            f.write(f"Best sorted formats: {best_sorted}\n")
        with open(file_name, "r") as f:
            print(f.read())
