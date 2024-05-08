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


def compute_csr_speedup(df, formats, formats_devices):
    """
    Compute speed-up of particular formats compared to CSR on CPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("CSR", "CPU", "1 threads", "time"),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "CSR CPU"),
                    )


def compute_cusparse_speedup(df, formats, formats_devices):
    """
    Compute speed-up of particular formats compared to Cusparse on GPU and (possibly) Hypre on GPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("cusparse", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "cusparse"),
                    )


def compute_hypre_speedup(df, formats, formats_devices):
    """
    Compute speed-up of particular formats compared to Hypre on GPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR", "Hypre"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("Hypre", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "Hypre"),
                    )


def compute_ginkgo_speedup(df, formats, formats_devices):
    """
    Compute speed-up of particular formats compared to Ginkgo on GPU and CSR on CPU
    """
    for device in ["GPU"]:
        for format in formats:
            if not format in ["cusparse", "CSR", "Ginkgo"]:
                if (format, device) in formats_devices:
                    divide_columns(
                        df,
                        ("Ginkgo", "GPU", "time", ""),
                        (format, device, "time", ""),
                        (format, device, "speed-up", "Ginkgo"),
                    )

    # TODO: Compute speedup of Ginkgo CSR on CPU compared to CSR on CPU


def compute_csr_light_speedup(df, formats):
    if "LightSpMV Vector" in formats:
        for light in ["CSR Light Automatic", "CSR Light Automatic Light"]:
            if light in formats:
                if light in formats:
                    divide_columns(
                        df,
                        (light, "GPU", "bandwidth", ""),
                        ("LightSpMV Vector", "GPU", "bandwidth", ""),
                        (light, "GPU", "speed-up", "LightSpMV Vector"),
                    )


def compute_binary_speedup(df, formats, formats_devices):
    for format in formats:
        if "Binary" in format and (format, "GPU") in formats_devices:
            non_binary_format = format.replace("Binary ", "")
            print(f"Adding speed-up of {format} vs {non_binary_format}")
            divide_columns(
                df,
                (format, "GPU", "time"),
                (non_binary_format, "GPU", "time"),
                (format, "GPU", "speed-up", "non-binary"),
            )


def compute_symmetric_speedup(df, formats, formats_devices):
    for format in formats:
        if "Symmetric" in format:
            if (format, "GPU") in formats_devices:
                non_symmetric_format = format.replace("Symmetric ", "")
                print(f"Adding speed-up of {format} vs {non_symmetric_format}")
                divide_columns(
                    df,
                    (non_symmetric_format, "GPU", "time"),
                    (format, "GPU", "time"),
                    (format, "GPU", "speed-up", "non-symmetric"),
                )


def compute_cpu_speedup(df, formats, threads_num_list):
    for format in ["CSR", "Hypre"]:
        if format in formats:
            sequential_times_list = df[(format, "CPU", "1 threads", "time")]
            speedup_list = []
            efficiency_list = []
            for threads in threads_num_list:
                if threads == 1:
                    continue
                parallel_times_list = df[
                    (format, "CPU", str(threads) + " threads", "time")
                ]
                # print( sequential_times_list)
                # print( parallel_times_list)
                for sequential_time, parallel_time in zip(
                    sequential_times_list, parallel_times_list
                ):
                    # print( sequential_time )
                    # print( parallel_time )
                    speedup = float(sequential_time) / float(parallel_time)
                    efficiency = speedup / threads
                    speedup_list.append(speedup)
                    efficiency_list.append(efficiency)
                df[(format, "CPU", str(threads) + " threads", "speed-up")] = (
                    speedup_list
                )
                df[(format, "CPU", str(threads) + " threads", "eff.")] = efficiency_list
                speedup_list.clear()
                efficiency_list.clear()


def compute_hypre_cpu_speedup(df, formats, threads_num_list):
    for threads in threads_num_list:
        divide_columns(
            df,
            ("Hypre", "CPU", str(threads) + " threads", "time"),
            ("CSR", "CPU", str(threads) + " threads", "time"),
            ("Hypre", "CPU", str(threads) + " threads", "TNL speed-up"),
        )


def compute_speedup(df, formats, threads_num_list, formats_devices):
    compute_csr_speedup(df, formats, formats_devices)
    if "cusparse" in formats:
        compute_cusparse_speedup(df, formats, formats_devices)
    if "Hypre" in formats:
        compute_hypre_speedup(df, formats, formats_devices)
    if "Ginkgo" in formats:
        compute_ginkgo_speedup(df, formats, formats_devices)
    compute_csr_light_speedup(df, formats)
    compute_binary_speedup(df, formats, formats_devices)
    compute_symmetric_speedup(df, formats, formats_devices)
    compute_cpu_speedup(df, formats, threads_num_list)
    if ("Hypre", "CPU") in formats_devices:
        compute_hypre_cpu_speedup(df, formats, threads_num_list)
