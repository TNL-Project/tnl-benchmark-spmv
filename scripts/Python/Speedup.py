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


def compute_csr_cpu_speedup(df, formats, threads_num_list):
    """
    Compute speed-up and efficiency of the CSR format on CPU with different number of threads
    """
    for format in ["CSR"]:
        if format in formats:
            for threads in threads_num_list:
                if threads == 1:
                    continue
                divide_columns(
                    df,
                    (format, "CPU", "1 threads", "time"),
                    (format, "CPU", str(threads) + " threads", "time"),
                    (format, "CPU", str(threads) + " threads", "speed-up"),
                )
                divide_column_by_number(
                    df,
                    (format, "CPU", str(threads) + " threads", "speed-up"),
                    threads,
                    (format, "CPU", str(threads) + " threads", "eff."),
                )


def compute_cusparse_speedup(df, formats, formats_devices):
    """
    Compute speed-up of particular formats compared to Cusparse on GPU and (possibly) Hypre on GPU
    """
    if "cusparse" in formats:
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


def compute_hypre_and_ginkgo_speedup(df, formats, formats_devices, threads_num_list):
    """
    Compute speed-up of particular formats compared to Hypre and Ginkgo on GPU and CSR on CPU
    """
    for ref_format in ["Hypre", "Ginkgo"]:
        if ref_format in formats:
            for device in ["GPU"]:
                for format in formats:
                    if not format in ["cusparse", "CSR", "Ginkgo", "Hypre"]:
                        if (format, device) in formats_devices:
                            divide_columns(
                                df,
                                (ref_format, "GPU", "time", ""),
                                (format, device, "time", ""),
                                (format, device, "speed-up", ref_format),
                            )

            for threads in threads_num_list:
                divide_columns(
                    df,
                    (ref_format, "CPU", f"{threads} threads", "time"),
                    ("CSR", "CPU", f"{threads} threads", "time"),
                    (ref_format, "CPU", f"{threads} threads", "TNL speed-up"),
                )
                if threads != 1:
                    divide_columns(
                        df,
                        (ref_format, "CPU", "1 threads", "time"),
                        (ref_format, "CPU", f"{threads} threads", "time"),
                        (ref_format, "CPU", f"{threads} threads", "speed-up"),
                    )
                    divide_column_by_number(
                        df,
                        (ref_format, "CPU", f"{threads} threads", "speed-up"),
                        threads,
                        (ref_format, "CPU", f"{threads} threads", "eff."),
                    )


def compute_csr_light_speedup(df, formats):
    """
    Compute speed-up of CSR Light Automatic and CSR Light Automatic Light compared to LightSpMV Vector
    """
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
    """
    Compute speed-up of Binary formats compared to their non-binary counterparts
    """
    for format in formats:
        if "Binary" in format and (format, "GPU") in formats_devices:
            non_binary_format = format.replace("Binary ", "")
            print(f"Adding speed-up of {format} vs {non_binary_format}")
            divide_columns(
                df,
                (non_binary_format, "GPU", "time"),
                (format, "GPU", "time"),
                (format, "GPU", "speed-up", "non-binary"),
            )


def compute_symmetric_speedup(df, formats, formats_devices):
    """
    Compute speed-up of Symmetric formats compared to their non-symmetric counterparts
    """
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


def compute_legacy_speedup(df, formats, formats_devices, legacy_counterparts):
    """
    Compute speed-up of formats compared to their legacy counterparts
    """
    for format in formats:
        if not legacy_counterparts.get(format):
            continue
        legacy_format = legacy_counterparts[format]
        if legacy_format in formats:
            if (format, "GPU") in formats_devices and (
                legacy_format,
                "GPU",
            ) in formats_devices:
                divide_columns(
                    df,
                    (legacy_format, "GPU", "time"),
                    (format, "GPU", "time"),
                    (format, "GPU", "speed-up", legacy_format),
                )
            if (format, "CPU") in formats_devices and (
                legacy_format,
                "CPU",
            ) in formats_devices:
                divide_columns(
                    df,
                    (legacy_format, "CPU", "time"),
                    (format, "CPU", "time"),
                    (format, "CPU", "speed-up", legacy_format),
                )


def compute_speedup(
    df, formats, threads_num_list, formats_devices, legacy_counterparts
):
    compute_csr_cpu_speedup(df, formats, threads_num_list)
    compute_cusparse_speedup(df, formats, formats_devices)
    compute_hypre_and_ginkgo_speedup(df, formats, formats_devices, threads_num_list)
    compute_csr_light_speedup(df, formats)
    compute_binary_speedup(df, formats, formats_devices)
    compute_symmetric_speedup(df, formats, formats_devices)
    compute_legacy_speedup(df, formats, formats_devices, legacy_counterparts)
