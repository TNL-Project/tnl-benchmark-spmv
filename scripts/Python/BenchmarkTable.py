"""
Binary (pickle) storage of the complete benchmark table, including all
computed speed-ups, together with the metadata needed to analyze it.

Written by tnl-spmv-benchmark-parse-logs.py and read by
tnl-spmv-benchmark-make-tables-json.py.
"""

import pickle


def save_table(file_name, df, formats, launch_configs, accelerator_devices):
    """
    Store the complete table together with the metadata needed by the
    analysis in a binary file, so that it can be loaded by load_table()
    """
    data = {
        "df": df,
        "formats": formats,
        "launch_configs": launch_configs,
        "accelerator_devices": accelerator_devices,
    }
    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Complete table written to {file_name}")


def load_table(file_name):
    """
    Load the complete table and its metadata stored by save_table()
    """
    print(f"Loading complete table from {file_name} ...")
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data["df"], data["formats"], data["launch_configs"], data["accelerator_devices"]


def print_formats_and_launch_configs(formats, launch_configs, accelerator_devices):
    """
    Print formats and their launch configurations.
    """
    print(f"Formats: {formats}")
    for format in formats:
        for device in ["CPU"] + accelerator_devices:
            if (format, device) in launch_configs:
                print(
                    f"Launch configs for {format} on {device}: {launch_configs[(format, device)]}"
                )
