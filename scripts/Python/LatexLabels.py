def latex_label(label):
    label = label.replace("<", "")
    label = label.replace(">", "")
    label = label.replace("Light  Automatic ", "")
    # print( f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{name}~~~')
    if label == "CSR":
        return "CSR on CPU"
    if label == "cusparse":
        return "cuSPARSE"
    if "SlicedEllpack" in label:
        return label.replace("SlicedEllpack", "Sliced Ellpack")
    if "ChunkedEllpack" in label:
        return label.replace("ChunkedEllpack", "Chunked Ellpack")
    if "BiEllpack" in label:
        return label.replace("BiEllpack", "Bisection Ellpack")
    if "CSR Scalar" in label:
        return label.replace("CSR Scalar", "Scalar CSR")
    if "CSR Vector" in label:
        return label.replace("CSR Vector", "Vector CSR")
    if "CSR Light" in label:
        return label.replace("CSR Light", "Light CSR")
    if "CSR Adaptive" in label:
        return label.replace("CSR Adaptive", "Adaptive CSR")
    return label
