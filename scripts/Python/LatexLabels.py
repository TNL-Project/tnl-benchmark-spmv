def latex_label(label):
    label = label.replace("<", "")
    label = label.replace(">", "")
    label = label.replace("Light  Automatic ", "")
    # print( f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{name}~~~')
    if label == "CSR":
        return "CSR on CPU"
    if label == "cusparse":
        return "cuSPARSE"
    if "SlicedEllpack SlicedEllpack" in label:
        return label.replace("SlicedEllpack SlicedEllpack", "Sliced Ellpack")
    if "ChunkedEllpack ChunkedEllpack" in label:
        return label.replace("ChunkedEllpack ChunkedEllpack", "Chunked Ellpack")
    if "Ellpack Ellpack" in label:
        return label.replace("Ellpack Ellpack", "Ellpack")
    if "BiEllpack BiEllpack" in label:
        return label.replace("BiEllpack BiEllpack", "Bisection Ellpack")
    if "CSR Scalar" in label:
        return label.replace("CSR Scalar", "Scalar CSR")
    if "CSR Vector" in label:
        return label.replace("CSR Vector", "Vector CSR")
    # if "CSR Light" in label:
    #    return label.replace("CSR Light", "Light CSR")
    if "CSR Adaptive" in label:
        return label.replace("CSR Adaptive", "Adaptive CSR")
    if "CSR Light Automatic Light" in label:
        return label.replace("CSR Light Automatic Light", "Light CSR")
    return label
