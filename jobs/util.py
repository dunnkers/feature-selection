def getDatasetNo(filename):
    return filename.split('_')[-1:][0].split('.')[0]