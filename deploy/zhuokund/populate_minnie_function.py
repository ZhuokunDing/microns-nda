from microns_nda.minnie_nda import minnie_function

def main():
    minnie_function.OrientationDV11521GD.fill()
    minnie_function.Orientation.fill()
    minnie_function.OrientationScanInfo.populate()

if __name__ == "__main__":
    main()