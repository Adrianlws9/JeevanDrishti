import netCDF4 as nc, pandas as pd
D = r"C:\Users\Janhvi Suryavanshi\Downloads"
for f in ["imdweb_5dgbAskl.nc", "imdweb_Oe9H04Bg.nc", "imdweb_tXexQbB6.nc"]:
    ds = nc.Dataset(D + "\\" + f)
    print("==", f, ds.ncattrs())
    def walk(g, p=""):
        for k, v in g.variables.items():
            print(p, k, v.dimensions, v.shape, getattr(v, "units", None), getattr(v, "long_name", None), v.dtype)
        for n, sg in g.groups.items():
            print("group", n); walk(sg, p + "  ")
    walk(ds)
h = pd.read_csv(D + r"\humid_tel_hr_cwprs_mh_2021_2025.csv", nrows=5)
print(h.T)
e = pd.read_csv(D + r"\maharashtra_epiclim_filtered.csv")
print(e.head(3).T); print(e.Disease.value_counts()); print(e.dtypes)
print(sorted(e.district.unique()))
