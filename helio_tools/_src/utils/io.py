import glob, os
from dateutil.parser import parse

def get_list_filenames(data_path: str="./", ext: str="*"):
    """Loads a list of file names within a directory
    """
    pattern = f"*{ext}"
    return sorted(glob.glob(os.path.join(data_path, "**", pattern), recursive=True))


def get_intersecting_files(path, dirs, months=None, years=None, n_samples=None, ext=None, basenames=None, **kwargs):
    pattern = '*' if ext is None else '*' + ext
    if basenames is None:
        basenames = [[os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)] for d in dirs]
        basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('_')[1]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse(bn.split('_')[1]).year in years]
    basenames = sorted(list(basenames))
    if n_samples:
        basenames = basenames[::len(basenames) // n_samples]
    return [[os.path.join(path, str(dir), b) for b in basenames] for dir in dirs]