import shutil
from os.path import join, basename


def create_new_paths(paths, source_dir):
    # copy desired audio files to a new directory
    # this function assumes that the label subdirectories have already been created
    # this structure is based on what Scaper needs
    for path in paths:
        # get label
        label = path.split("/")[-2]
        # move to corresponding folder in new source dir
        shutil.copyfile(path, join(source_dir, label, basename(path)))


def get_class_assignments(variant_id, vocab_idx):
    """
    Return class assignments to KK, KU and UU categories

    Params
    -------
    variant_id: dataset variant number (1-5)
    vocab_idx: shuffled class indices

    Returns
    -------
    If low openness: return kk_idx, ku_idx and uu_idx
    If high openness: return kk_idx and uu_idx
    """

    # get kk, ku and uu classes
    uu_idx = (
        vocab_idx[(variant_id - 1) * 18 : variant_id * 18]
        if variant_id != 5
        else vocab_idx[4 * 18 :]
    )
    ku_idx = (
        vocab_idx[(variant_id % 5) * 18 : ((variant_id + 1) % 5) * 18]
        if variant_id != 4
        else vocab_idx[4 * 18 :]
    )
    kk_idx = list(set(vocab_idx) - set(uu_idx) - set(ku_idx))

    return kk_idx, ku_idx, uu_idx


def get_source_path_splits(
    source_paths,
    known_classes,
    unknown_classes,
    train_frac=0.8,
    val_frac=0.1,
):
    """
    Return the train, val and test split paths
    For openness='high', ku_idx should be an empty list.

    Params
    -------
    source_paths: List of source wav file paths
    kk_idx, uu_idx, kk_idx: List of class indices in each group
    train_frac, val_frac: Fraction of source paths reserved for each split

    Returns
    -------
    train_paths, val_paths, test_paths
    """

    # Get train, val and test based on class distributions
    uu_paths = []
    # get uu paths first
    for path in source_paths:
        if any(str(x) == path.split("/")[-2] for x in unknown_classes):
            uu_paths.append(path)

    # collect paths by class
    non_uu_paths = []
    for c in known_classes:
        # grab all files belonging to the class
        class_paths = [path for path in source_paths if str(c) == path.split("/")[-2]]
        non_uu_paths.append(class_paths)

    # split class_paths to get train, val, test splits
    train_paths, val_paths, test_paths = [], [], []
    for class_paths in non_uu_paths:
        n_sources_in_class = len(class_paths)
        n_train_sources = int(train_frac * n_sources_in_class)
        n_val_sources = int(val_frac * n_sources_in_class)

        train_paths.append(class_paths[:n_train_sources])
        val_paths.append(class_paths[n_train_sources : n_train_sources + n_val_sources])
        test_paths.append(class_paths[n_train_sources + n_val_sources :])

    # Flatten lists
    train_paths = [item for sublist in train_paths for item in sublist]
    val_paths = [item for sublist in val_paths for item in sublist]
    test_paths = [item for sublist in test_paths for item in sublist] + uu_paths

    assert all([int(f.split("/")[-2]) in known_classes for f in train_paths])
    assert all([int(f.split("/")[-2]) in known_classes for f in val_paths])

    return train_paths, val_paths, test_paths
