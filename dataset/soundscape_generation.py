from collections import Counter
import scaper
import numpy as np

SEED = 123  # To reproduce OST as in the paper, do not update this


def generate_without_audio(
    sc, jamsfile, allow_repeated_label=False, allow_repeated_source=False
):
    # sc : Scaper object with events already added
    # jamsfile : path to JAMS file where the soundscape annotation is saved

    sc.generate(
        audio_path=None,
        jams_path=jamsfile,
        allow_repeated_label=allow_repeated_label,
        allow_repeated_source=allow_repeated_source,
        reverb=0,
        disable_sox_warnings=True,
        no_audio=True,
        txt_path=None,
        fix_clipping=True,
        disable_instantiation_warnings=False,
    )


def create_soundscape(
    sc,
    paths,
    source_counts,
    class_id=None,
    snr_min=-5,
    snr_max=20,
    pitch_shift_min=-2.0,
    pitch_shift_max=2.0,
    time_stretch_min=0.8,
    time_stretch_max=1.2,
    add_bg=True,
):
    # sc : Scaper object
    # paths : source paths
    # source_counts : Counter object containing occurrence counts of classes
    # class_id : class label
    # This function is intended to add an event based on the train, val or test split
    # The args should be adjusted accordingly
    # Returns : Scaper object with added events

    sc.reset_fg_event_spec()
    sc.reset_bg_event_spec()

    if add_bg:
        sc.add_background(
            label=("const", "brownnoise"),
            source_file=("choose", []),
            source_time=("const", 0),
        )

    p = np.array([1.0 / (i + 1) for i in range(4)])  # p(n) = k x 1/n
    n_events = np.random.choice(a=[1, 2, 3, 4], size=1, replace=False, p=p / sum(p))
    class_idx = []

    if class_id is not None:
        sc.add_event(
            label=("const", str(class_id)),
            source_file=(
                "choose",
                [path for path in paths if path.split("/")[-2] == str(class_id)],
            ),
            source_time=("uniform", 0, 4),
            event_time=("uniform", 0, 9),
            event_duration=("uniform", 0.5, 4),
            snr=("uniform", snr_min, snr_max),
            pitch_shift=("uniform", pitch_shift_min, pitch_shift_max),
            time_stretch=("uniform", time_stretch_min, time_stretch_max),
        )

        # new dict to eliminate label repetition
        new_source_counts = Counter(
            {k: v for (k, v) in source_counts.items() if k != str(class_id)}
        )
        if n_events != 1:
            class_idx = (
                np.random.choice(  # TODO : would scaper weighted choice be better?
                    a=list(new_source_counts.keys()),
                    size=n_events - 1,
                    replace=False,
                    p=np.array(list(new_source_counts.values()))
                    / sum(new_source_counts.values()),
                )
            )

    else:
        # Inherent stochasticity here
        class_idx = np.random.choice(  # TODO : would scaper weighted choice be better?
            a=list(source_counts.keys()),
            size=n_events,
            replace=False,
            p=np.array(list(source_counts.values())) / sum(source_counts.values()),
        )

    for c in class_idx:
        sc.add_event(
            label=("const", str(c)),
            source_file=(
                "choose",
                [path for path in paths if path.split("/")[-2] == str(c)],
            ),
            source_time=("uniform", 0, 4),
            event_time=("uniform", 0, 9),
            event_duration=("uniform", 0.5, 4),
            snr=("uniform", snr_min, snr_max),
            pitch_shift=("uniform", pitch_shift_min, pitch_shift_max),
            time_stretch=("uniform", time_stretch_min, time_stretch_max),
        )

    return sc


def choose_n_events(p_min=1, p_max=4):
    p_range = [i for i in range(p_min, p_max + 1)]
    p = np.array([1.0 / i for i in p_range])  # p(n) = k x 1/n
    n_events = np.random.choice(a=p_range, replace=False, p=p / sum(p))

    return n_events


def choose_labels_for_soundscape(labels, n_events=None):
    # TODO: update p and n_events for bg only examples, maybe the same likelihood as polyphony 4

    if n_events is None:
        n_events = choose_n_events()
    sc_labels = []

    sc_labels = np.random.choice(
        a=labels,
        size=n_events,
        replace=False,
    )

    return sc_labels.tolist()


def add_events_to_sc(
    sc,
    sc_labels,
    paths,
    snr_min,
    snr_max,
    pitch_shift_min,
    pitch_shift_max,
    time_stretch_min,
    time_stretch_max,
    event_start_min=0,
    event_start_max=9,
    source_start_min=0,
    source_start_max=4,
    event_duration_min=0.5,
    event_duration_max=4,
):
    for c in sc_labels:
        sc.add_event(
            label=("const", str(c)),
            source_file=(
                "choose",
                [path for path in paths if path.split("/")[-2] == str(c)],
            ),
            source_time=("uniform", source_start_min, source_start_max),
            event_time=("uniform", event_start_min, event_start_max),
            event_duration=("uniform", event_duration_min, event_duration_max),
            snr=("uniform", snr_min, snr_max),
            pitch_shift=("uniform", pitch_shift_min, pitch_shift_max),
            time_stretch=("uniform", time_stretch_min, time_stretch_max),
        )
    return sc


def oss_tiny_soundscape(
    sc,
    paths,
    labels,
    allowed_combos=None,
    snr_min=-5,
    snr_max=20,
    pitch_shift_min=-2.0,
    pitch_shift_max=2.0,
    time_stretch_min=0.8,
    time_stretch_max=1.2,
    add_bg=False,
):
    # sc : Scaper object
    # paths : source paths
    # labels : allowed labels
    # allowed_combos : dictionary of allowed class combinations indexed by 'px', x the polyphony
    # class_id : class label
    # This function is intended to add an event based on the train, val or test split
    # The args should be adjusted accordingly
    # Returns : Scaper object with added events

    sc.reset_fg_event_spec()
    sc.reset_bg_event_spec()

    if add_bg:
        # TODO: update to sonyc backgrounds
        sc.add_background(
            label=("const", "brownnoise"),
            source_file=("choose", []),
            source_time=("const", 0),
        )

    if allowed_combos is None:
        sc_labels = choose_labels_for_soundscape(labels)
    else:
        n_events = choose_n_events()
        if n_events == 1:
            sc_labels = choose_labels_for_soundscape(labels, n_events=1)
        else:
            combo_choice = np.random.choice(
                [i for i in range(len(allowed_combos[f"p{n_events}"]))]
            )
            sc_labels = allowed_combos[f"p{n_events}"][combo_choice]

    sc = add_events_to_sc(
        sc,
        sc_labels,
        paths,
        snr_min,
        snr_max,
        pitch_shift_min,
        pitch_shift_max,
        time_stretch_min,
        time_stretch_max,
    )

    return sc, sc_labels


def oss_tiny_val_or_test_soundscape(
    sc,
    paths,
    kk_labels,
    uu_labels,
    seen_kk_combos,
    unseen_kk_combos,
    snr_min=-5,
    snr_max=20,
    pitch_shift_min=-2.0,
    pitch_shift_max=2.0,
    time_stretch_min=0.8,
    time_stretch_max=1.2,
    add_bg=False,
    debug=False,
):
    sc.reset_fg_event_spec()
    sc.reset_bg_event_spec()

    if add_bg:
        # TODO: update to sonyc backgrounds
        sc.add_background(
            label=("const", "brownnoise"),
            source_file=("choose", []),
            source_time=("const", 0),
        )

    if uu_labels is not None:
        sc_type = np.random.choice(["kk", "uu"])
    else:  # validation set behavior
        sc_type = "kk"

    if sc_type == "kk":
        kk_combo = np.random.choice(["seen", "unseen"])

        if kk_combo == "seen":
            # pick polyphony from a uniform distribution
            n_events = np.random.choice([1, 2, 3, 4])
            if n_events == 1:
                sc_labels = choose_labels_for_soundscape(kk_labels, n_events=1)
            else:
                combo_choice = np.random.choice(
                    [i for i in range(len(seen_kk_combos[f"p{n_events}"]))]
                )
                sc_labels = seen_kk_combos[f"p{n_events}"][combo_choice]

        else:
            # uniform distribution excluding polyphony 1, which is seen during training
            n_events = np.random.choice([2, 3, 4])
            combo_choice = np.random.choice(
                [i for i in range(len(unseen_kk_combos[f"p{n_events}"]))]
            )
            sc_labels = unseen_kk_combos[f"p{n_events}"][combo_choice]

    else:
        n_events = choose_n_events()

        # ensure one event is uu
        uu_label = choose_labels_for_soundscape(uu_labels, n_events=1)

        if n_events > 1:
            allowed_labels = kk_labels + uu_labels
            allowed_labels.remove(uu_label[0])
            labels = choose_labels_for_soundscape(allowed_labels, n_events=n_events - 1)
            sc_labels = uu_label + labels

        else:
            sc_labels = uu_label
    if debug:
        return sc_labels, sc_type, kk_combo if "kk_combo" in locals() else None

    sc = add_events_to_sc(
        sc,
        sc_labels,
        paths,
        snr_min,
        snr_max,
        pitch_shift_min,
        pitch_shift_max,
        time_stretch_min,
        time_stretch_max,
    )
    return sc, sc_type, kk_combo if "kk_combo" in locals() else ""


def unit_test_oss_tiny_test_soundscape(
    seen_combos,
    unseen_combos,
    n_soundscapes=10_000,
    n_kk_classes=15,
    uu_end_idx=89,
    uu_start_idx=54,
    kk_labels=None,
    uu_labels=None,
):
    kk, unseen, seen, uu = 0, 0, 0, 0
    if uu_labels is None:
        uu_labels = [i for i in range(uu_start_idx, uu_end_idx)]
    if kk_labels is None:
        kk_labels = [i for i in range(n_kk_classes)]

    for i in range(n_soundscapes):
        sc_labels = oss_tiny_val_or_test_soundscape(
            sc=None,
            paths=None,
            kk_labels=kk_labels,
            uu_labels=uu_labels,
            seen_kk_combos=seen_combos,
            unseen_kk_combos=unseen_combos,
            debug=True,
        )
        if sc_labels[1] == "kk":
            kk += 1
            assert all([l in kk_labels for l in sc_labels[0]])
            if sc_labels[2] == "seen":
                seen += 1
                assert tuple(sc_labels[0]) in seen_combos
            else:
                unseen += 1
                assert tuple(sc_labels[0]) in unseen_combos
        else:
            uu += 1
            assert any([l in uu_labels for l in sc_labels[0]])

    print(f"Tested {n_soundscapes} soundscapes: ")
    print(
        f"{kk} were kk, with {seen} seen kk class combos and {unseen} unseen kk class combos"
    )
    print(f"{uu} were uu")
    print("Tests passed successfully!")
