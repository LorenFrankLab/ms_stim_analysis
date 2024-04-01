import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

animal_style = [
    {"name": "Winnie", "color": "darkorange", "alias": "Rat 1"},
    {"name": "Frodo", "color": "goldenrod", "alias": "Rat 2"},
    {"name": "Totoro", "color": "sandybrown", "alias": "Rat 3"},
    {"name": "Banner", "color": "coral", "alias": "Rat 4"},
    # controls
    {"name": "Wallie", "color": "slategrey", "alias": "Rat 5"},
    {"name": "Olive", "color": "steelblue", "alias": "Rat 6"},
    {"name": "Yoshi", "color": "cadetblue", "alias": "Rat 7"},
    {"name": "Bilbo", "color": "mediumslateblue", "alias": "Rat 8"},
]
animal_style = pd.DataFrame(animal_style)
animal_style.set_index("name", inplace=True)

interval_style = {"test": "fuchsia", "control": "g"}

transfection_style = {"transfected": "indianred", "control": "grey"}


def style_sample():
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    x = np.linspace(0, 1, 50)
    for i, animal in enumerate(animal_style.index):
        ax[0].plot(
            x,
            np.random.normal(i, 0.3, x.size),
            color=animal_style.loc[animal, "color"],
            label=animal_style.loc[animal, "alias"],
        )
    # put int transfection_style
    ax[0].plot(
        x,
        np.random.normal(-1, 0.3, x.size),
        color=transfection_style["transfected"],
        label="transfected",
    )
    ax[0].plot(
        x,
        np.random.normal(i + 1, 0.3, x.size),
        color=transfection_style["control"],
        label="control",
    )

    ax[0].set_title("Animals")
    ax[0].set_yticks(
        np.arange(-1, i + 2),
        ["transfected (grouped)"] + list(animal_style.index) + ["control (grouped)"],
    )
    ax[0].set_xticks([])

    for i, interval in enumerate(interval_style):
        ax[1].plot(
            x,
            np.random.normal(i, 0.3, x.size),
            color=interval_style[interval],
            label=interval,
        )
    ax[1].set_title("Optogenetic intervals")
    ax[1].set_yticks(np.arange(i + 1), interval_style.keys())
    return fig
