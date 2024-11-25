import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

transfection_style = {"transfected": "indianred", "control": "grey"}

animal_style = [
    {
        "name": "Winnie",
        "color": "darkorange",
        "alias": "Rat V",
        "background": transfection_style["transfected"],
    },
    {
        "name": "Frodo",
        "color": "goldenrod",
        "alias": "Rat F",
        "background": transfection_style["transfected"],
    },
    {
        "name": "Totoro",
        "color": "sandybrown",
        "alias": "Rat T",
        "background": transfection_style["transfected"],
    },
    {
        "name": "Banner",
        "color": "coral",
        "alias": "Rat B",
        "background": transfection_style["transfected"],
    },
    {
        "name": "Odins",
        "color": "peru",
        "alias": "Rat O",
        "background": transfection_style["transfected"],
    },
    # controls
    {
        "name": "Wallie",
        "color": "slategrey",
        "alias": "Rat W",
        "background": transfection_style["control"],
    },
    {
        "name": "Olive",
        "color": "lightsteelblue",
        "alias": "Rat L",
        "background": transfection_style["control"],
    },
    {
        "name": "Yoshi",
        "color": "silver",
        "alias": "Rat Y",
        "background": transfection_style["control"],
    },
    {
        "name": "Bilbo",
        "color": "darkgrey",
        "alias": "Rat I",
        "background": transfection_style["control"],
    },
    {
        "name": "transfected",
        "color": transfection_style["transfected"],
        "alias": "transfected",
        "background": transfection_style["transfected"],
    },
    {
        "name": "control",
        "color": transfection_style["control"],
        "alias": "control",
        "background": transfection_style["control"],
    },
]
animal_style = pd.DataFrame(animal_style)
animal_style.set_index("name", inplace=True)

interval_style = {"test": "#DA70D6", "control": "#7BC8F6"}  # C875C4",


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
        lw=2,
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
            lw=2,
        )
    ax[1].set_title("Optogenetic intervals")
    ax[1].set_yticks(np.arange(i + 1), interval_style.keys())
    return fig
