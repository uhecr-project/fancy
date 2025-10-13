"""Simple plotting utility functions"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fancy import Data
from fancy.plotting import AllSkyMapCartopy as AllSkyMap


def plot_skymap_gb(data: Data, truths: dict, slice_idx: int = 10):
    skymap_gb = AllSkyMap()
    skymap_gb.set_gridlines(ypadding=10, fontsize=12)

    sc = skymap_gb.scatter(
        lons=truths["skycoord_gb_truths"].galactic.l.deg[::slice_idx],
        lats=truths["skycoord_gb_truths"].galactic.b.deg[::slice_idx],
        c=truths["rigidity_truths_samples"][::slice_idx],
        cmap="plasma",
        s=10,
        label="GB samples",
        alpha=0.5,
        vmin=1,
        vmax=25,
    )

    # for kappa egmf, draw a circle around it, color coded with the theta value
    theta_egmf_cmap = mpl.cm.get_cmap("RdPu")
    theta_egmf_norm = mpl.colors.Normalize(
        vmin=0,
        vmax=20,
    )

    sm = mpl.cm.ScalarMappable(
        cmap=theta_egmf_cmap,
        norm=theta_egmf_norm,
    )
    for i, kappa_egmf in enumerate(truths["kappa_egmf_truth_samples"]):
        if i % slice_idx != 0:
            continue
        if kappa_egmf > 0:
            theta_egmf = np.sqrt(7552 / kappa_egmf)  # in degrees
            skymap_gb.tissot(
                truths["skycoord_gb_truths"][i].galactic.l.deg,
                truths["skycoord_gb_truths"][i].galactic.b.deg,
                theta_egmf,
                color=theta_egmf_cmap(theta_egmf_norm(theta_egmf)),
                alpha=0.3,
                zorder=0,
            )

    # add the source direction
    skymap_gb.scatter(
        lons=data.source.coord.galactic.l.deg,
        lats=data.source.coord.galactic.b.deg,
        c="red",
        s=50,
        label="Source",
        marker="*",
        zorder=10,
    )
    # add double colorbars for the energy and kappa_egmf
    cbar = skymap_gb.fig.colorbar(
        sc, ax=skymap_gb.ax, orientation="horizontal", shrink=0.7
    )
    cbar.set_label("log10(Rigidity / EV)")
    cbar2 = skymap_gb.fig.colorbar(
        sm, ax=skymap_gb.ax, orientation="horizontal", pad=0.05, shrink=0.7
    )
    cbar2.set_label("EGMF deflection angle (deg)")

    skymap_gb.ax.legend(loc="upper right")

    skymap_gb.fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}"
    )

    return skymap_gb


def plot_skymap_earth(data: Data, truths: dict, gmf_model: str):
    skymap_earth = AllSkyMap()
    skymap_earth.set_gridlines(ypadding=10, fontsize=12)

    sc = skymap_earth.scatter(
        lons=truths["skycoord_earth_truths"].galactic.l.deg,
        lats=truths["skycoord_earth_truths"].galactic.b.deg,
        c=truths["Etruths_samples"],
        cmap="viridis",
        s=10,
        label="Truths",
        alpha=0.5,
    )

    # add the source direction
    skymap_earth.scatter(
        lons=data.source.coord.galactic.l.deg,
        lats=data.source.coord.galactic.b.deg,
        c="red",
        s=50,
        label="Source",
        marker="*",
    )

    # add colorbar for the energy
    cbar = skymap_earth.fig.colorbar(sc, ax=skymap_earth.ax, orientation="horizontal")
    cbar.set_label("Energy (EeV)")

    skymap_earth.ax.legend(loc="upper right")

    skymap_earth.fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    return skymap_earth


def plot_mean_sigma_lnA(data: Data, truths: dict, config: dict):
    fig, axs = plt.subplots(2, 1)

    dis_labels = ["src", "bg"]
    dis_lss = ["--", ":"]

    for k in range(data.source.N + 1):
        alpha_idx = np.digitize(truths["alphas"][k], config["alpha_grid"]) - 1
        for ims, massid in enumerate(config["mass_ids_grid"]):
            # get the lnA grid
            axs[0].semilogx(
                config["lnA_energy_grid"],
                config["mean_lnA_grid"][:, alpha_idx, ims, k]
                * truths["mass_fracs"][ims, k],
                color=f"C{ims}",
                ls=dis_lss[k],
                label=f"{massid}, {dis_labels[k]}",
            )
            axs[1].semilogx(
                config["lnA_energy_grid"],
                config["var_lnA_grid"][:, alpha_idx, ims, k]
                * truths["mass_fracs"][ims, k],
                color=f"C{ims}",
                ls=dis_lss[k],
            )

        axs[0].semilogx(
            config["lnA_energy_grid"],
            np.sum(
                config["mean_lnA_grid"][:, alpha_idx, :, k]
                * truths["mass_fracs"][None, :, k],
                axis=1,
            ),
            color="k",
            ls=dis_lss[k],
            lw=2,
            label=f"total, {dis_labels[k]}",
        )
        axs[1].semilogx(
            config["lnA_energy_grid"],
            np.sum(
                config["var_lnA_grid"][:, alpha_idx, :, k]
                * truths["mass_fracs"][None, :, k],
                axis=1,
            ),
            color="k",
            ls=dis_lss[k],
            lw=2,
        )
    axs[0].semilogx(
        config["lnA_energy_grid"],
        truths["mean_lnA_truths"],
        color="k",
        ls="-",
        lw=3,
        label="total",
    )
    axs[1].semilogx(
        config["lnA_energy_grid"],
        truths["var_lnA_truths"],
        color="k",
        ls="-",
        lw=3,
    )
    axs[0].set_ylabel("mean lnA")
    axs[1].set_ylabel("var lnA")
    axs[1].set_xlabel(r"$E$ [EeV]")
    fig.legend(loc="upper right", bbox_to_anchor=(1.1, 0.8))

    fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}"
    )

    return fig, axs


def plot_energy(data: Data, truths: dict, config: dict):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    dis_labels = ["src", "bg"]
    dis_lss = ["--", ":"]

    tot_espect = np.zeros_like(config["energy_grid"])
    energy_binedges = np.logspace(
        np.log10(config["energy_grid"].min()),
        np.log10(config["energy_grid"].max()),
        21,
    )

    N_prev_idx = 0
    for k in range(data.source.N + 1):
        alpha_idx = np.digitize(truths["alphas"][k], config["alpha_grid"]) - 1
        for ims, massid in enumerate(config["mass_ids_grid"]):
            # get the energy grid
            axs[0].loglog(
                config["energy_grid"],
                config["spectrum_grid"][:, alpha_idx, ims, k]
                * truths["mass_fracs"][ims, k],
                color=f"C{ims}",
                ls=dis_lss[k],
                label=f"{massid}, {dis_labels[k]}",
            )

        tot_espect_per_d = np.sum(
            config["spectrum_grid"][:, alpha_idx, :, k]
            * truths["mass_fracs"][np.newaxis, :, k],
            axis=-1,
        )

        axs[0].loglog(
            config["energy_grid"],
            tot_espect_per_d,
            color="k",
            ls=dis_lss[k],
            lw=2,
            label=f"total, {dis_labels[k]}",
        )

        tot_espect += truths["Nex_per_src"][k] * tot_espect_per_d / truths["Nex"]

        # also plot histogrammed energy samples per source
        axs[1].hist(
            truths["Etruths_samples"][N_prev_idx : config["Nsamples_per_src"][k]],
            bins=energy_binedges,
            density=False,
            ls=dis_lss[k],
            alpha=0.2,
            label=dis_labels[k],
        )

        N_prev_idx = config["Nsamples_per_src"][k] + N_prev_idx

    # plot total spectrum
    axs[0].loglog(
        config["energy_grid"], tot_espect, color="k", ls="-", lw=3, label="total"
    )

    # plot total histogrammed energy samples
    axs[1].hist(
        truths["Etruths_samples"],
        bins=energy_binedges,
        density=False,
        ls="-",
        lw=3,
        label="total",
        alpha=0.2,
        zorder=0,
    )

    # histogram the samples
    hist_vals, ebinedges = np.histogram(
        truths["Etruths_samples"], bins=energy_binedges, density=False
    )
    yvals = hist_vals / np.sum(hist_vals) / np.diff(ebinedges)
    yerr = np.sqrt(hist_vals) / np.sum(hist_vals) / np.diff(ebinedges)  # Error bars
    axs[0].errorbar(
        np.sqrt(ebinedges[:-1] * ebinedges[1:]),
        yvals,
        yerr=yerr,
        fmt="o",
        label="sampled",
        color="gray",
    )

    axs[0].set_xlabel(r"$E$ [EeV]")
    axs[0].set_ylabel("energy spectrum")
    axs[0].legend()
    axs[0].set_ylim(ymin=1e-7, ymax=1)

    axs[1].set_xlabel(r"$E$ [EeV]")
    axs[1].set_ylabel("counts")
    axs[1].legend()
    axs[1].set_yscale("log")

    fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}"
    )

    return fig, axs


def plot_detected_events(data: Data, truths: dict, gmf_model: str, config: dict):
    skymap_earth = AllSkyMap()
    skymap_earth.set_gridlines(ypadding=10, fontsize=12)

    sc = skymap_earth.scatter(
        lons=truths["skycoord_earth_dets"].galactic.l.deg,
        lats=truths["skycoord_earth_dets"].galactic.b.deg,
        c=truths["Edets"],
        cmap="viridis",
        s=10,
        label="Truths",
        alpha=0.5,
    )

    # add the source direction
    skymap_earth.scatter(
        lons=data.source.coord.galactic.l.deg,
        lats=data.source.coord.galactic.b.deg,
        c="red",
        s=50,
        label="Source",
        marker="*",
    )

    # add colorbar for the energy
    cbar = skymap_earth.fig.colorbar(
        sc, ax=skymap_earth.ax, orientation="horizontal", shrink=0.7
    )
    cbar.set_label("Energy (EeV)")

    skymap_earth.ax.legend(loc="upper right")

    skymap_earth.fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    fig_lnA, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].semilogx(
        config["lnA_energy_grid"],
        truths["mean_lnA_truths"],
        color="k",
        ls="-",
        lw=2,
        label="true",
    )
    axs[1].semilogx(
        config["lnA_energy_grid"],
        truths["var_lnA_truths"],
        color="k",
        ls="-",
        lw=2,
        label="true",
    )

    axs[0].semilogx(
        config["lnA_energy_grid"],
        truths["mean_lnA_dets"],
        color="r",
        ls="--",
        lw=2,
        label="det",
    )
    axs[1].semilogx(
        config["lnA_energy_grid"],
        truths["var_lnA_dets"],
        color="r",
        ls="--",
        lw=2,
        label="det",
    )

    axs[0].set_ylabel("mean lnA")
    axs[1].set_ylabel("var lnA")
    axs[1].set_xlabel(r"$E$ [EeV]")
    axs[0].legend(loc="upper right")

    fig_en, ax = plt.subplots(figsize=(8, 6))
    energy_binedges = np.logspace(
        np.log10(config["energy_grid"].min()),
        np.log10(config["energy_grid"].max()),
        21,
    )
    ax.hist(
        truths["Edets"],
        bins=energy_binedges,
        density=False,
        histtype="step",
        label="Detected energies",
    )
    ax.hist(
        truths["Etruths"],
        bins=energy_binedges,
        density=False,
        histtype="step",
        label="True energies",
    )
    ax.set_xlabel("Energy (EeV)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    return skymap_earth, fig_en, fig_lnA


def plot_backprop_skymap(data: Data, truths: dict, gmf_model: str, gb : bool=True):
    skymap_defl = AllSkyMap()
    skymap_defl.set_gridlines(ypadding=10, fontsize=12)

    if gb:
        skymap_defl.scatter(
            lons=truths["skycoord_gb_truths"].galactic.l.deg,
            lats=truths["skycoord_gb_truths"].galactic.b.deg,
            c="gray",
            marker="o",
            s=10,
            label="Truths - GB",
            alpha=0.3,
            zorder=0,
        )
    sc = skymap_defl.scatter(
        lons=truths["skycoord_earth_dets"].galactic.l.deg,
        lats=truths["skycoord_earth_dets"].galactic.b.deg,
        c="black",
        marker="+",
        s=40,
        label="Truths - Earth",
        alpha=0.8,
        zorder=1,
    )

    skymap_defl.scatter(
        lons=truths["skycoord_gb_truths_bp"].galactic.l.deg,
        lats=truths["skycoord_gb_truths_bp"].galactic.b.deg,
        c="blue",
        marker="s",
        s=10,
        label="Backpropagated - GB",
        alpha=0.8,
        zorder=5,
    )

    # for kappa gmf, draw a circle around it, color coded with the theta value
    theta_gmf_cmap = mpl.cm.get_cmap("RdPu")
    theta_gmf_norm = mpl.colors.Normalize(
        vmin=0,
        vmax=50,
    )

    sm = mpl.cm.ScalarMappable(
        cmap=theta_gmf_cmap,
        norm=theta_gmf_norm,
    )
    for i, theta_gmf in enumerate(truths["theta_gmfs"]):
        skymap_defl.tissot(
            truths["skycoord_gb_truths_bp"][i].galactic.l.deg,
            truths["skycoord_gb_truths_bp"][i].galactic.b.deg,
            theta_gmf,
            color=theta_gmf_cmap(theta_gmf_norm(theta_gmf)),
            alpha=0.3,
            zorder=2,
        )

    # add the source direction
    skymap_defl.scatter(
        lons=data.source.coord.galactic.l.deg,
        lats=data.source.coord.galactic.b.deg,
        c="red",
        s=50,
        label="Source",
        marker="*",
        zorder=10,
    )
    # add double colorbars for the energy and kappa_egmf
    cbar = skymap_defl.fig.colorbar(
        sm, ax=skymap_defl.ax, orientation="horizontal", pad=0.05
    )
    cbar.set_label("GMF deflection angle (deg)")

    skymap_defl.ax.legend(loc="upper right")

    skymap_defl.fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    return skymap_defl


def plot_kappas(data: Data, truths: dict, gmf_model: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        truths["kappa_egmf_truths"],
        bins=20,
        density=True,
        histtype="step",
        label="Kappa EGMF at source",
        color="black",
    )
    ax.hist(
        truths["kappa_gmfs"],
        bins=20,
        density=True,
        histtype="step",
        label="Kappa GMF from backpropagation",
        color="blue",
    )
    ax.set_xlabel("Kappa")
    ax.set_ylabel("Density")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    return fig


def plot_thetas(data: Data, truths: dict, gmf_model: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    theta_binedges = np.linspace(0, 80, 21)
    ax.hist(
        truths["theta_gmfs"],
        bins=theta_binedges,
        density=True,
        histtype="step",
        label="Theta GMF from backpropagation",
        color="blue",
    )

    # plot only those with kappa > 0
    kappa_egmf_truths = truths["kappa_egmf_truths"][truths["kappa_egmf_truths"] > 0]
    ax.hist(
        np.sqrt(7552.0 / kappa_egmf_truths),
        bins=theta_binedges,
        density=True,
        histtype="step",
        label="Theta EGMF at source",
        color="black",
    )
    ax.set_xlabel("Theta / deg")
    ax.set_ylabel("Density")
    ax.legend()

    fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    return fig


def plot_backprop_rigidities(data: Data, truths: dict, gmf_model: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    rigidity_binedges = np.logspace(
        np.log10(1), np.log10(25), 21
    )
    ax.hist(
        truths["rigidity_truths"],
        bins=rigidity_binedges,
        density=True,
        histtype="step",
        label="Rigidity at source",
        color="black",
    )
    ax.hist(
        truths["rigidity_bp"].flatten(),
        bins=rigidity_binedges,
        density=True,
        histtype="step",
        label="Rigidity from backpropagation",
        color="blue",
    )
    ax.set_xlabel("Rigidity (EV)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.suptitle(
        f"{data.detector.label}, {data.source.label}, {data.detector.mass_model}, {gmf_model}"
    )

    return fig
