from matplotlib.colors import LogNorm
from sklearn import mixture
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
def train_clustering(adata, genesuse, pthresh = 0.5): #doesnt need embedding coords, just uses neighbors attr.
    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning)
    adata = adata[:,genesuse]
    data = adata.X#.to_df() # counts

    sc.tl.leiden(adata, resolution=0.25)
    #check if log1p exists, if so set = to none to avoid errors
    if "log1p" in adata.uns.keys():
        adata.uns['log1p']["base"] = None
    else:
        print("No log1p uns is found")
    from collections import Counter
    print(Counter(adata.obs["leiden"]))
    un = list(np.unique(adata.obs["leiden"]))
    #print("newleiden",newleiden[:10])
    exclude = []
    for i in un:
        if ((adata[adata.obs["leiden"]==i,:].copy().shape[0]) < 2):
            exclude.append((i))
    include = [i for i in un if i not in exclude]
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', groups = include)
    names = np.array(adata.uns['rank_genes_groups']['names'].tolist()).flatten()
    pvals = np.array(adata.uns['rank_genes_groups']['pvals'].tolist()).flatten()
    logfcs = np.array(adata.uns['rank_genes_groups']['logfoldchanges'].tolist()).flatten()
    #print(adata.uns['rank_genes_groups']['names'])

    maxnames = []
    minnames = []
    maxpvals = []
    maxlogfcs = []
    notsignames = []
    minpvals=[]
    #for each unique value
    for i in np.unique(names):
        #get indexes of this value
        inds = list(np.where(names == i))
        #look up which have pvals <.05
        #update
        if (pthresh!=0.5):
            
            i_names = list(itertools.compress(names[inds].tolist()[0],(pvals[inds]<100).tolist()[0]))
            i_pvals = list(itertools.compress(pvals[inds].tolist()[0],(pvals[inds]<100).tolist()[0]))
            lowest = sorted(i_pvals, reverse=False)[0]
            minind = np.where(np.array(i_pvals)==lowest)[0][0]
            minnames.append(i_names[minind])
            minpvals.append(i_pvals[minind])
            continue
        if (np.sum(pvals[inds]<0.05)>0):
            #print(np.array(names)[inds])
            #print(pvals[inds]<0.05)
            i_names = list(itertools.compress(names[inds].tolist()[0],(pvals[inds]<0.05).tolist()[0]))
            i_pvals = list(itertools.compress(pvals[inds].tolist()[0],(pvals[inds]<0.05).tolist()[0]))
            i_logfcs = list(itertools.compress(logfcs[inds].tolist()[0],(pvals[inds]<0.05).tolist()[0]))
            #get new list of indices

            inds = list(range(len(i_names)))

            #lookup which has highest logfc

            highest = sorted(i_logfcs, reverse=True)[0]
            lowest = sorted(i_pvals, reverse=False)[0]
            #get maximum index
            maxind = np.where(np.array(i_logfcs)==highest)[0][0]
            minind = np.where(np.array(i_pvals)==lowest)[0][0]
            #print("maxind",maxind)
            #update arrays
            maxnames.append(i_names[maxind])
            minnames.append(i_names[minind])
            maxpvals.append(i_pvals[maxind])
            minpvals.append(i_pvals[minind])
            maxlogfcs.append(i_logfcs[maxind])
        else:
            notsignames.append(i)
    debugging = [maxlogfcs,maxnames]
    sortinds = [np.argsort(maxlogfcs)[::-1]]
    maxlogfcs = np.array(maxlogfcs)[sortinds]
    #print(maxlogfcs[:10])
    maxpvals = np.array(maxpvals)[sortinds]
    maxnames = np.array(maxnames)[sortinds]
    sortinds = [np.argsort(minpvals)]#[::-1]]
    minnames = np.array(minnames)[sortinds]
    print("maxnames_len",len(maxnames))
    return (maxlogfcs,maxpvals,maxnames,minnames, notsignames, sortinds)

def train_clustering_logpadj(adata, genesuse, resolution=0.25, method="wilcoxon"):
    ad = adata[:, genesuse].copy()
    if "connectivities" not in ad.obsp:
        raise ValueError("Neighbors missing: run sc.pp.neighbors first (use_rep='coords' etc).")

    if np.min(ad.X) >= 0:
        sc.pp.log1p(ad)

    sc.tl.leiden(ad, resolution=resolution)
    if "log1p" in ad.uns:
        ad.uns["log1p"]["base"] = None

    sizes = ad.obs["leiden"].value_counts()
    groups = sizes.index[sizes >= 2].tolist()
    if not groups:
        raise ValueError("All Leiden clusters have <2 cells.")

    sc.tl.rank_genes_groups(ad, "leiden", groups=groups, method=method)

    rg = ad.uns["rank_genes_groups"]
    names = pd.DataFrame(rg["names"]).loc[:, groups].to_numpy().T
    padj  = pd.DataFrame(rg["pvals_adj"]).loc[:, groups].to_numpy().T
    if "logfoldchanges" in rg:
        lfc = pd.DataFrame(rg["logfoldchanges"]).loc[:, groups].to_numpy().T
    else:
        lfc = np.full(padj.shape, np.nan, dtype=float)

    n_groups, n_top = names.shape
    df = pd.DataFrame({
        "gene": names.ravel(),
        "group": np.repeat(np.array(groups, dtype=object), n_top),
        "pvals_adj": padj.ravel(),
        "logfoldchange": lfc.ravel(),
    }).dropna(subset=["gene", "pvals_adj"]).sort_values(["gene", "pvals_adj"], kind="mergesort")

    best = df.drop_duplicates("gene", keep="first").set_index("gene")
    best["log10_padj"] = -np.log10(np.maximum(best["pvals_adj"].to_numpy(), 1e-300))
    return best.sort_values("pvals_adj")

def parse_output(adata,res,localres):
    #get vector of wstats for every gene
    #also, get expression volume contained in each gene
    keys = list(res.keys())
    varnames = list(adata.var_names)
    wstats = [j["local_zscore"] for i,j in localres.items()]
    
    enrvols = []
    enrspecs = []
    
    for wst_i, c in zip(wstats,range(len(keys))):
        enr_ind = np.array(wst_i) > 0
        #specificity: proportion of enriched cells that are expressing the gene
        enrspecs.append( np.sum(np.logical_and(adata.X[:,c]>0,enr_ind))/np.sum(enr_ind)  )
        #sensitivity: proportion of total expression volume accounted for by enriched cells
        enrvols.append( np.sum(adata.X[:,c] * (enr_ind/1))/np.sum(adata.X[:,c])  )

    return(np.array(wstats).T, enrvols, enrspecs)

def plot_gene_localization_summary(
    genes,
    locat_df,
    adata,
    suptitle="Gene Localization Summary",
    embedding_key="X_umap",
    embedding_dims=2
):
    """
    Plots expression, GMM fit, and localized masks for each gene.
    """
    umap = adata.obsm[embedding_key][:, :embedding_dims]
    n_genes = len(genes)

    # One row per gene, 3 plots per row
    fig, axes = plt.subplots(
        nrows=n_genes,
        ncols=3,
        figsize=(15, 5 * n_genes),
        squeeze=False
    )

    for i, gene in enumerate(genes):
        # Extract data from locat_df
        rec = locat_df.loc[gene]
        gmm = rec["gmm"]
        gene_prior = rec["gene_prior"]
        m_cuts = rec["m_cuts"]
        components_to_use = rec["components_to_use"]

        # Expression vector
        expr = gene_prior
        is_expressing = expr > 0

        # GMM density
        density = gmm.pdf(umap)

        # Mahalanobis distances
        m0 = gmm.mahalanobis_dist(umap)

        # Compute is_localized mask
        is_localized = ~np.all(m0 > m_cuts[None, :], axis=1)

        # Localized vs unlocalized masks
        loc_mask = is_expressing & is_localized
        unloc_mask = is_expressing & (~is_localized)

        ### 1️⃣ Expression plot (ordered by expression)
        sort_idx = np.argsort(expr)
        ax = axes[i, 0]
        sc = ax.scatter(
            umap[sort_idx, 0],
            umap[sort_idx, 1],
            c=expr[sort_idx],
            cmap="viridis",
            s=60,
            edgecolor="none"
        )
        ax.set_title(f"{gene}\nExpression", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Expression")

        ### 2️⃣ GMM density plot (ordered by density)
        sort_idx_dens = np.argsort(density)
        ax = axes[i, 1]
        sc = ax.scatter(
            umap[sort_idx_dens, 0],
            umap[sort_idx_dens, 1],
            c=density[sort_idx_dens],
            cmap="magma",
            s=20,
            edgecolor="none"
        )
        ax.set_title(f"{gene}\nGMM Density", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("Density")

        ### 3️⃣ Localized vs unlocalized plot (order: background, unloc, loc)
        ax = axes[i, 2]
        # Background
        ax.scatter(
            umap[:, 0],
            umap[:, 1],
            c="lightgrey",
            s=10,
            alpha=0.3,
            edgecolor="none"
        )
        # Unlocalized
        ax.scatter(
            umap[unloc_mask, 0],
            umap[unloc_mask, 1],
            c="blue",
            s=25,
            edgecolor="none",
            label="Unlocalized"
        )
        # Localized
        ax.scatter(
            umap[loc_mask, 0],
            umap[loc_mask, 1],
            c="red",
            s=35,
            edgecolor="black",
            linewidth=0.3,
            label="Localized"
        )
        ax.set_title(
            (
                f"{gene}\nLocalized: {loc_mask.sum()} | "
                f"Unloc: {unloc_mask.sum()} | "
                f"ExpUnloc: {rec.get('expected_unlocalized', np.nan):.1f}\n"
                f"LocPval: {rec['localization_pval']:.2e}"
            ),
            fontsize=11
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(suptitle, fontsize=18, y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    plt.show()


def plotgenes(
    adata, 
    d0, 
    topgenes, 
    suptitle="TITLEARG", 
    size=10, 
    emb="X_umap", 
    genes_per_row=5, 
    text_size=12,
    geneinf = False
):


    # Convert gene list
    genes = np.array(topgenes)
    n_genes = len(genes)

    # UMAP coordinates
    umap = adata.obsm[emb]

    # Grid size
    ncols = genes_per_row
    nrows = int(np.ceil(n_genes / ncols))

    # Create figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 4, nrows * 4),
        squeeze=False
    )

    # Flatten axes
    axes_flat = axes.flatten()

    # Loop through genes
    for i, (gene, ax) in enumerate(zip(genes, axes_flat)):
        expr = adata[:, gene].X

        # Convert sparse matrix if needed
        if not isinstance(expr, np.ndarray):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()

        # Order cells so high-expression on top
        order = np.argsort(expr)
        x = umap[order, 0]
        y = umap[order, 1]
        c = expr[order]

        sc = ax.scatter(
            x, y,
            c=c,
            cmap="viridis",
            s=size,
            edgecolor="none"
        )

        # Multiline title
        if geneinf:
            ax.set_title(
                f"{gene}\n"
                f"pval: {d0.loc[gene]['pval']:.2e}\n"
                f"conc_pval: {d0.loc[gene]['concentration_pval']:.2e}\n"
                f"loca_pval: {d0.loc[gene]['localization_pval']:.2e}",
                fontsize=text_size,
                pad=10
            )
        else:
            ax.set_title(
                f"{gene}\n"
                f"pval: {d0.loc[gene]['pval']:.2e}\n",
                fontsize=text_size,
                pad=2
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # Colorbar for each subplot
        #cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        #cbar.set_label("Expression", fontsize=text_size-2)

    # Remove unused axes
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Add suptitle
    fig.suptitle(
        suptitle,
        fontsize=text_size + 6,
        y=1.02
    )

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    plt.show()

