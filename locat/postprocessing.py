import pandas as pd


def build_locat_df(results, alpha=0.05):
    df = pd.DataFrame({gene: res._asdict() for gene, res in results.items()}).T
    df.index.name = "gene"
    df = df.dropna(axis="columns", how="all")
    df["is_conc_sig"] = df["concentration_pval"] < alpha
    df["is_depl_sig"] = df["depletion_pval"] < alpha
    df["is_joint_sig"] = df["pval"] < alpha
    return df.sort_values(["pval", "concentration_pval", "depletion_pval"])
