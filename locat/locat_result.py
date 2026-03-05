from typing import NamedTuple

class LocatResult(NamedTuple):
    """
    This class stores the location results for a gene
    """
    #: The name of the analysed gene
    gene_name: str
    #: The BIC score of the analysed gene computed by the GMM
    bic: float
    #: The Z-score of the analysed gene
    zscore: float
    #: The sensitivity score of the analysed gene
    sens_score: float
    #: The sensitivity score of the analysed gene
    depletion_pval: float
    #: The depletion p-value of the analysed gene
    concentration_pval: float
    #: The h_size
    h_size: float
    #: The h_sens
    h_sens: float
    #: The combined p-value
    pval: float
    #: The number of components used in the GMM
    K_components: float
    #: The sample size, i.e. the number of cells expressing the gene
    sample_size: float
    #: A large amount of debug features
    depletion_scan: dict| None = None
