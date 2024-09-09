from nico import Annotations as sann
from nico import Interactions as sint
from nico import Covariations as scov
import nico

#import Annotations as sann
#import Interactions as sint
#import Covariations as scov
import scanpy as sc
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

# Helvetica font publication purpose
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts in PDF files

# Default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana']

import warnings
warnings.filterwarnings("ignore")

#parameters for saving plots
saveas='png'
transparent_mode=False

ref_datapath='./inputRef/'
query_datapath='./inputQuery/'

print("version",nico.__version__)
# answer is version 1.3.0

output_nico_dir='./nico_analysis/'
output_annotation_dir=None
#output_annotation_dir=output_nico_dir+'annotations/'
annotation_save_fname= 'nico_celltype_annotation.h5ad'
inputRadius=0

ref_cluster_tag='cluster' #scRNAseq cell type slot
annotation_slot='nico_ct' #spatial cell type slot

#Module A: Perform cell type annotation of spatial data
anchors_and_neighbors_info=sann.find_anchor_cells_between_ref_and_query(refpath=ref_datapath,
quepath=query_datapath,output_nico_dir=output_nico_dir,
output_annotation_dir=output_annotation_dir)

output_info=sann.nico_based_annotation(anchors_and_neighbors_info,
guiding_spatial_cluster_resolution_tag='leiden0.4',
across_spatial_clusters_dispersion_cutoff=0.15,
ref_cluster_tag=ref_cluster_tag,
resolved_tie_issue_with_weighted_nearest_neighbor='No')


sann.delete_files(output_info)

sann.save_annotations_in_spatial_object(output_info,anndata_object_name=annotation_save_fname)


#Module A: cell type annotation visualization
print('\n\nModule A visualization')
sann.visualize_umap_and_cell_coordinates_with_all_celltypes( #
output_nico_dir=output_nico_dir,
output_annotation_dir=output_annotation_dir,
anndata_object_name=annotation_save_fname,
spatial_cluster_tag=annotation_slot,
spatial_coordinate_tag='spatial',
umap_tag='X_umap',
saveas=saveas,transparent_mode=transparent_mode)


# For visualizing every cell type individually, leave list choose_celltypes empty.
sann.visualize_umap_and_cell_coordinates_with_selected_celltypes(
output_nico_dir=output_nico_dir,
output_annotation_dir=output_annotation_dir,
anndata_object_name=annotation_save_fname,
spatial_cluster_tag=annotation_slot,
spatial_coordinate_tag='spatial',
umap_tag='X_umap',
choose_celltypes=[],
saveas=saveas,transparent_mode=transparent_mode)


#Module B: Infer significant niche cell type interactions
print('\n\nModule B')
do_not_use_following_CT_in_niche=['Basophils','Cycling/GC B cell','pDC']

niche_pred_output=sint.spatial_neighborhood_analysis(
Radius=inputRadius,
output_nico_dir=output_nico_dir,
anndata_object_name=annotation_save_fname,
spatial_cluster_tag=annotation_slot,
removed_CTs_before_finding_CT_CT_interactions=do_not_use_following_CT_in_niche)


celltype_niche_interaction_cutoff=0.1

sint.plot_niche_interactions_with_edge_weight(niche_pred_output,
niche_cutoff=celltype_niche_interaction_cutoff,
saveas=saveas,transparent_mode=transparent_mode)

sint.plot_niche_interactions_without_edge_weight(niche_pred_output,
niche_cutoff=celltype_niche_interaction_cutoff,
saveas=saveas,transparent_mode=transparent_mode)

sint.find_interacting_cell_types(niche_pred_output,
choose_celltypes=[],
celltype_niche_interaction_cutoff=celltype_niche_interaction_cutoff,
coeff_cutoff=30,
saveas=saveas,transparent_mode=transparent_mode,figsize=(4.0,2.0))

#sint.plot_roc_results(niche_pred_output,saveas=saveas,transparent_mode=transparent_mode)
sint.plot_confusion_matrix(niche_pred_output,
saveas=saveas,transparent_mode=transparent_mode)

sint.plot_coefficient_matrix(niche_pred_output,
saveas=saveas,transparent_mode=transparent_mode)
#st.plot_predicted_probabilities(niche_pred_output)

sint.plot_evaluation_scores(niche_pred_output,
saveas=saveas, transparent_mode=transparent_mode, figsize=(4,3))

#Module C: Perform niche cell state covariation analysis using latent factors
print('\n\nModule C')
cov_out=scov.gene_covariation_analysis(iNMFmode=True,
Radius=inputRadius,
no_of_factors=5,
spatial_integration_modality='double',
refpath=ref_datapath,quepath=query_datapath,
output_niche_prediction_dir=output_nico_dir,
ref_cluster_tag=ref_cluster_tag,
LRdbFilename='NiCoLRdb.txt'
)

#Cosine and spearman correlation: visualize the correlation of genes from NMF
scov.plot_cosine_and_spearman_correlation_to_factors(cov_out,
choose_celltypes=[],
NOG_Fa=30,
saveas=saveas,transparent_mode=transparent_mode,
figsize=(15,10))

scov.make_excel_sheet_for_gene_correlation(cov_out)

#Module D: Cell type covariation visualization
print('\n\nModule D')
scov.plot_significant_regression_covariations_as_circleplot(cov_out,
choose_celltypes=[],
pvalue_cutoff=0.05,mention_pvalue=True,
saveas=saveas,transparent_mode=transparent_mode,
figsize=(6,1.25))

#sppath.plot_significant_regression_covariations_as_heatmap(cov_out,
#choose_celltypes=[],
#saveas=saveas,transparent_mode=transparent_mode,figsize=(6,1.25))

#Module E: Analysis of ligand-receptor interactions between covarying niche cell types
print('\n\nModule E')
scov.save_LR_interactions_in_excelsheet_and_regression_summary_in_textfile_for_interacting_cell_types(cov_out,
pvalueCutoff=0.05,correlation_with_spearman=True,
LR_plot_NMF_Fa_thres=0.1,LR_plot_Exp_thres=0.1,number_of_top_genes_to_print=5)


#Perform ligand receptors analysis")
scov.find_LR_interactions_in_interacting_cell_types(cov_out,
choose_interacting_celltype_pair=[],
choose_factors_id=[],
pvalueCutoff=0.05,LR_plot_NMF_Fa_thres=0.2,LR_plot_Exp_thres=0.2,
saveas=saveas,transparent_mode=transparent_mode,
figsize=(12, 10))


#Module G: Visualization of top genes across cell types and factors as dotplot
print('\n\nModule G')
scov.plot_top_genes_for_a_given_celltype_from_all_factors(
cov_out,choose_celltypes=[],
top_NOG=20,saveas=saveas,transparent_mode=transparent_mode)


scov.plot_top_genes_for_pair_of_celltypes_from_two_chosen_factors(cov_out,
choose_interacting_celltype_pair=['Stem/TA','Paneth'],
visualize_factors_id=[1,1],
top_NOG=20,saveas=saveas,transparent_mode=transparent_mode)



#Module F: Perform functional enrichment analysis for genes associated with latent factors
print('\n\nModule F')
scov.pathway_analysis(cov_out,choose_celltypes=[],
    NOG_pathway=50,choose_factors_id=[],savefigure=True,
    positively_correlated=True,saveas='pdf',rps_rpl_mt_genes_included=False)

#Module H: Visualize factor values in the UMAP
print('\n\nModule H')

scov.visualize_factors_in_spatial_umap(cov_out,
visualize_factors_id=[1,1],
choose_interacting_celltype_pair=['Stem/TA','Paneth'],
saveas=saveas,transparent_mode=transparent_mode,figsize=(8,3.5))

scov.visualize_factors_in_scRNAseq_umap(cov_out,
choose_interacting_celltype_pair=['Stem/TA','Paneth'],
visualize_factors_id=[1,1],
saveas=saveas,transparent_mode=transparent_mode,figsize=(8,3.5))
