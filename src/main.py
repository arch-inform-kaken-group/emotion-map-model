# For displaying Japanese characters in plots
import japanize_matplotlib

from dataloader import load_combined_qna_data
from analysis import analyze_and_plot_stacked_emotions, perform_kmeans_clustering_with_collages

if __name__ == "__main__":
    SELECTED_LANGUAGE = 'malaysia'

    # Leave empty to process all pottery
    POTTERY_SELECTION = ['IN0295', 'IN0306', 'MH0037', 'NM0239', 'NZ0001', 'SK0035', 'TK0020', 'UD0028']
    # POTTERY_SELECTION = []

    # Mode: True to INCLUDE items in POTTERY_SELECTION,
    #       False to EXCLUDE items in POTTERY_SELECTION.
    INCLUDE_POTTERY = False
    
    DATASET_ROOT_DIR = r""
    POTTERY_MODELS_DIR = r""

    combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR, POTTERY_MODELS_DIR)

    if not combined_dataframe.empty and POTTERY_SELECTION:
        base_ids = combined_dataframe['pottery_id'].str.split('(', expand=True)[0] # expand to 2 columns, take the first

        initial_count = len(combined_dataframe['pottery_id'].unique())
        if INCLUDE_POTTERY:
            combined_dataframe = combined_dataframe[base_ids.isin(POTTERY_SELECTION)]
            print(f"Included {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique pottery items.")
        else:
            combined_dataframe = combined_dataframe[~base_ids.isin(POTTERY_SELECTION)]
            print(f"Excluded IDs, {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique items remaining.")
        
        if combined_dataframe.empty:
            print("Warning: After filtering, the dataframe is empty. No analysis will be run.")

    if not combined_dataframe.empty:            
        # Stacked Emotion Plots and Timelines
        analyze_and_plot_stacked_emotions(combined_dataframe, language=SELECTED_LANGUAGE, fontsize=8)

        # K-Means Clustering with PCA and Collages
        perform_kmeans_clustering_with_collages(
            combined_dataframe,
            POTTERY_MODELS_DIR,
            language=SELECTED_LANGUAGE
        )
    else:
        print("No data was loaded or data became empty after filtering. No analysis performed.")
