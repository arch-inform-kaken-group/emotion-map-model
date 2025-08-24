import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import EMOTION_COLOR_MAP_EN, EMOTION_COLOR_MAP_JP
from rendering import create_cluster_collage


# Takes a combined DataFrame and generates stacked bar graphs for all pottery.
def analyze_and_plot_stacked_emotions(combined_df: pd.DataFrame, language='malaysia', fontsize=8):
    if combined_df.empty:
        print("Combined DataFrame is empty. No analysis performed.", file=sys.stderr)
        return

    if language == 'japan':
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_JP
        EMOTION_STACK_ORDER = ["何も感じない", "不気味・不安・怖い", "不思議・意味不明", "美しい・芸術的だ", "面白い・気になる形だ", "NO RESPONSE"]
    else:
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_EN
        EMOTION_STACK_ORDER = ["Feel nothing", "Creepy / unsettling / scary", "Strange and incomprehensible", "Beautiful and artistic", "Interesting and attentional shape", "NO RESPONSE"]

    df = combined_df.copy()
    df['answer'] = df['answer'].str.strip()
    df = df.sort_values(by=['pottery_id', 'timestamp']).reset_index(drop=True)

    ######################################################################################
    # Plot 1: Percentage by Event Count (Session-Normalized)
    # Finds the percentage of each invididual pottery record then averages the percentages
    print("Generating plot for percentage breakdown of emotions (by event count)")

    session_counts_df = pd.crosstab([df['pottery_id'], df['session_id']], df['answer'])
    print(session_counts_df.head())

    session_percentage_df = session_counts_df.div(session_counts_df.sum(axis=1), axis=0) * 100
    print(session_percentage_df.head())

    percentage_df = session_percentage_df.groupby('pottery_id').mean()
    print(percentage_df.head())

    # Enforce consistent column order for stacking
    for emotion in EMOTION_COLOR_MAP.keys():
        if emotion not in percentage_df.columns:
            percentage_df[emotion] = 0
    plot1_order = [
        e for e in EMOTION_STACK_ORDER
        if e in percentage_df.columns and e != "NO RESPONSE"
    ]
    percentage_df = percentage_df[plot1_order]

    # Plot a stacked bar
    ax1 = percentage_df.plot(kind='bar',
                             stacked=True,
                             figsize=(20, 8),
                             color=[
                                 EMOTION_COLOR_MAP.get(e, '#CCCCCC')
                                 for e in percentage_df.columns
                             ],
                             width=0.7,
                             fontsize=fontsize)

    # Apply percentage text labels to each stacked bar
    for container in ax1.containers:
        labels = [f'{v:.1f}' if v > 0.1 else '' for v in container.datavalues]
        ax1.bar_label(container,
                      labels=labels,
                      label_type='center',
                      fontsize=fontsize - 3,
                      color='black',
                      weight='bold')

    plt.title('Average Percentage of Emotions per Pottery (by Event Count)', fontsize=16)
    plt.ylabel('Average Percentage (%)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('emotion_stacked_percentage_plot_by_event_count.png')
    plt.show()

    #########################################################################################
    # Plot 2: Average duration
    print("\nCalculating emotion durations")

    # First groups the data by pottery_id | session_id and calculate the difference between consecutive timestamps
    df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()

    # Then, find the indexes where emotion response change with .shift()
    emotion_changed = df['answer'] != df.groupby(['pottery_id', 'session_id'])['answer'].shift()

    # Also find discontinuous timestamps, >50ms between records
    time_gap_exceeded = df['time_diff'] > 0.05

    # If either emotion change or discontinuous timestamp then start a new block
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    # Group by pottery_id | ssession_id | block_id
    block_durations = df.groupby(['pottery_id', 'session_id', 'block_id']).agg(
                                            tart_time=('timestamp', 'min'),
                                            end_time=('timestamp', 'max'),
                                            answer=('answer', 'first')
                                        ).reset_index()
    
    # Calculate each blocks duration
    block_durations['duration'] = block_durations['end_time'] - block_durations['start_time']

    duration_df = block_durations.pivot_table(index='pottery_id',
                                              columns='answer',
                                              values='duration',
                                              aggfunc='sum',
                                              fill_value=0)
    pottery_session_counts = df.groupby('pottery_id')['session_id'].nunique()
    average_duration_df = duration_df.div(pottery_session_counts, axis=0)

    # Enforce consistent column order for stacking
    for emotion in EMOTION_COLOR_MAP.keys():
        if emotion not in average_duration_df.columns:
            average_duration_df[emotion] = 0
    plot2_order = [
        e for e in EMOTION_STACK_ORDER
        if e in average_duration_df.columns and e != "NO RESPONSE"
    ]
    average_duration_df = average_duration_df[plot2_order]

    ax2 = average_duration_df.plot(kind='bar',
                                   stacked=True,
                                   figsize=(20, 8),
                                   color=[
                                       EMOTION_COLOR_MAP.get(e, '#CCCCCC')
                                       for e in average_duration_df.columns
                                   ],
                                   width=0.7,
                                   fontsize=fontsize)
    
    for container in ax2.containers:
        labels = [f'{v:.1f}' if v > 0.1 else '' for v in container.datavalues]
        ax2.bar_label(container,
                      labels=labels,
                      label_type='center',
                      fontsize=fontsize - 3,
                      color='black',
                      weight='bold')
        
    plt.title('Average Duration of Emotions per Pottery (50ms gap limit)', fontsize=16)
    plt.ylabel('Average Duration (seconds)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('emotion_stacked_duration_plot.png')
    plt.show()

    #########################################################################################
    # Plot 3: Percentage Viewing Time (including the no response label)
    print("\nGenerating plot for percentage of viewing time (including no response)")

    session_durations = df.groupby(['pottery_id', 'session_id'])['timestamp'].agg(['min', 'max'])
    session_durations['total_duration'] = (session_durations['max'] - session_durations['min']).clip(upper=60)
    emotion_duration_per_session = block_durations.groupby(['pottery_id', 'session_id'])['duration'].sum()
    session_summary = pd.merge(
        session_durations,
        emotion_duration_per_session.rename('emotion_duration'),
        on=['pottery_id', 'session_id'])
    session_summary['NO RESPONSE'] = session_summary['total_duration'] - session_summary['emotion_duration']
    total_emotion_durations = block_durations.groupby(['pottery_id', 'answer'])['duration'].sum().unstack(fill_value=0)
    total_not_viewing = session_summary.groupby('pottery_id')['NO RESPONSE'].sum()
    final_durations = pd.concat([total_emotion_durations, total_not_viewing], axis=1)
    percentage_viewing_time_df = final_durations.div(final_durations.sum(axis=1), axis=0) * 100

    # Enforce consistent column order for stacking
    all_possible_cols = list(EMOTION_COLOR_MAP.keys()) + ["NO RESPONSE"]
    for col in all_possible_cols:
        if col not in percentage_viewing_time_df.columns:
            percentage_viewing_time_df[col] = 0
    plot3_order = [
        e for e in EMOTION_STACK_ORDER
        if e in percentage_viewing_time_df.columns
    ]
    percentage_viewing_time_df = percentage_viewing_time_df[plot3_order]

    ax3 = percentage_viewing_time_df.plot(
        kind='bar',
        stacked=True,
        figsize=(20, 8),
        color=[
            EMOTION_COLOR_MAP.get(e, '#CCCCCC')
            for e in percentage_viewing_time_df.columns
        ],
        width=0.7,
        fontsize=fontsize)

    for container in ax3.containers:
        # Format as integer percentage, only show if value is > 0.1%
        labels = [f'{v:.1f}' if v > 0.1 else '' for v in container.datavalues]
        ax3.bar_label(container,
                      labels=labels,
                      label_type='center',
                      fontsize=fontsize - 3,
                      color='black',
                      weight='bold')

    plt.title('Percentage of Viewing Time per Pottery (60s per session)', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('viewing_time_percentage_plot_including_no_response.png')
    plt.show()

    #########################################################################################
    # Plot 4: Pottery timeline of all participants
    output_dir = "timelines"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating stacked emotion timelines. Plots will be saved in '{output_dir}'")

    # Group data by each pottery ID to create a separate plot for each
    pottery_groups = combined_df.groupby('pottery_id')

    for pottery_id, pottery_df in tqdm(pottery_groups, desc="Creating Stacked Timelines"):
        sessions = sorted(pottery_df['session_id'].unique())
        num_sessions = len(sessions)

        # Create a figure with height proportional to the number of sessions
        fig, ax = plt.subplots(figsize=(20, num_sessions * 0.7))

        # Process and plot each session as a separate row in the figure
        for i, session_id in enumerate(sessions):
            session_df = pottery_df[pottery_df['session_id'] == session_id].copy().sort_values('timestamp')

            if session_df.empty:
                continue

            # Normalize timestamps to start from 0 for this session
            session_start_time = session_df['timestamp'].min()
            session_df['timestamp'] = session_df['timestamp'] - session_start_time

            # Identify continuous blocks of the same emotion
            # same as plot 2 calculations
            session_df['time_diff'] = session_df['timestamp'].diff()
            emotion_changed = session_df['answer'] != session_df['answer'].shift()
            time_gap_exceeded = session_df['time_diff'] > 0.05
            session_df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

            block_df = session_df.groupby('block_id').agg(
                start_time=('timestamp', 'min'),
                end_time=('timestamp', 'max'),
                answer=('answer', 'first')).reset_index()
            block_df['duration'] = block_df['end_time'] - block_df['start_time']

            colors = [
                EMOTION_COLOR_MAP.get(ans, "#808080")
                for ans in block_df["answer"]
            ]

            ax.barh(
                y=[i] * len(block_df),
                width=block_df['duration'],
                left=block_df['start_time'],
                color=colors,
                height=0.8,
            )

        ax.set_yticks(range(num_sessions))
        ax.set_yticklabels(sessions, fontsize=8)
        ax.invert_yaxis()  # Puts the first session at the top
        ax.set_xlabel("Time Since Session Start (seconds)")
        ax.set_xlim(left=0)
        ax.set_title(f"Emotion Timelines for {pottery_id}")

        fig.tight_layout(rect=[0, 0, 0.97, 1], pad=2)

        filename = f"timeline_stacked_{pottery_id}.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)


# Utility function for visualization aid
# Expands the ellipse by a factor to ensure all points are inside
# Clusters from K-means cclustering should be taken as the accurate measure
def draw_ellipse(points, ax=None, **kwargs):
    ax = ax or plt.gca()
    if len(points) < 2: return
    
    cov = np.cov(points, rowvar=False)
    if np.isclose(np.linalg.det(cov), 0):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        width = (x_max - x_min) * 1.05
        height = (y_max - y_min) * 1.05
        ellipse = mpatches.Ellipse(xy=center, width=width or 0.5, height=height or 0.5, angle=0, **kwargs)
        ax.add_patch(ellipse)
        return
        
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    transformed_points = points @ eigvecs
    x_min, y_min = np.min(transformed_points, axis=0)
    x_max, y_max = np.max(transformed_points, axis=0)
    center_transformed = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    distances_from_center = transformed_points - center_transformed
    a, b = np.max(np.abs(distances_from_center), axis=0)
    width = 2 * a * 1.45
    height = 2 * b * 1.45
    final_center = center_transformed @ eigvecs.T
    ellipse = mpatches.Ellipse(xy=final_center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


# Performs K-Means clustering and generates plots and 3D model collages.
def perform_kmeans_clustering_with_collages(combined_df: pd.DataFrame, pottery_models_dir, language='malaysia'):
    if combined_df.empty:
        print("DataFrame is empty. No clustering analysis performed.", file=sys.stderr)
        return

    EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_JP if language == 'japan' else EMOTION_COLOR_MAP_EN

    # Preparing data for clustering    
    session_counts_df = pd.crosstab([combined_df['pottery_id'], combined_df['session_id']], combined_df['answer'])
    session_percentage_df = session_counts_df.div(session_counts_df.sum(axis=1), axis=0) * 100
    percentage_df = session_percentage_df.groupby('pottery_id').mean()
    all_emotions = list(EMOTION_COLOR_MAP.keys())
    for emotion in all_emotions:
        if emotion not in percentage_df.columns:
            percentage_df[emotion] = 0
    percentage_df = percentage_df.fillna(0)[all_emotions]

    # Finding optimal number of clusters using the Elbow Method
    inertia = []
    k_range = range(1, min(21, len(percentage_df)))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
        kmeans.fit(percentage_df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig('kmeans_elbow_plot.png')
    plt.close()

    # Performing K-Means clustering from k=2 to k=20, for quick visualization
    for j in range(2, 21):
        optimal_k = j
        data_to_cluster = percentage_df.copy()

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=25)
        cluster_labels = kmeans.fit_predict(data_to_cluster)

        output_dir = f'k_{j}'
        os.makedirs(output_dir, exist_ok=True)

        # Visualizing clusters using PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(data_to_cluster)
        fig, ax = plt.subplots(figsize=(14, 10))
        cmap_obj = plt.get_cmap('viridis', optimal_k)
        scatter = plt.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            c=cluster_labels, 
            cmap=cmap_obj, 
            s=100, 
            alpha=0.9, 
            edgecolors='k')

        for i in range(optimal_k):
            points = reduced_features[cluster_labels == i]
            draw_ellipse(points, ax=ax, edgecolor=cmap_obj(i / (optimal_k - 1) if optimal_k > 1 else 0), facecolor='none', lw=2, linestyle='--')

        for i, txt in enumerate(data_to_cluster.index):
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=9)

        plt.title(f'K-Means Clustering of Pottery (K={optimal_k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(optimal_k)], title="Clusters")
        
        # Zoom in on the points
        x_min, x_max = reduced_features[:, 0].min(), reduced_features[:, 0].max()
        y_min, y_max = reduced_features[:, 1].min(), reduced_features[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - (x_padding or 1), x_max + (x_padding or 1))
        ax.set_ylim(y_min - (y_padding or 1), y_max + (y_padding or 1))
        plt.grid(True)

        cluster_plot_path = os.path.join(output_dir, 'pottery_kmeans_cluster_plot.png')
        plt.savefig(cluster_plot_path)
        plt.close(fig)

        # Generating cluster assignments and 3D model collages, can help identify which potteries are in a cluster
        # Also saves a .txt file with the clusters
        with open(os.path.join(output_dir, 'cluster_assignments.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Cluster Assignments for K={optimal_k}\n")
            for i in range(optimal_k):
                members = data_to_cluster.index[cluster_labels == i].tolist()
                f.write(f"\nCluster {i}:\n" + ", ".join(members) + "\n")
                if members:
                    create_cluster_collage(pottery_ids=members, pottery_dir=pottery_models_dir, cluster_id=i, output_dir=output_dir)