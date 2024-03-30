import pandas as pd
import os

dirs = [
    'Traces_Base/',
    'Traces_Base_Shield/',
    'Traces_Base_RSS_Shield/',
    'Traces_Higher_Penalty_Collision/',
    'Traces_Higher_Penalty_Collision_Shield/',
    'Traces_Higher_Penalty_Collision_RSS_Shield/',
]


for d in dirs:
    csv_dist = os.path.join(d, 'SimulationStatistics_Distance.csv')
    csv_rwrd = os.path.join(d, 'SimulationStatistics_Reward.csv')
    csv_velo = os.path.join(d, 'SimulationStatistics_Velocity.csv')
    csv_right = os.path.join(d, 'SimulationStatistics_Right_Lane.csv')

    df_dist = pd.read_csv(csv_dist)
    df_rwrd = pd.read_csv(csv_rwrd)
    df_velo = pd.read_csv(csv_velo)
    df_right_lane = pd.read_csv(csv_right)

    trace_lengths = df_dist['Trace Length'] - 2
    dists = df_dist['Estimated Value']
    rewards = df_rwrd['Estimated Value']
    velocities = df_velo['Estimated Value']
    right_lane = df_right_lane['Estimated Value']

    print('#', d)
    print(f"Episode Length: {trace_lengths.mean():.2f} +- {trace_lengths.std():.2f}")
    print(f"Velocity: {velocities.mean():.2f} +- {velocities.std():.2f}")
    print(f"Distance: {dists.mean():.2f} +- {dists.std():.2f}")
    print(f"Reward: {rewards.mean():.2f} +- {rewards.std():.2f}")
    print(f"Right Lane Time: {right_lane.mean():.2f} +- {right_lane.std():.2f}")
    print()
