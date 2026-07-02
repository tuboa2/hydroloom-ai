import os
import sys
import time
import polars as pl
from sklearn.cluster import KMeans
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def print_slow(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def generate_random_profile(mins, maxs):
    return [
        np.random.uniform(mins[f][0], maxs[f][0])
        for f in [
            "log_per_capita_usage",
            "dry_day_spike_factor",
            "efficiency_penalty_ratio",
            "landscape_demand_index",
        ]
    ]

def get_cluster_labels(kmeans):
    centers = kmeans.cluster_centers_
    labels = {}
    usages = centers[:, 0]
    landscape = centers[:, 3]
    
    sorted_usage = np.argsort(usages)
    labels[sorted_usage[0]] = "Conservationists (Low Volume)"
    labels[sorted_usage[-1]] = "Heavy Users (High Volume)"
    
    remaining = [i for i in range(4) if i not in labels]
    if landscape[remaining[0]] > landscape[remaining[1]]:
        labels[remaining[0]] = "Outdoor/Landscape Heavy"
        labels[remaining[1]] = "Standard Average Consumers"
    else:
        labels[remaining[1]] = "Outdoor/Landscape Heavy"
        labels[remaining[0]] = "Standard Average Consumers"
        
    return labels


def main():
    print_slow(
        f"{CYAN}=================================================={RESET}", 0.002
    )
    print_slow(
        f"{CYAN}    WELCOME TO HYDROLOOM UNSUPERVISED SANDBOX     {RESET}", 0.002
    )
    print_slow(
        f"{CYAN}=================================================={RESET}", 0.002
    )
    print_slow(f"{YELLOW}Loading datasets and training KMeans model...{RESET}")

    try:
        north_df = pl.read_parquet("data/processed/north_final.parquet")
        south_df = pl.read_parquet("data/processed/south_final.parquet")
    except Exception as e:
        print_slow(f"{RED}Error loading data: {e}{RESET}")
        return

    # Train models
    kmeans_north = KMeans(n_clusters=4, random_state=2033, n_init="auto")
    kmeans_north.fit(north_df.to_numpy().astype(np.float64))
    kmeans_north_labels = get_cluster_labels(kmeans_north)

    kmeans_south = KMeans(n_clusters=4, random_state=2033, n_init="auto")
    kmeans_south.fit(south_df.to_numpy().astype(np.float64))
    kmeans_south_labels = get_cluster_labels(kmeans_south)

    print_slow(
        f"{GREEN}Models trained successfully on North and South Hemisphere data!{RESET}"
    )
    print_slow(
        "Your goal: Play with the 4 behavioral features to see how the model reacts."
    )
    print_slow("Can you discover the profile of each cluster or find an anomaly?")

    while True:
        print_slow(f"\n{CYAN}--- Select Hemisphere ---{RESET}")
        print("1. North Hemisphere")
        print("2. South Hemisphere")
        print("q. Quit")

        hemi_choice = input("> ")
        if hemi_choice.lower() == "q":
            print_slow(f"{MAGENTA}Thanks for playing!{RESET}")
            break
        elif hemi_choice == "1":
            df = north_df
            kmeans = kmeans_north
            cluster_labels = kmeans_north_labels
            hemi_name = "NORTH"
        elif hemi_choice == "2":
            df = south_df
            kmeans = kmeans_south
            cluster_labels = kmeans_south_labels
            hemi_name = "SOUTH"
        else:
            print(f"{RED}Invalid choice.{RESET}")
            continue

        mins = df.min()
        maxs = df.max()
        features = [
            "log_per_capita_usage",
            "dry_day_spike_factor",
            "efficiency_penalty_ratio",
            "landscape_demand_index",
        ]

        while True:
            print(
                f"\n{CYAN}--- {hemi_name} HEMISPHERE: Current Feature Inputs ---{RESET}"
            )
            user_vals = []

            print("Choose an option:")
            print("1. Enter manual feature values")
            print("2. Generate random valid profile")
            print("3. Generate extreme outlier profile (Break the system!)")
            print("b. Back to Hemisphere Selection")
            print("q. Quit")

            choice = input("> ")
            if choice.lower() == "q":
                print_slow(f"{MAGENTA}Thanks for playing!{RESET}")
                return
            elif choice.lower() == "b":
                break
            elif choice == "2":
                user_vals = generate_random_profile(mins, maxs)
            elif choice == "3":
                # Create a completely crazy profile
                user_vals = [
                    np.random.uniform(10.0, 50.0),  # Impossible log usage
                    np.random.uniform(10.0, 20.0),
                    np.random.uniform(-5.0, 0.0),  # Negative efficiency penalty
                    np.random.uniform(2.0, 5.0),
                ]
            elif choice == "1":
                for feat in features:
                    val_min = mins[feat][0]
                    val_max = maxs[feat][0]
                    while True:
                        try:
                            val = input(
                                f"Enter {feat} (typical range {val_min:.2f} to {val_max:.2f}): "
                            )
                            user_vals.append(float(val))
                            break
                        except ValueError:
                            print(f"{RED}Invalid input. Please enter a number.{RESET}")
            else:
                print(f"{RED}Invalid choice.{RESET}")
                continue

            print(f"\nTesting profile: {user_vals}")

            # Predict
            user_vals_arr = np.array([user_vals], dtype=np.float64)
            prediction = kmeans.predict(user_vals_arr)[0]
            distances = kmeans.transform(user_vals_arr)[0]
    
            print_slow(f"\n{YELLOW}Analyzing...{RESET}", 0.02)
            print_slow(
                f"{GREEN}>> The model assigned this profile to CLUSTER {prediction}: {cluster_labels[prediction]} <<{RESET}"
            )
    
            # Display distances to each cluster centroid
            print("\nDistance to each cluster centroid:")
            for i, dist in enumerate(distances):
                # Scale bar for visual effect
                bar_len = int(max(1, 20 - dist * 2))
                bar = "█" * (bar_len if bar_len > 0 else 1)
                label = cluster_labels[i]
                if i == prediction:
                    print(f"{GREEN}Cluster {i} ({label}): {dist:.2f} | {bar}{RESET}")
                else:
                    print(f"{CYAN}Cluster {i} ({label}): {dist:.2f} | {bar}{RESET}")
    
            # Discovering failures / edge cases
            min_dist = min(distances)
            if min_dist > 5.0:
                print_slow(f"\n{RED}🚨 ANOMALY DETECTED! 🚨{RESET}")
                print_slow(
                    f"{RED}This profile is extremely far (Distance: {min_dist:.2f}) from all known clusters.{RESET}"
                )
                print_slow(
                    f"{RED}You discovered a failure mode / edge case in the unsupervised service!{RESET}"
                )
                print_slow(
                    f"{RED}The KMeans model forced an assignment to Cluster {prediction}, but it clearly does not belong.{RESET}"
                )
    
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
