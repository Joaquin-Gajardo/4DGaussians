import json
import os

import pandas as pd

def summarize_results(base_folder):
    # List to store results
    results = []

    # Iterate through each subfolder in the base folder
    for scene_folder in os.listdir(base_folder):
        scene_path = os.path.join(base_folder, scene_folder)
        
        # Check if it's a directory
        if os.path.isdir(scene_path):
            json_path = os.path.join(scene_path, 'results.json')
            
            # Check if the JSON file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract mean values
                data = data.get('ours_14000', {})
                result = {
                    'Scene': scene_folder,
                    'PSNR': data.get('PSNR', 'N/A'),
                    'SSIM': data.get('SSIM', 'N/A'),
                    'LPIPS (vgg)': data.get('LPIPS-vgg', 'N/A')
                }
                results.append(result)
            else:
                print(f"Warning: No JSON file found for scene {scene_folder}")

    # Create a DataFrame
    df = pd.DataFrame(results).sort_values('Scene')

    # Calculate overall mean
    mean_row = pd.DataFrame({
        'Scene': ['Overall Mean'],
        'PSNR': [df['PSNR'].mean()],
        'SSIM': [df['SSIM'].mean()],
        'LPIPS (vgg)': [df['LPIPS (vgg)'].mean()]
    })

    # Concatenate the mean row to the original DataFrame
    df = pd.concat([df, mean_row], ignore_index=True)

    # Set 'Scene' as index for better display
    df.set_index('Scene', inplace=True)

    return df

method="def4DGS-3"
base_folder = f"../../output/WAT/{method}"
summary_table = summarize_results(base_folder)

print(summary_table.to_string())

summary_table.to_csv(f"WAT_{method}_results_summary.csv")