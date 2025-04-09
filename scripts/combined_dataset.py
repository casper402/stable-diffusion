import os
import csv
import argparse

def find_and_save_pairs(ct_root, cbct_root, output_manifest_file):
    """
    Scans CT and CBCT directories, finds corresponding slice pairs based on naming
    convention (volume-X -> REC-X, slice_Y -> slice_Y), and saves them to a CSV manifest file.

    Args:
        ct_root (str): Path to the root directory containing CT volumes (e.g., volume-0).
        cbct_root (str): Path to the root directory containing CBCT recordings (e.g., REC-0).
        output_manifest_file (str): Path where the CSV manifest file will be saved.
    """
    paired_paths = []
    print(f"Starting scan for CT volumes in: {ct_root}")

    try:
        ct_volumes = sorted([
            d for d in os.listdir(ct_root)
            if os.path.isdir(os.path.join(ct_root, d)) and d.startswith('volume-')
        ])
        print(f"Found {len(ct_volumes)} potential CT volumes.")
    except FileNotFoundError:
        print(f"CT path not found: {ct_root}")
        return
    except Exception as e:
        print(f"Error listing CT volumes in {ct_root}: {e}")
        return

    if not ct_volumes:
        print("No directories starting with 'volume-' found in CT path.")
        return

    # Use tqdm for progress bar if installed
    for ct_volume_name in ct_volumes:
        try:
            volume_num_str = ct_volume_name.split('-')[-1]
            # Assume exact matching naming convention
            cbct_rec_name = f"REC-{volume_num_str}"

            ct_volume_path = os.path.join(ct_root, ct_volume_name)
            cbct_rec_path = os.path.join(cbct_root, cbct_rec_name)

            # Optional: Add a check here if you want robustness against missing CBCT dirs
            # if not os.path.isdir(cbct_rec_path):
            #     logging.warning(f"Missing corresponding CBCT directory: {cbct_rec_path}. Skipping {ct_volume_name}")
            #     continue

            ct_slices = sorted([
                f for f in os.listdir(ct_volume_path)
                if os.path.isfile(os.path.join(ct_volume_path, f))
            ])

            if not ct_slices:
                 print(f"No slice files found in {ct_volume_path}. Skipping.")
                 continue

            for slice_name in ct_slices:
                ct_slice_path = os.path.join(ct_volume_path, slice_name)
                cbct_slice_path = os.path.join(cbct_rec_path, slice_name)

                # Optional: Add check if you want robustness against missing CBCT slices
                # if not os.path.isfile(cbct_slice_path):
                #     logging.warning(f"Missing corresponding CBCT slice: {cbct_slice_path}. Skipping.")
                #     continue

                # Store the absolute paths for clarity, though relative might work
                paired_paths.append((os.path.abspath(cbct_slice_path), os.path.abspath(ct_slice_path)))

        except FileNotFoundError:
            print(f"Error accessing expected directory/file within {ct_volume_name} or {cbct_rec_name}. Skipping volume.")
            continue
        except Exception as e:
            print(f"Unexpected error processing volume {ct_volume_name}: {e}. Skipping.")
            continue

    print(f"Found {len(paired_paths)} paired slices.")

    # Save to CSV
    try:
        with open(output_manifest_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['CBCT_Path', 'CT_Path'])  # Write header
            writer.writerows(paired_paths)
        print(f"Successfully saved manifest file to: {output_manifest_file}")
    except IOError as e:
        print(f"Error writing manifest file {output_manifest_file}: {e}")
    except Exception as e:
        print(f"Unexpected error saving manifest file: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Set your actual paths here or use command-line arguments
    CT_DATA_ROOT = "../training_data/CT"
    CBCT_DATA_ROOT = "../training_data/CBCT"
    MANIFEST_FILE_PATH = "/training_data/dataset_manifest.csv"
    # -------------------

    # # --- Alternative: Using command-line arguments ---
    # parser = argparse.ArgumentParser(description="Create a manifest file for paired CBCT/CT slices.")
    # parser.add_argument("--ct_dir", required=True, help="Path to the root CT directory (containing volume-X folders)")
    # parser.add_argument("--cbct_dir", required=True, help="Path to the root CBCT directory (containing REC-X folders)")
    # parser.add_argument("--output_csv", required=True, help="Path to save the output manifest CSV file")
    # args = parser.parse_args()
    # find_and_save_pairs(args.ct_dir, args.cbct_dir, args.output_csv)
    # # -------------------------------------------------

    # --- Run with hardcoded paths ---
    find_and_save_pairs(CT_DATA_ROOT, CBCT_DATA_ROOT, MANIFEST_FILE_PATH)
    # --------------------------------