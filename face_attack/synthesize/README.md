### Steps
1. Select the MMDA model, new parameters and the base-image you want to use. To do so, make changes to `synthesize_parameters.py`.
2. Just run `synthesize.py` to generate a new image with the given settings. 
---
*Note:* 

Synthesized images are saved in the directory: `SAVE_DIR_SYNTHESIZE`. The parameters along with the file name of the newly generated synthesized images are saved in the CSV file: `synthesize_history.csv`.

Low dimensional attributes for different modes are saved in the `MMDA_attr_vals_*.txt` files in this same directory for reference when you need to change the new parameters.   