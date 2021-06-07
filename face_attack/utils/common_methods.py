import csv


def write_to_csv(csv_file_path, fields):
    ## WRITE NEW PARAMETERS IN CSV ##
    with open(csv_file_path, 'a', newline='') as f:  # Save the parameters in CSV
        # fields = [_time] + np.squeeze(new_attr).tolist() + [closest_face_in_gallery] + [min_dist]
        # print(fields)
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fields)
