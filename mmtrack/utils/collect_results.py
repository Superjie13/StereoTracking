def save_prediction_results(file_path: str):
    """Save the prediction results to a file."""
    import csv
    import os
    def decorator(predict_func):

        # if file exists, delete it
        if os.path.exists(file_path):
            os.remove(file_path)
            
        def wrapper(*args, **kwargs):
            results = predict_func(*args, **kwargs)

            # parse the results
            track_instance = results[0].pred_track_instances
            frame_id = results[0].frame_id
            bboxes = track_instance.get('bboxes').cpu().numpy()
            instances_id = track_instance.get('instances_id').cpu().numpy()
            labels = track_instance.get('labels').cpu().numpy()
            scores = track_instance.get('scores').cpu().numpy()
            depth  = track_instance.get('depth')
            gt_depth  = track_instance.get('gt_depth')

            tracks = []
            for iid, label, box, d, gt_d, s in zip(
                    instances_id, labels, bboxes, depth, gt_depth, scores):
                tracks.append([frame_id, iid, label, *box, d, gt_d, s])

            if file_path.endswith('.csv'):
                with open(file_path, 'a') as f:
                    writer = csv.writer(f)
                    # if file is empty, write the header
                    # move the cursor to the end of file
                    f.seek(0, os.SEEK_END)
                    if f.tell() == 0:  # if the file is empty
                        writer.writerow(['frame', 'id', 'label', 'tl_x', 'tl_y', 'br_x', 'br_y', 'depth', 'gt_depth', 'score'])
                    writer.writerows(tracks)
            else:
                raise ValueError('The saving format is not supported.')
            
            return results
        
        return wrapper
    return decorator