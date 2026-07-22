def calculate_iou(prediction_box, ground_truth_box):
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = prediction_box
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = ground_truth_box

    intersection_xmin = max(pred_xmin, gt_xmin)
    intersection_ymin = max(pred_ymin, gt_ymin)
    intersection_xmax = min(pred_xmax, gt_xmax)
    intersection_ymax = min(pred_ymax, gt_ymax)

    intersection_width = max(
        0.0,
        intersection_xmax - intersection_xmin,
    )
    intersection_height = max(
        0.0,
        intersection_ymax - intersection_ymin,
    )

    intersection_area = intersection_width * intersection_height

    prediction_area = max(
        0.0,
        pred_xmax - pred_xmin,
    ) * max(
        0.0,
        pred_ymax - pred_ymin,
    )

    ground_truth_area = max(
        0.0,
        gt_xmax - gt_xmin,
    ) * max(
        0.0,
        gt_ymax - gt_ymin,
    )

    union_area = (
        prediction_area
        + ground_truth_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area




def match_predictions(
        predictions, 
        ground_truth, 
        iou_threshold=0.5
        ):
    
    predictions = sorted(
        predictions,
        key=lambda item:item["confidence"],
        reverse=True
    )

    matched_ground_truth = set()
    matches = []
    false_positives = []
    for prediction in predictions:
        best_iou = 0.0
        best_gt_index = None
        for gt_index , gt in enumerate(ground_truth):
            if gt_index in matched_ground_truth:
                continue
            

            iou = calculate_iou(
                prediction["box"],
                gt["box"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if(
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):
            matched_ground_truth.add(best_gt_index)

            matches.append({
                "prediction": prediction,
                "ground_truth": ground_truth[best_gt_index],
                "iou": best_iou
            })

        else:
            false_positives.append(prediction)

    false_negative = [
        gt
        for gt_index, gt in enumerate(ground_truth)
        if gt_index not in matched_ground_truth
    ]

    return matches, false_positives, false_negative