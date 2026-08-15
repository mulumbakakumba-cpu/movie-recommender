def precision_recall_f1_at_k(
    recommended_ids,
    relevant_ids,
    k=10
):
    recommended_ids = list(
        recommended_ids
    )[:k]

    relevant_ids = set(
        relevant_ids
    )

    if not recommended_ids:
        return 0.0, 0.0, 0.0

    hits = len(
        set(recommended_ids)
        & relevant_ids
    )

    precision = (
        hits / len(recommended_ids)
    )

    recall = (
        hits / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1