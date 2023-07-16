def top_prob(probs, ground_truth_position=1):
    return max(probs)

def true_prob(probs, ground_truth_position=1):
    return probs[ground_truth_position]