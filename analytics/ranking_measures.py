def stability(rankings_list):
    ## the rankings should be ordered based on time
    num = 0
    current_ranking = []
    for ranking in rankings_list:
        if ranking!=current_ranking:
            num +=1
            current_ranking = ranking
    return 1/num

def correctness(rankings_list, margin=0, ground_truth_position=1):
    num = 0
    for ranking in rankings_list:
        if ranking[ground_truth_position]<=margin:
            num += 1
    return num/rankings_list.size


def first_correct(rankings_list, margin=0, ground_truth_position=1):
    index = -1
    for i,ranking in enumerate(rankings_list):
        if ranking[ground_truth_position]<=margin:
            index = i 
            break
    return index/rankings_list.size

def last_change(rankings_list, ground_truth_position=1):
    observations = rankings_list.size
    last_rank = -1
    for i in range(observations):
        rank = rankings_list.iloc[-i-1][ground_truth_position]
        if last_rank == -1:
            last_rank = rank 
        else:
            if rank != last_rank:
                return (observations-i-1)/observations
    return 0