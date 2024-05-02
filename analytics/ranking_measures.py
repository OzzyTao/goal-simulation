def stability(rankings_list, ground_truth_position=1):
    ## the rankings should be ordered based on time
    num = 0
    top_ranked_intention = -1
    for ranking in rankings_list:
        if isinstance(ranking, str):
            ranking = eval(ranking)
        this_top_ranked_intention = ranking.index(0)
        if this_top_ranked_intention == top_ranked_intention:
            num +=1
        else:
            top_ranked_intention = this_top_ranked_intention
    return num/(rankings_list.size-1) if rankings_list.size>1 else 0

def correctness(rankings_list, margin=0, ground_truth_position=1):
    num = 0
    for ranking in rankings_list:
        if isinstance(ranking, str):
            ranking = eval(ranking)
        if ranking[ground_truth_position]<=margin:
            num += 1
    return num/rankings_list.size

def final_correctness(rankings_list, margin=0, ground_truth_position=1):
    ranking =rankings_list.iloc[-1]
    if isinstance(ranking, str):
        ranking = eval(ranking)
    if int(ranking[ground_truth_position])<=margin:
        return 1
    else:
        return 0


def first_correct(rankings_list, margin=0, ground_truth_position=1):
    index = -1
    for i,ranking in enumerate(rankings_list):
        if isinstance(ranking, str):
            ranking = eval(ranking)
        if ranking[ground_truth_position]<=margin:
            index = i 
            break
    return index/rankings_list.size

def last_change(rankings_list, ground_truth_position=1):
    observations = rankings_list.size
    for i in range(observations):
        ranking = rankings_list.iloc[-i-1]
        if isinstance(ranking, str):
            ranking = eval(ranking)
        rank = ranking[ground_truth_position]
        if rank != 0:
            return i/observations
    return 1