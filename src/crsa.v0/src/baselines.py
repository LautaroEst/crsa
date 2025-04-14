

def get_random_performance(prior):
    return 1 / len(prior)

def get_prior_performance(prior):
    return prior.max()


def get_baseline_performance(baseline, lexicon, prior):
    if baseline == "random":
        return get_random_performance(prior)
    elif baseline == "prior":
        return get_prior_performance(prior)
    else:
        raise ValueError(f"Unknown baseline {baseline}.")