
# im1 - later frame, im2 - earlier frame
def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    f1_norms = np.linalg.norm(im1_features, axis=1)
    f2_norms = np.linalg.norm(im2_features, axis=1)
    A = np.reshape(np.square(f1_norms), (-1, 1)) + np.reshape(np.square(f2_norms), (1, -1))
    B = 2 * im1_features @ im2_features.T
    D = np.sqrt(A - B)

    num_matches = im1_features.shape[0]
    first_closest_indices = np.argmin(D, axis=1)
    first_closest_distances = D[np.arange(num_matches), first_closest_indices]
    D[np.arange(num_matches), first_closest_indices] = np.nan
    second_closest_indices = np.nanargmin(D, axis=1)
    second_closest_distances = D[np.arange(num_matches), second_closest_indices]

    matches = np.stack((np.arange(num_matches), first_closest_indices), axis=-1)
    confidences = second_closest_distances / first_closest_distances # inverse of nndr

    return matches, confidences