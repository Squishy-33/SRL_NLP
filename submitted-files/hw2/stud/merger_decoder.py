
"""
The original sentence merger is way more complicate than this one
The original merger written in notebook handles merging operation for all sentences at once
Following method extracts predicate positions in sentence
Then for each predicate, finds its prediction, and put them together 
"""
def sentence_merger(predictions, predPos):

    # Extract predicate positions
    # For example if the sentence has four predicates
    # The value for 'position_pred_flatten' would be [8, 12, 32, 45]
    position_predicate = [batch['position_predicate'] for batch in predPos]
    position_pred_flatten = [xx for x in position_predicate for xx in x]

    pred_flatten = [xx for x in predictions for xx in x]

    # Sentence Merger
    roles_prediction = {}
    tmp_dict = {}

    for i, item in enumerate(pred_flatten):
        tmp_dict[position_pred_flatten[i]] = pred_flatten[i]

    roles_prediction['roles'] = tmp_dict
    return roles_prediction

"""
For each predicate index in prediction 
replace the ids with the actual labels
"""
def sentence_decoder(predictions, id2vocabs):
    pred = predictions['roles']
    for idx in pred.keys():
        decoded_sentence = [id2vocabs[item.item()] for item in pred[idx]]
        pred[idx] = decoded_sentence
    return predictions
